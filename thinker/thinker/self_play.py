import time, timeit
from collections import namedtuple
import numpy as np
import argparse
import traceback
import torch
from torch import nn
from torch.nn import functional as F
import ray
from thinker.buffer import ActorBuffer, ModelBuffer, GeneralBuffer
from thinker.buffer import AB_CAN_WRITE, AB_FULL, AB_FINISH
from thinker.net import ActorNet, ModelNet, ActorOut
from thinker.env import Environment, EnvOut
from thinker.learn_model import SModelLearner
from thinker.learn_actor import SActorLearner
import thinker.util as util

# from torchvision.transforms import ToPILImage
# from PIL import Image

_fields = tuple(
    item
    for item in ActorOut._fields + EnvOut._fields
    if item not in ["baseline_enc", "reg_loss"]
)
TrainActorOut = namedtuple("TrainActorOut", _fields + ("id",))
TrainModelOut = namedtuple(
    "TrainModelOut",
    [
        "gym_env_out",
        "policy_logits",
        "action",
        "reward",
        "done",
        "truncated_done",
        "baseline",
    ],
)
PO_NET, PO_MODEL, PO_NSTEP = 0, 1, 2


@ray.remote
class SelfPlayWorker:
    def __init__(
        self,
        buffers: dict,
        policy: int,
        policy_params: dict,
        rank: int,
        num_p_actors: int,
        flags: argparse.Namespace,
        base_seed: int = 1,
    ):
        self._logger = util.logger()
        if not flags.disable_cuda and torch.cuda.is_available() and num_p_actors > 1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.actor_buffer = buffers["actor"]
        self.model_buffer = buffers["model"]
        self.actor_param_buffer = buffers["actor_param"]
        self.model_param_buffer = buffers["model_param"]
        self.signal_buffer = buffers["signal"]
        self.test_buffer = buffers["test"]

        self._logger.info(
            "Initalizing actor %d with device %s %s"
            % (
                rank,
                "cuda" if self.device == torch.device("cuda") else "cpu",
                "(test mode)" if self.test_buffer is not None else "",
            )
        )

        self.policy = policy
        self.policy_params = policy_params
        self.greedy_policy = (
            self.policy_params is not None
            and "greedy" in self.policy_params
            and self.policy_params["greedy"]
        )
        self.rank = rank
        self.num_p_actors = num_p_actors
        self.flags = flags
        self.model_wrap = policy == PO_NET and not flags.disable_model
        self.timing = util.Timings()
        self.actor_id = (
            torch.arange(self.num_p_actors, device=self.device)
            + self.rank * self.num_p_actors
        ).unsqueeze(0)
        self.time = self.rank == 0 and flags.profile

        self.env = Environment(
            flags, model_wrap=self.model_wrap, env_n=num_p_actors, device=self.device
        )
        seed = [base_seed + i + num_p_actors * rank for i in range(num_p_actors)]
        self.env.seed(seed)

        if self.policy == PO_NET and not self.flags.merge_play_actor:
            self.actor_net = ActorNet(
                obs_shape=self.env.model_out_shape if self.model_wrap else None,
                gym_obs_shape=self.env.gym_env_out_shape,
                num_actions=self.env.num_actions,
                flags=flags,
            )
            self.actor_net.to(self.device)
            self.actor_net.train(False)

        if not self.flags.merge_play_model and not self.flags.disable_model:
            self.model_net = ModelNet(
                obs_shape=self.env.gym_env_out_shape,
                num_actions=self.env.num_actions,
                flags=flags,
            )
            self.model_net.train(False)
            self.model_net.to(self.device)

        # the networks weight are set by the respective learner; but if the respective
        # learner does not exist, then rank 0 worker will set the weights
        if (
            rank == 0
            and not self.flags.train_actor
            and self.policy == PO_NET
            and not self.flags.merge_play_actor
        ):
            if self.flags.preload_actor:
                checkpoint = torch.load(
                    self.flags.preload_actor, map_location=torch.device("cpu")
                )
                self.actor_net.set_weights(checkpoint["actor_net_state_dict"])
                self._logger.info(
                    "Loadded actor network from %s" % self.flags.preload_actor
                )
            self.actor_param_buffer.set_data.remote(
                "actor_net", self.actor_net.get_weights()
            )

        if (
            rank == 0
            and not self.flags.train_model
            and not self.flags.merge_play_model
            and not self.flags.disable_model
        ):
            if self.flags.preload_model:
                checkpoint = torch.load(
                    self.flags.preload_model, map_location=torch.device("cpu")
                )
                self.model_net.set_weights(
                    checkpoint["model_state_dict"]
                    if "model_state_dict" in checkpoint
                    else checkpoint["model_net_state_dict"]
                )
                self._logger.info(
                    "Loadded model network from %s" % self.flags.preload_model
                )
            self.model_param_buffer.set_data.remote(
                "model_net", self.model_net.get_weights()
            )

        # synchronize weight before start
        if not self.flags.merge_play_actor and self.policy == PO_NET:
            while True:
                weights = ray.get(
                    self.actor_param_buffer.get_data.remote("actor_net")
                )  # set by actor_learner
                if weights is not None:
                    self.actor_net.set_weights(weights)
                    break
                time.sleep(0.1)

        if not self.flags.merge_play_model and not self.flags.disable_model:
            while True:
                weights = ray.get(
                    self.model_param_buffer.get_data.remote("model_net")
                )  # set by rank 0 self_play_worker or model learner
                if weights is not None:
                    self.model_net.set_weights(weights)
                    break
                time.sleep(0.1)

        if self.flags.merge_play_model and not self.flags.disable_model and rank == 0:
            self.model_learner = SModelLearner(buffers=buffers, rank=0, flags=flags)
            self.model_net = self.model_learner.model_net
            self.model_net.train(False)

        if flags.train_model and not self.flags.disable_model:
            self.model_local_buffer = [
                self.empty_model_buffer(),
                self.empty_model_buffer(),
            ]
            self.model_n = 0
            self.model_t = 0

        if self.flags.merge_play_actor and rank == 0:
            self.actor_learner = SActorLearner(buffers=None, rank=0, flags=flags)
            self.actor_net = self.actor_learner.actor_net
            self.actor_net.train(False)

        if self.rank == 0:
            if self.policy == PO_NET:
                self._logger.info(
                    "Actor network size: %d"
                    % sum(p.numel() for p in self.actor_net.parameters())
                )
            if not self.flags.disable_model:
                self._logger.info(
                    "Model network size: %d"
                    % sum(p.numel() for p in self.model_net.parameters())
                )

    def gen_data(self, test_eps_n: int = 0, verbose: bool = True):
        """Generate self-play data
        Args:
            test_eps_n (int): number of episode to test for (only for testing mode);
            if set to non-zero, the worker will stop once reaching test_eps_n episodes
            and the data will not be sent out to model or actor buffer
            verbose (bool): whether to print output
        """
        try:
            if verbose:
                self._logger.info(
                    "Actor %d started. %s"
                    % (self.rank, "(test mode)" if test_eps_n > 0 else "")
                )
            n = 0
            if test_eps_n > 0:
                self.test_eps_done_n = torch.zeros(
                    self.num_p_actors, device=self.device
                )

            if self.policy == PO_NET and self.model_wrap:
                env_out = self.env.initial(self.model_net)
            else:
                env_out = self.env.initial()

            """
            for i in range(self.num_p_actors):
                tensor = env_out.gym_env_out.cpu()[0,i]
                to_pil = ToPILImage()
                pil_img = to_pil(tensor)
                pil_img.save("tmp/%d.png" % (i + self.rank * self.num_p_actors))
            """

            if self.policy == PO_NET:
                actor_state = self.actor_net.initial_state(
                    batch_size=self.num_p_actors, device=self.device
                )
                actor_out, _ = self.actor_net(env_out, actor_state)
            elif self.policy == PO_MODEL:
                actor_out = self.po_model(env_out, self.model_net)
                action = actor_out.action
                actor_state = None
            elif self.policy == PO_NSTEP:
                actor_out = self.po_nstep(self.env, self.model_net)
                action = actor_out.action
                actor_state = None

            train_actor = (
                self.flags.train_actor and self.policy == PO_NET and test_eps_n == 0
            )
            train_model = self.flags.train_model and test_eps_n == 0
            # config for preloading before actor network start learning
            preload_needed = self.flags.train_model and self.flags.load_checkpoint
            preload = False
            learner_actor_start = train_actor and (not preload_needed or preload)

            timer = timeit.default_timer
            start_time = timer()

            if not self.flags.merge_play_model:
                signal_ptr = self.signal_buffer.get_data.remote("self_play_signals")

            while True:
                with torch.set_grad_enabled(False):
                    if self.time:
                        self.timing.reset()
                    # prepare train_actor_out data to be written
                    initial_actor_state = actor_state
                    if learner_actor_start:
                        self.write_actor_buffer(env_out, actor_out, 0)
                    if self.time:
                        self.timing.time("misc1")

                    for t in range(self.flags.unroll_length):
                        # generate action
                        if self.policy == PO_NET:
                            # policy from applying actor network on the model-wrapped environment
                            if self.flags.float16 and self.device == torch.device(
                                "cuda"
                            ):
                                with torch.autocast(
                                    device_type="cuda", dtype=torch.float16
                                ):
                                    actor_out, actor_state = self.actor_net(
                                        env_out, actor_state, greedy=self.greedy_policy
                                    )
                            else:
                                actor_out, actor_state = self.actor_net(
                                    env_out, actor_state, greedy=self.greedy_policy
                                )
                            if self.time:
                                self.timing.time("actor_net")
                            if not self.flags.disable_model:
                                action = [
                                    actor_out.action,
                                    actor_out.im_action,
                                    actor_out.reset_action,
                                ]
                                action = torch.cat(
                                    [a.unsqueeze(-1) for a in action], dim=-1
                                )
                            else:
                                action = actor_out.action.unsqueeze(-1)
                        elif self.policy == PO_MODEL:
                            # policy directly from the model
                            actor_out = self.po_model(env_out, self.model_net)
                            action = actor_out.action
                        elif self.policy == PO_NSTEP:
                            actor_out = self.po_nstep(self.env, self.model_net)
                            action = actor_out.action
                        if self.policy == PO_NET and self.model_wrap:
                            if self.flags.float16 and self.device == torch.device(
                                "cuda"
                            ):
                                with torch.autocast(
                                    device_type="cuda", dtype=torch.float16
                                ):
                                    env_out = self.env.step(action, self.model_net)
                            else:
                                env_out = self.env.step(action, self.model_net)
                        else:
                            env_out = self.env.step(action)

                        if self.time:
                            self.timing.time("step env")
                        # write the data to the respective buffers
                        if learner_actor_start:
                            self.write_actor_buffer(env_out, actor_out, t + 1)
                        if self.time:
                            self.timing.time("finish actor buffer")
                        if train_model and (
                            self.policy != PO_NET or env_out.cur_t[:, 0] == 0
                        ):
                            baseline = None
                            if self.policy == PO_NET:
                                if self.flags.model_bootstrap_type == 0:
                                    baseline = self.env.env.baseline_mean_q
                                elif self.flags.model_bootstrap_type == 1:
                                    baseline = self.env.env.baseline_max_q
                                elif self.flags.model_bootstrap_type == 2:
                                    baseline = actor_out.baseline[:, :, 0] / (
                                        self.flags.discounting
                                        ** ((self.flags.rec_t - 1) / self.flags.rec_t)
                                    )
                            self.write_send_model_buffer(env_out, actor_out, baseline)
                        if self.time:
                            self.timing.time("write send model buffer")
                        if test_eps_n > 0:
                            finish, all_returns = self.write_test_buffer(
                                env_out, actor_out, test_eps_n, verbose
                            )
                            if finish:
                                return all_returns
                    if learner_actor_start and not self.flags.merge_play_actor:
                        # send the data to remote actor buffer
                        initial_actor_state = util.tuple_map(
                            initial_actor_state, lambda x: x.cpu().numpy()
                        )
                        status = 0
                        if self.time:
                            self.timing.time("mics2")
                        while True:
                            data_full_ptr = self.actor_buffer.get_status.remote()
                            status = ray.get(data_full_ptr)
                            if status == AB_FULL:
                                time.sleep(0.1)
                            else:
                                break
                            if status == AB_FINISH:
                                return True
                        self.actor_buffer.write.remote(
                            ray.put(self.actor_local_buffer),
                            ray.put(initial_actor_state),
                        )
                        if self.time:
                            self.timing.time("send actor buffer")
                    # if preload buffer needed, check if preloaded
                    if train_actor and preload_needed and not preload:
                        preload, tran_n = ray.get(
                            self.model_buffer.check_preload.remote()
                        )
                        if self.rank == 0:
                            if preload:
                                self._logger.info("Finish preloading")
                                ray.get(self.model_buffer.set_preload.remote())
                            else:
                                self._logger.info(
                                    "Preloading: %d/%d"
                                    % (tran_n, self.flags.model_buffer_n)
                                )
                        learner_actor_start = not preload_needed or preload
                    if self.time:
                        self.timing.time("mics3")

                    # Signal control for all self-play threads (only when it is not in testing mode)
                    # note that the signal control is only between learn_model and self_play (not learn_actor)
                    if (
                        test_eps_n == 0
                        and n % 1 == 0
                        and not self.flags.merge_play_model
                    ):
                        signals = ray.get(signal_ptr)
                        signal_ptr = self.signal_buffer.get_data.remote(
                            "self_play_signals"
                        )
                        if (
                            signals is not None
                            and "term" in signals
                            and signals["term"]
                        ):
                            return True
                        while (
                            signals is not None
                            and "halt" in signals
                            and signals["halt"]
                        ):
                            time.sleep(0.1)
                            signals = ray.get(signal_ptr)
                            signal_ptr = self.signal_buffer.get_data.remote(
                                "self_play_signals"
                            )
                            if (
                                signals is not None
                                and "term" in signals
                                and signals["term"]
                            ):
                                return True
                    if self.time:
                        self.timing.time("signal control")

                if self.flags.merge_play_actor:
                    data = (self.actor_local_buffer, initial_actor_state)
                    self.actor_net.train(True)
                    self.actor_learner.consume_data(data)
                    self.actor_net.train(False)

                if self.flags.merge_play_model:
                    torch.cuda.empty_cache()
                    self.model_net.train(True)
                    self.model_learner.s_learn_data(
                        timing=self.timing if self.time else None
                    )
                    self.model_net.train(False)

                # update model weight
                if (
                    n % self.flags.splay_actor_update_freq == 0
                    and self.flags.train_actor
                    and self.policy == PO_NET
                    and not self.flags.merge_play_actor
                ):
                    actor_weights_ptr = self.actor_param_buffer.get_data.remote(
                        "actor_net"
                    )
                    weights = ray.get(actor_weights_ptr)
                    self.actor_net.set_weights(weights)
                    del weights
                if self.time:
                    self.timing.time("update actor net weight")
                if (
                    n % self.flags.splay_model_update_freq == 0
                    and self.flags.train_model
                    and not self.flags.merge_play_model
                ):
                    model_weights_ptr = self.model_param_buffer.get_data.remote(
                        "model_net"
                    )
                    weights = ray.get(model_weights_ptr)
                    self.model_net.set_weights(weights)
                    del weights
                if self.time:
                    self.timing.time("update model net weight")

                n += 1
                if self.time and timer() - start_time > 5:
                    self._logger.info(self.timing.summary())
                    start_time = timer()

        except Exception as e:
            self._logger.error(f"Exception detected in self_play: {e}")
            self._logger.error(traceback.format_exc())
        finally:
            return True

    def write_actor_buffer(self, env_out: EnvOut, actor_out: ActorOut, t: int):
        # write local

        if t == 0:
            if not self.flags.merge_play_actor:
                id = self.actor_id
            else:
                id = [self.actor_id[0]]
            fields = {"id": id}
            for field in TrainActorOut._fields:
                if field in ["id"]:
                    continue
                out = getattr(env_out if field in EnvOut._fields else actor_out, field)
                fields[field] = None
                if out is None:
                    continue
                if (
                    not self.flags.perfect_model
                    and not self.flags.disable_model
                    and field in ["gym_env_out"]
                ):
                    continue
                if self.flags.actor_see_type < 0 and field in [
                    "gym_env_out",
                    "model_encodes",
                ]:
                    continue
                if self.flags.actor_see_type >= 1 and field == "gym_env_out":
                    continue
                if not self.flags.merge_play_actor:
                    fields[field] = torch.empty(
                        size=(self.flags.unroll_length + 1, self.num_p_actors)
                        + out.shape[2:],
                        dtype=out.dtype,
                        device=self.device,
                    )
                else:
                    fields[field] = []
                # each is in the shape of (T x B xdim_1 x dim_2 ...)
            self.actor_local_buffer = TrainActorOut(**fields)

        for field in TrainActorOut._fields:
            v = getattr(self.actor_local_buffer, field)
            if v is not None and field not in ["id"]:
                new_val = getattr(
                    env_out if field in EnvOut._fields else actor_out, field
                )[0]
                if not self.flags.merge_play_actor:
                    v[t] = new_val
                else:
                    v.append(new_val)

        if self.time:
            self.timing.time("write_actor_buffer")
        if t == self.flags.unroll_length:
            # post-processing
            if not self.flags.merge_play_actor:
                self.actor_local_buffer = util.tuple_map(
                    self.actor_local_buffer, lambda x: x.cpu().numpy()
                )
            else:
                self.actor_local_buffer = util.tuple_map(
                    self.actor_local_buffer, lambda x: torch.stack(x, dim=0)
                )
        if self.time:
            self.timing.time("move_actor_buffer_to_cpu")

    def empty_model_buffer(self):
        pre_shape = (
            self.flags.model_unroll_length + 2 * self.flags.model_k_step_return,
            self.num_p_actors,
        )
        return TrainModelOut(
            gym_env_out=torch.zeros(
                pre_shape + self.env.gym_env_out_shape,
                dtype=torch.uint8,
                device=self.device,
            ),
            policy_logits=torch.zeros(
                pre_shape + (self.env.num_actions,),
                dtype=torch.float32,
                device=self.device,
            ),
            action=torch.zeros(pre_shape, dtype=torch.long, device=self.device),
            reward=torch.zeros(pre_shape, dtype=torch.float32, device=self.device),
            done=torch.ones(pre_shape, dtype=torch.bool, device=self.device),
            truncated_done=torch.ones(pre_shape, dtype=torch.bool, device=self.device),
            baseline=torch.zeros(pre_shape, dtype=torch.float32, device=self.device),
        )

    def write_single_model_buffer(
        self,
        n: int,
        t: int,
        env_out: EnvOut,
        actor_out: ActorOut,
        baseline: torch.tensor,
    ):
        self.model_local_buffer[n].gym_env_out[t] = env_out.gym_env_out[0]
        self.model_local_buffer[n].reward[t] = env_out.reward[0, :, 0]
        self.model_local_buffer[n].done[t] = env_out.done[0]
        self.model_local_buffer[n].truncated_done[t] = env_out.truncated_done[0]
        self.model_local_buffer[n].policy_logits[t] = actor_out.policy_logits[0]
        self.model_local_buffer[n].action[t] = actor_out.action[0]
        if baseline is not None:
            self.model_local_buffer[n].baseline[t] = baseline

    def write_send_model_buffer(
        self, env_out: EnvOut, actor_out: ActorOut, baseline: torch.tensor
    ):
        n, t, cap_t, k = (
            self.model_n,
            self.model_t,
            self.flags.model_unroll_length,
            self.flags.model_k_step_return,
        )
        self.write_single_model_buffer(n, t, env_out, actor_out, baseline)

        if t >= cap_t:
            # write the beginning of another buffer
            self.write_single_model_buffer(
                1 - n, t - cap_t, env_out, actor_out, baseline
            )

        if t >= cap_t + 2 * k - 2:
            # finish writing a buffer, send it
            send_model_local_buffer = util.tuple_map(
                self.model_local_buffer[n], lambda x: x.cpu().numpy()
            )
            self.model_buffer.write.remote(ray.put(send_model_local_buffer), self.rank)
            self.model_local_buffer[n] = self.empty_model_buffer()
            self.model_n = 1 - n
            self.model_t = t - cap_t + 1
        else:
            self.model_t += 1

    def write_test_buffer(
        self,
        env_out: EnvOut,
        actor_out: ActorOut,
        test_eps_n: int = 0,
        verbose: bool = True,
    ):
        mask = env_out.real_done[0] & (self.test_eps_done_n < test_eps_n)
        if torch.any(mask):
            episode_returns = env_out.episode_return[0, mask, 0]
            episode_returns = list(episode_returns.detach().cpu().numpy())
            self.test_eps_done_n += mask.float()
            for r in episode_returns:
                all_returns = ray.get(
                    self.test_buffer.extend_data.remote("episode_returns", [r])
                )
                all_returns = np.array(all_returns)
                if verbose:
                    if self.policy == 0:
                        if self.greedy_policy:
                            prefix = "Greedy Actor"
                        else:
                            prefix = "Actor"
                    else:
                        prefix = "Model"
                    log_str = "%s %d Mean (Std.) : %.4f (%.4f) - %.4f" % (
                        prefix,
                        len(all_returns),
                        np.mean(all_returns),
                        np.std(all_returns) / np.sqrt(len(all_returns)),
                        r,
                    )
                    if "Sokoban" in self.flags.env:
                        solved = (all_returns > 10).astype(float)
                        log_str += " Solve rate: %.4f (%.4f)" % (
                            np.mean(solved),
                            np.std(solved) / np.sqrt(len(solved)),
                        )
                    self._logger.info(log_str)
            if torch.all(self.test_eps_done_n >= test_eps_n):
                return True, all_returns

        return False, None

    def po_model(self, env_out, model_net):
        model_net_out = model_net(
            env_out.gym_env_out[0], env_out.last_action[:, :, 0], one_hot=False
        )
        policy_logits = model_net_out.logits
        action = torch.multinomial(
            F.softmax(policy_logits[0], dim=1), num_samples=1
        ).unsqueeze(0)
        actor_out = util.construct_tuple(
            ActorOut, policy_logits=policy_logits, action=action
        )
        # policy_logits has shape (T, B, num_actions)
        # action has shape (T, B, 1)
        return actor_out

    def po_nstep(self, env, model_net):
        discounting = self.flags.discounting
        if self.policy_params is not None:
            n = self.policy_params["n"]
            temp = self.policy_params["temp"]
        else:
            n, temp = 2, 0.01  # default policy param
        policy_logits, action, _ = self.nstep(env.env, model_net, discounting, n, temp)
        policy_logits = policy_logits.unsqueeze(0)
        action = action.unsqueeze(0)
        actor_out = util.construct_tuple(
            ActorOut, policy_logits=policy_logits, action=action
        )
        return actor_out

    def nstep(self, env, model_net, discounting, n, temp):
        with torch.no_grad():
            num_actions = env.action_space[0].n
            q_ret = torch.zeros(self.num_p_actors, num_actions)
            state = env.clone_state([n for n in range(self.num_p_actors)])
            for act in range(num_actions):
                action = torch.full(
                    size=(self.num_p_actors,),
                    fill_value=act,
                    dtype=torch.long,
                    device=self.device,
                )
                obs, reward, done, info = env.step(action.cpu().numpy())
                obs = torch.tensor(obs, device=self.device)
                reward = torch.tensor(reward)
                done = torch.tensor(done)
                if n > 1:
                    _, _, sub_q_ret = self.nstep(
                        env, model_net, discounting, n - 1, temp
                    )
                    ret = (
                        reward
                        + discounting * torch.max(sub_q_ret, dim=1)[0] * (~done).float()
                    )
                else:
                    model_net_out = model_net(
                        obs, action.unsqueeze(0).unsqueeze(-1), one_hot=False
                    )
                    ret = (
                        reward
                        + discounting * model_net_out.vs[0].cpu() * (~done).float()
                    )
                q_ret[:, act] = ret
                env.restore_state(state, [n for n in range(self.num_p_actors)])
            policy_logits = q_ret / temp
            prob = F.softmax(policy_logits, dim=1)
            action = torch.multinomial(prob, num_samples=1)
        return policy_logits, action, q_ret
