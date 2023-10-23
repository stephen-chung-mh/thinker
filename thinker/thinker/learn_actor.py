import time
import timeit
import os
import numpy as np
import argparse
import traceback
import ray
import torch
import torch.nn.functional as F

from thinker.core.vtrace import from_importance_weights, VTraceFromLogitsReturns
from thinker.core.file_writer import FileWriter
from thinker.net import ActorNet
from thinker.env import Environment
import thinker.util as util


def L2_loss(target_x, x):
    return (target_x - x) ** 2


def compute_baseline_loss(
    new_actor_out,
    ind,
    target_baseline,
    actor_net,
    c,
    flags,
):

    target_baseline = target_baseline.detach()
    baseline = new_actor_out.baseline[:, :, ind]
    loss = torch.sum(L2_loss(target_baseline, baseline)) * c
    return loss


def compute_policy_gradient_loss(logits_ls, actions_ls, masks_ls, c_ls, advantages):
    assert len(logits_ls) == len(actions_ls) == len(masks_ls) == len(c_ls)
    loss = 0.0
    for logits, actions, masks, c in zip(logits_ls, actions_ls, masks_ls, c_ls):
        if torch.sum(1 - masks) > 0.0:
            cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")(
                input=torch.flatten(logits, 0, 1), target=torch.flatten(actions, 0, 1)
            )
            cross_entropy = cross_entropy.view_as(advantages)
            adv_cross_entropy = cross_entropy * advantages.detach()
            loss = loss + torch.sum(adv_cross_entropy * (1 - masks)) * c
    return loss


def compute_entropy_loss(logits_ls, masks_ls, c_ls):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    loss = 0.0
    assert len(logits_ls) == len(masks_ls) == len(c_ls)
    for logits, masks, c in zip(logits_ls, masks_ls, c_ls):
        if torch.sum(1 - masks) > 0.0:
            logits = torch.flatten(logits, 0, 1)
            ent = -torch.nn.CrossEntropyLoss(reduction="none")(
                input=logits, target=F.softmax(logits, dim=-1)
            )
            ent = ent.view_as(masks)
            ent = ent * (1 - masks)
            loss = loss + torch.sum(ent) * c
    return loss


def action_log_probs(policy_logits, actions):
    return -torch.nn.CrossEntropyLoss(reduction="none")(
        input=torch.flatten(policy_logits, 0, 1), target=torch.flatten(actions, 0, 1)
    ).view_as(actions)


def from_logits(
    behavior_logits_ls,
    target_logits_ls,
    actions_ls,
    masks_ls,
    discounts,
    rewards,
    values,
    values_enc,
    rv_tran,
    bootstrap_value,
    flags,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    lamb=1.0,
    norm_stat=None,
):
    """V-trace for softmax policies."""
    assert (
        len(behavior_logits_ls)
        == len(target_logits_ls)
        == len(actions_ls)
        == len(masks_ls)
    )
    log_rhos = torch.tensor(0.0, device=behavior_logits_ls[0].device)
    for behavior_logits, target_logits, actions, masks in zip(
        behavior_logits_ls, target_logits_ls, actions_ls, masks_ls
    ):
        if torch.sum(1 - masks) > 0.0:
            behavior_log_probs = action_log_probs(behavior_logits, actions)
            target_log_probs = action_log_probs(target_logits, actions)
            log_rho = target_log_probs - behavior_log_probs
            log_rhos = log_rhos + log_rho * (1 - masks)

    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        values_enc=values_enc,
        rv_tran=rv_tran,
        bootstrap_value=bootstrap_value,
        flags=flags,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        lamb=lamb,
        norm_stat=norm_stat,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=None,
        target_action_log_probs=None,
        **vtrace_returns._asdict(),
    )


class SActorLearner:
    def __init__(self, buffers: dict, rank: int, flags: argparse.Namespace):
        if buffers is not None:
            self.param_buffer = buffers["actor_param"]
            self.actor_buffer = buffers["actor"]
        else:
            self.param_buffer = None
            self.actor_buffer = None

        self.rank = rank
        self.flags = flags
        self._logger = util.logger()
        self.time = flags.profile

        env = Environment(flags, model_wrap=True, env_n=1)
        self.actor_net = ActorNet(
            obs_shape=env.model_out_shape if not self.flags.disable_model else None,
            gym_obs_shape=env.gym_env_out_shape,
            num_actions=env.num_actions,
            flags=flags,
        )
        env.close()
        # initialize learning setting

        if not self.flags.disable_cuda and torch.cuda.is_available():
            self._logger.info("Actor-learning: Using CUDA.")
            self.device = torch.device("cuda")
        else:
            self._logger.info("Actor-learning: Not using CUDA.")
            self.device = torch.device("cpu")

        if not self.flags.use_rms:
            self.optimizer = torch.optim.Adam(
                self.actor_net.parameters(), lr=flags.learning_rate, eps=flags.adam_eps
            )
        else:
            self.optimizer = torch.optim.RMSprop(
                self.actor_net.parameters(),
                lr=flags.learning_rate,
                momentum=0,
                eps=0.01,
                alpha=0.99,
            )
        lr_lambda = lambda epoch: 1 - min(
            epoch * self.flags.unroll_length * self.flags.batch_size,
            self.flags.total_steps * self.flags.rec_t,
        ) / (self.flags.total_steps * self.flags.rec_t)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # other init. variables for consume_data
        self.step = 0
        self.tot_eps = 0
        self.real_step = 0
        max_actor_id = (
            self.flags.gpu_num_actors * self.flags.gpu_num_p_actors
            + self.flags.cpu_num_actors * self.flags.cpu_num_p_actors
        )
        self.last_returns = RetBuffer(max_actor_id, mean_n=400)
        self.last_im_returns = RetBuffer(max_actor_id, mean_n=20000)
        self.anneal_c = 1
        self.n = 0

        if self.flags.preload_actor and not flags.load_checkpoint:
            checkpoint = torch.load(
                self.flags.preload_actor, map_location=torch.device("cpu")
            )
            self.actor_net.set_weights(checkpoint["actor_net_state_dict"])
            self._logger.info(
                "Loadded actor network from %s" % self.flags.preload_actor
            )
            if "actor_net_optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(
                    checkpoint["actor_net_optimizer_state_dict"]
                )
                self._logger.info(
                    "Loadded actor network's optimizer from %s"
                    % self.flags.preload_actor
                )

        if flags.load_checkpoint:
            self.load_checkpoint(os.path.join(flags.load_checkpoint, "ckp_actor.tar"))
            self.flags.savedir = os.path.split(self.flags.load_checkpoint)[0]
            self.flags.xpid = os.path.split(self.flags.load_checkpoint)[-1]

        self.im_discounting = self.flags.discounting ** (1 / self.flags.rec_t)

        # initialize file logs
        self.flags.git_revision = util.get_git_revision_hash()
        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            overwrite=not self.flags.load_checkpoint,
        )

        self.check_point_path = "%s/%s/%s" % (
            flags.savedir,
            flags.xpid,
            "ckp_actor.tar",
        )

        # set shared buffer's weights
        if self.param_buffer is not None:
            self.param_buffer.set_data.remote("actor_net", self.actor_net.get_weights())

        # move network and optimizer to process device
        self.actor_net.to(self.device)
        util.optimizer_to(self.optimizer, self.device)

        self.norm_stat, self.im_norm_stat  = None, None, 

        # variables for timing
        self.queue_n = 0
        self.timer = timeit.default_timer
        self.start_time = self.timer()
        self.sps_buffer = [(self.step, self.start_time)] * 36
        self.sps = 0
        self.sps_buffer_n = 0
        self.sps_start_time, self.sps_start_step = self.start_time, self.step
        self.ckp_start_time = int(time.strftime("%M")) // 10

        if self.flags.float16:
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2**8)

    def learn_data(self):
        timing = util.Timings() if self.time else None
        try:
            while self.real_step < self.flags.total_steps:
                if timing is not None:
                    timing.reset()
                # get data remotely
                while True:
                    data_ptr = self.actor_buffer.read.remote()
                    data = ray.get(data_ptr)
                    if data is not None:
                        break
                    time.sleep(0.001)
                    self.queue_n += 0.001
                if timing is not None:
                    timing.time("get_data")
                train_actor_out, initial_actor_state = data
                train_actor_out = util.tuple_map(
                    train_actor_out, lambda x: torch.tensor(x, device=self.device)
                )
                initial_actor_state = util.tuple_map(
                    initial_actor_state, lambda x: torch.tensor(x, device=self.device)
                )
                if timing is not None:
                    timing.time("convert_data")
                data = (train_actor_out, initial_actor_state)
                # start consume data
                self.consume_data(data)
                del train_actor_out, initial_actor_state, data
                ray.internal.free(data_ptr)
                self.param_buffer.set_data.remote(
                    "actor_net", self.actor_net.get_weights()
                )
                if timing is not None:
                    timing.time("set weight")
            self.close(0)
            return True
        except Exception as e:
            self._logger.error(f"Exception detected in learn_actor: {e}")
            self._logger.error(traceback.format_exc())
        finally:
            self.close(0)
            return True

    def consume_data(self, data, timing=None):
        train_actor_out, initial_actor_state = data
        actor_id = train_actor_out.id

        # compute losses
        if self.flags.float16:
            with torch.cuda.amp.autocast():
                losses, train_actor_out = self.compute_losses(
                    train_actor_out, initial_actor_state
                )
        else:
            losses, train_actor_out = self.compute_losses(
                train_actor_out, initial_actor_state
            )
        total_loss = losses["total_loss"]
        if timing is not None:
            timing.time("compute loss")

        # gradient descent on loss
        self.optimizer.zero_grad()
        if self.flags.float16:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        if timing is not None:
            timing.time("compute gradient")

        optimize_params = self.optimizer.param_groups[0]["params"]
        if self.flags.float16:
            self.scaler.unscale_(self.optimizer)
        if self.flags.grad_norm_clipping > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                optimize_params, self.flags.grad_norm_clipping
            )
            total_norm = total_norm.detach().cpu().item()
        else:
            total_norm = util.compute_grad_norm(optimize_params)
        if timing is not None:
            timing.time("compute norm")

        if self.flags.float16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if timing is not None:
            timing.time("grad descent")

        self.scheduler.step()
        self.anneal_c = max(1 - self.real_step / self.flags.total_steps, 0)

        # statistic output
        stats = self.compute_stat(train_actor_out, losses, total_norm, actor_id)
        stats["sps"] = self.sps

        # write to log file
        self.plogger.log(stats)

        # print statistics
        if self.timer() - self.start_time > 5:
            self.sps_buffer[self.sps_buffer_n] = (self.step, self.timer())
            self.sps_buffer_n = (self.sps_buffer_n + 1) % len(self.sps_buffer)
            self.sps = (
                self.sps_buffer[self.sps_buffer_n - 1][0]
                - self.sps_buffer[self.sps_buffer_n][0]
            ) / (
                self.sps_buffer[self.sps_buffer_n - 1][1]
                - self.sps_buffer[self.sps_buffer_n][1]
            )
            tot_sps = (self.step - self.sps_start_step) / (
                self.timer() - self.sps_start_time
            )
            print_str = (
                "[%s] Steps %i @ %.1f SPS (%.1f). (T_q: %.2f) Eps %i. Ret %f (%f). Loss %.2f"
                % (
                    self.flags.xpid,
                    self.real_step,
                    self.sps,
                    tot_sps,
                    self.queue_n,
                    self.tot_eps,
                    stats["rmean_episode_return"],
                    stats["rmean_im_episode_return"],
                    total_loss,
                )
            )
            print_stats = [
                "max_rollout_depth",
                "entropy_loss",
                "reg_loss",
                "total_norm",
                "sat",
                "mean_abs_v",
            ]
            for k in print_stats:
                print_str += " %s %.2f" % (k, stats[k])

            self._logger.info(print_str)
            self.start_time = self.timer()
            self.queue_n = 0
            if self.time:
                print(self.timing.summary())

        if int(time.strftime("%M")) // 10 != self.ckp_start_time:
            self.save_checkpoint()
            self.ckp_start_time = int(time.strftime("%M")) // 10

        if timing is not None:
            timing.time("misc")
        del train_actor_out, losses, total_loss, stats, total_norm
        torch.cuda.empty_cache()

        # update shared buffer's weights
        self.n += 1

    def compute_losses(self, train_actor_out, initial_actor_state):
        # compute loss and then discard the first step in train_actor_out

        T, B = train_actor_out.done.shape
        T = T - 1
        new_actor_out, unused_state = self.actor_net(
            train_actor_out, initial_actor_state
        )

        # Take final value function slice for bootstrapping.
        bootstrap_value = new_actor_out.baseline[-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        train_actor_out = util.tuple_map(train_actor_out, lambda x: x[1:])
        new_actor_out = util.tuple_map(new_actor_out, lambda x: x[:-1])
        rewards = train_actor_out.reward

        # compute advantage w.r.t real rewards
        discounts = (~train_actor_out.done).float() * self.im_discounting

        behavior_logits_ls = [
            train_actor_out.policy_logits,
            train_actor_out.im_policy_logits,
            train_actor_out.reset_policy_logits,
        ]
        target_logits_ls = [
            new_actor_out.policy_logits,
            new_actor_out.im_policy_logits,
            new_actor_out.reset_policy_logits,
        ]
        actions_ls = [
            train_actor_out.action,
            train_actor_out.im_action,
            train_actor_out.reset_action,
        ]
        im_mask = (
            (train_actor_out.cur_t == 0).float()
            if not self.flags.disable_model
            else torch.ones(T, B, device=self.device)
        )
        real_mask = 1 - im_mask
        masks_ls = [real_mask, im_mask, im_mask]
        c_ls = [self.flags.real_cost, self.flags.real_im_cost, self.flags.real_im_cost]

        vtrace_returns = from_logits(
            behavior_logits_ls,
            target_logits_ls,
            actions_ls,
            masks_ls,
            discounts=discounts,
            rewards=rewards[:, :, 0],
            values=new_actor_out.baseline[:, :, 0],
            values_enc=new_actor_out.baseline_enc[:, :, 0]
            if new_actor_out.baseline_enc is not None
            else None,
            rv_tran=self.actor_net.rv_tran,
            bootstrap_value=bootstrap_value[:, 0],
            flags=self.flags,
            lamb=self.flags.lamb,
            norm_stat=self.norm_stat,
        )
        self.norm_stat = vtrace_returns.norm_stat
        advs = vtrace_returns.pg_advantages
        pg_loss = compute_policy_gradient_loss(
            target_logits_ls, actions_ls, masks_ls, c_ls, advs
        )
        baseline_loss = self.flags.baseline_cost * compute_baseline_loss(
            new_actor_out=new_actor_out,
            ind=0,
            target_baseline=vtrace_returns.vs,
            actor_net=self.actor_net,
            c=self.flags.real_cost,
            flags=self.flags,
        )
      
        # compute advantage w.r.t imagainary rewards

        if self.flags.im_cost > 0.0:
            discounts = (~(train_actor_out.cur_t == 0)).float() * self.im_discounting
            behavior_logits_ls = [
                train_actor_out.im_policy_logits,
                train_actor_out.reset_policy_logits,
            ]
            target_logits_ls = [
                new_actor_out.im_policy_logits,
                new_actor_out.reset_policy_logits,
            ]
            actions_ls = [train_actor_out.im_action, train_actor_out.reset_action]
            masks_ls = [im_mask, im_mask]
            if not self.flags.im_cost_anneal:
                c_ls = [
                    self.flags.im_cost,
                    self.flags.im_cost,
                ]
            else:
                c_ls = [
                    self.flags.im_cost * self.anneal_c,
                    self.flags.im_cost * self.anneal_c,
                ]

            vtrace_returns = from_logits(
                behavior_logits_ls,
                target_logits_ls,
                actions_ls,
                masks_ls,
                discounts=discounts,
                rewards=rewards[:, :, 1],
                values=new_actor_out.baseline[:, :, 1],
                values_enc=new_actor_out.baseline_enc[:, :, 1]
                if new_actor_out.baseline_enc is not None
                else None,
                rv_tran=self.actor_net.rv_tran,
                bootstrap_value=bootstrap_value[:, 1],
                flags=self.flags,
                lamb=self.flags.lamb,
                norm_stat=self.im_norm_stat,
            )
            self.im_norm_stat = vtrace_returns.norm_stat

            advs = vtrace_returns.pg_advantages
            im_pg_loss = compute_policy_gradient_loss(
                target_logits_ls, actions_ls, masks_ls, c_ls, advs
            )
            im_baseline_loss = self.flags.im_baseline_cost * compute_baseline_loss(
                new_actor_out=new_actor_out,
                ind=1,
                target_baseline=vtrace_returns.vs,
                actor_net=self.actor_net,
                c=self.flags.im_cost
                if not self.flags.im_cost_anneal
                else self.flags.im_cost * self.anneal_c,
                flags=self.flags,
            )

        else:
            im_pg_loss = torch.zeros(1, device=self.device)
            im_baseline_loss = torch.zeros(1, device=self.device)

        target_logits_ls = [
            new_actor_out.policy_logits,
            new_actor_out.im_policy_logits,
            new_actor_out.reset_policy_logits,
        ]
        masks_ls = [real_mask, im_mask, im_mask]
        im_ent_c = self.flags.im_entropy_cost * (
            self.flags.real_im_cost
            + (
                (
                    self.flags.im_cost
                    if not self.flags.im_cost_anneal
                    else self.flags.im_cost * self.anneal_c
                )
                if self.flags.im_cost > 0.0
                else 0
            )
        )
        c_ls = [self.flags.entropy_cost * self.flags.real_cost, im_ent_c, im_ent_c]
        entropy_loss = compute_entropy_loss(target_logits_ls, masks_ls, c_ls)

        reg_loss = self.flags.reg_cost * torch.sum(new_actor_out.reg_loss)
        total_loss = pg_loss + baseline_loss + entropy_loss + reg_loss

        if self.flags.im_cost > 0.0:
            total_loss = total_loss + im_pg_loss + im_baseline_loss


        losses = {
            "pg_loss": pg_loss,
            "im_pg_loss": im_pg_loss,
            "baseline_loss": baseline_loss,
            "im_baseline_loss": im_baseline_loss,
            "entropy_loss": entropy_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
        }
        return losses, train_actor_out

    def compute_stat(self, train_actor_out, losses, total_norm, actor_id):
        """Update step, real_step and tot_eps; return training stat for printing"""
        T, B, *_ = train_actor_out.episode_return.shape
        if torch.any(train_actor_out.real_done):
            episode_returns = train_actor_out.episode_return[train_actor_out.real_done][
                :, 0
            ]
            episode_returns = tuple(episode_returns.detach().cpu().numpy())
            episode_lens = train_actor_out.episode_step[train_actor_out.real_done]
            episode_lens = tuple(episode_lens.detach().cpu().numpy())
            done_ids = actor_id.broadcast_to(train_actor_out.real_done.shape)[
                train_actor_out.real_done
            ]
            done_ids = tuple(done_ids.detach().cpu().numpy())
        else:
            episode_returns, episode_lens, done_ids = (), (), ()
        self.last_returns.insert(episode_returns, done_ids)
        rmean_episode_return = self.last_returns.get_mean()

        if self.flags.im_cost > 0.0:
            self.last_im_returns.insert_raw(
                train_actor_out.episode_return,
                ind=1,
                actor_id=actor_id,
                done=train_actor_out.cur_t == 0,
            )
            rmean_im_episode_return = self.last_im_returns.get_mean()
        else:
            rmean_im_episode_return = 0.0

        if not self.flags.disable_model:
            max_rollout_depth = (
                (train_actor_out.max_rollout_depth[train_actor_out.cur_t == 0])
                .detach()
                .cpu()
                .numpy()
            )
            max_rollout_depth = (
                np.average(max_rollout_depth) if len(max_rollout_depth) > 0 else 0.0
            )
            cur_real_step = torch.sum(train_actor_out.cur_t == 0).item()
            mean_plan_step = (
                self.flags.unroll_length * self.flags.batch_size / max(cur_real_step, 1)
            )
        else:
            max_rollout_depth = 0.0
            cur_real_step = T * B
            mean_plan_step = 0.0

        self.step += self.flags.unroll_length * self.flags.batch_size
        self.real_step += cur_real_step
        self.tot_eps += torch.sum(train_actor_out.real_done).item()

        if train_actor_out.cur_t is not None:
            real_mask = train_actor_out.cur_t == 0
        else:
            real_mask = torch.ones(T, B, device=self.device, dtype=bool)
        sat = torch.mean(
            torch.max(
                torch.softmax(train_actor_out.policy_logits[real_mask], dim=-1), dim=-1
            )[0]
        ).item()
        mean_abs_v = torch.mean(torch.abs(train_actor_out.baseline)).item()

        stats = {
            "step": self.step,
            "real_step": self.real_step,
            "tot_eps": self.tot_eps,
            "rmean_episode_return": rmean_episode_return,
            "rmean_im_episode_return": rmean_im_episode_return,
            "episode_returns": episode_returns,
            "episode_lens": episode_lens,
            "done_ids": done_ids,
            "cur_real_step": cur_real_step,
            "mean_plan_step": mean_plan_step,
            "max_rollout_depth": max_rollout_depth,
            "total_norm": total_norm,
            "sat": sat,
            "mean_abs_v": mean_abs_v,
        }

        if losses is not None:
            for k, v in losses.items():
                stats[k] = v.item()

        return stats

    def save_checkpoint(self):
        self._logger.info("Saving actor checkpoint to %s" % self.check_point_path)
        torch.save(
            {
                "step": self.step,
                "real_step": self.real_step,
                "tot_eps": self.tot_eps,
                "last_returns": self.last_returns,
                "last_im_returns": self.last_im_returns,
                "actor_net_optimizer_state_dict": self.optimizer.state_dict(),
                "actor_net_scheduler_state_dict": self.scheduler.state_dict(),
                "actor_net_state_dict": self.actor_net.state_dict(),
                "flags": vars(self.flags),
            },
            self.check_point_path + ".tmp",
        )
        os.replace(self.check_point_path + ".tmp", self.check_point_path)

    def load_checkpoint(self, check_point_path: str):
        train_checkpoint = torch.load(check_point_path, torch.device("cpu"))
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self.tot_eps = train_checkpoint["tot_eps"]
        self.last_returns = train_checkpoint["last_returns"]
        self.last_im_returns = train_checkpoint["last_im_returns"]
        self.optimizer.load_state_dict(
            train_checkpoint["actor_net_optimizer_state_dict"]
        )
        self.scheduler.load_state_dict(
            train_checkpoint["actor_net_scheduler_state_dict"]
        )
        self.actor_net.set_weights(train_checkpoint["actor_net_state_dict"])
        self._logger.info("Loaded actor checkpoint from %s" % check_point_path)

    def close(self, exit_code=0):
        self.plogger.close()


@ray.remote
class ActorLearner(SActorLearner):
    pass


class RetBuffer:
    def __init__(self, max_actor_id, mean_n=400):
        """
        Compute the trailing mean return by storing the last return for each actor
        and average them;
        Args:
            max_actor_id (int): maximum actor id
            mean_n (int): size of data for computing mean
        """
        buffer_n = mean_n // max_actor_id + 1
        self.return_buffer = np.zeros((max_actor_id, buffer_n))
        self.return_buffer_pointer = np.zeros(
            max_actor_id, dtype=int
        )  # store the current pointer
        self.return_buffer_n = np.zeros(
            max_actor_id, dtype=int
        )  # store the processed data size
        self.max_actor_id = max_actor_id
        self.mean_n = mean_n
        self.all_filled = False

    def insert(self, returns, actor_ids):
        """
        Insert new returnss to the return buffer
        Args:
            returns (tuple): tuple of float, the return of each ended episode
            actor_ids (tuple): tuple of int, the actor id of the corresponding ended episode
        """
        # actor_id is a tuple of integer, corresponding to the returns
        if len(returns) == 0:
            return
        assert len(returns) == len(actor_ids)
        for r, actor_id in zip(returns, actor_ids):
            if actor_id >= self.max_actor_id:
                continue
            # Find the current pointer for the actor
            pointer = self.return_buffer_pointer[actor_id]
            # Update the return buffer for the actor with the new return
            self.return_buffer[actor_id, pointer] = r
            # Update the pointer for the actor
            self.return_buffer_pointer[actor_id] = (
                pointer + 1
            ) % self.return_buffer.shape[1]
            # Update the processed data size for the actor
            self.return_buffer_n[actor_id] = min(
                self.return_buffer_n[actor_id] + 1, self.return_buffer.shape[1]
            )

        if not self.all_filled:
            # check if all filled
            self.all_filled = np.all(
                self.return_buffer_n >= self.return_buffer.shape[1]
            )

    def insert_raw(self, episode_returns, ind, actor_id, done):
        episode_returns = episode_returns[done][:, ind]
        episode_returns = tuple(episode_returns.detach().cpu().numpy())
        done_ids = actor_id.broadcast_to(done.shape)[done]
        done_ids = tuple(done_ids.detach().cpu().numpy())
        self.insert(episode_returns, done_ids)

    def get_mean(self):
        """
        Compute the mean of the returns in the buffer;
        """
        if self.all_filled:
            overall_mean = np.mean(self.return_buffer)
        else:
            # Create a mask of filled items in the return buffer
            col_indices = np.arange(self.return_buffer.shape[1])
            # Create a mask of filled items in the return buffer
            filled_mask = (
                col_indices[np.newaxis, :] < self.return_buffer_n[:, np.newaxis]
            )
            if np.any(filled_mask):
                # Compute the sum of returns for each actor considering only filled items
                sum_returns = np.sum(self.return_buffer * filled_mask)
                # Compute the mean for each actor by dividing the sum by the processed data size
                overall_mean = sum_returns / np.sum(filled_mask.astype(float))
            else:
                overall_mean = 0.0
        return overall_mean
