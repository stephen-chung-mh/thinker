import os
import shutil
import time, timeit
from collections import namedtuple
import numpy as np
import traceback
import torch
import ray
from thinker.buffer import AB_CAN_WRITE, AB_FULL, AB_FINISH
from thinker.actor_net import ActorNet, ActorOut
from thinker.learn_actor import ActorLearner, SActorLearner
from thinker.main import Env
import thinker.util as util

_fields = ("tree_reps", "real_states", "xs", "hs")
_fields += ("reward", "episode_return", "episode_step")
_fields += ("done", "real_done", "truncated_done")
_fields += ("max_rollout_depth", "step_status")
_fields += ("last_pri", "last_reset")
EnvOut = namedtuple("EnvOut", _fields)
_fields = tuple(ActorOut._fields) + tuple(EnvOut._fields) + ("id",)

exc_list = ["action",
            "action_prob",
            "entropy_loss",
            "reg_loss", 
            "baseline_enc", 
            "misc",
            ]
_fields = (item for item in _fields if item not in exc_list)
TrainActorOut = namedtuple("TrainActorOut", _fields)

@ray.remote
class SelfPlayWorker:
    def __init__(self, ray_obj_actor, ray_obj_env, rank, env_n, flags):
        self._logger = util.logger()
        gpu = False
        if flags.gpu_self_play > 0 and torch.cuda.is_available():
            gpu = True
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.actor_buffer = ray_obj_actor["actor_buffer"]
        self.actor_param_buffer = ray_obj_actor["actor_param_buffer"]
        self.test_buffer = None

        self._logger.info(
            "Initializing actor %d with device %s %s"
            % (
                rank,
                "cuda" if gpu else "cpu",
                "(test mode)" if self.test_buffer is not None else "",
            )
        )

        self.rank = rank
        self.env_n = env_n
        self.flags = flags
     
        self.timing = util.Timings()
        self.actor_id = (
            torch.arange(self.env_n, device=self.device)
            + self.rank * self.env_n
        ).unsqueeze(0)
        self.time = self.rank == 0 and flags.profile

        self.env = Env(
            name = flags.name, 
            ray_obj = ray_obj_env,
            env_n = env_n,
            gpu = gpu,

        )

        obs_space = self.env.observation_space
        action_space = self.env.action_space        

        self.has_actor = True
        self.train_actor = self.has_actor and flags.train_actor

        if self.has_actor:
            actor_param = {
                "obs_space":obs_space,
                "action_space":action_space,
                "flags":flags
            }
            self.actor_net = ActorNet(**actor_param)
            if self.rank == 0:
                self._logger.info(
                    "Actor network size: %d"
                    % sum(p.numel() for p in self.actor_net.parameters())
                )
            self._load_net()          
            self.actor_net.to(self.device)
            self.actor_net.train(False)
            if self.train_actor and self.rank == 0:
                if self.flags.parallel_actor:
                    # init. the actor learner thread
                    self.actor_learner = ActorLearner.options(
                        num_cpus=1, num_gpus=self.flags.gpu_learn_actor,
                    ).remote(ray_obj_actor, actor_param, self.flags)
                    # start learning
                    self.r_learner = self.actor_learner.learn_data.remote()
                else:
                    self.actor_learner = SActorLearner(
                        None, actor_param, self.flags, self.actor_net, self.device)

        self.disable_thinker = flags.wrapper_type == 1
        self.finish_train_actor = False

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
                    self.env_n, device=self.device
                )
            state = self.env.reset()
            env_out = self.init_env_out(state)                        
            actor_state = self.actor_net.initial_state(
                    batch_size=self.env_n, device=self.device
            )
            actor_out, actor_state, env_out, info = self.env_step(env_out, actor_state)
       
            timer = timeit.default_timer
            start_time = timer()

            self.actor_net.train(False)
            while True:
          
                if self.time: self.timing.reset()
                # prepare train_actor_out data to be written
                initial_actor_state = actor_state

                send_buffer = self.train_actor and not (self.flags.train_model and 
                        not info["model_status"]["running"] and
                        not info["model_status"]["finish"]
                        and self.flags.ckp) 
                if send_buffer:
                    self.write_actor_buffer(env_out, actor_out, 0)
                if self.time: self.timing.time("misc1")
         
                with torch.set_grad_enabled(False):
                    for t in range(self.flags.actor_unroll_len):
                        # generate action
                        actor_out, actor_state, env_out, info = \
                            self.env_step(env_out, actor_state)
                        if self.time: self.timing.time("step env")
                        # write the data to the respective buffers
                        if send_buffer: self.write_actor_buffer(env_out, actor_out, t + 1)
                        if self.time: self.timing.time("finish actor buffer")                      
   
                if send_buffer and self.flags.parallel_actor:
                    # send the data to remote actor buffer
                    initial_actor_state = util.tuple_map(
                        initial_actor_state, lambda x: x.detach().cpu().numpy()
                    )
                    status = 0
                    if self.time: self.timing.time("mics2")
        
                    while True:
                        data_full_ptr = self.actor_buffer.get_status.remote()
                        status = ray.get(data_full_ptr)
                        if status == AB_FULL:
                            time.sleep(0.1)
                        else:
                            if status == AB_FINISH: self.train_actor = False
                            break
                    if self.train_actor:
                        self.actor_buffer.write.remote(
                            ray.put(self.actor_local_buffer),
                            ray.put(initial_actor_state),
                        )
                    if self.time: self.timing.time("send actor buffer")                    
  
                if self.time: self.timing.time("mics3")
      
                if send_buffer and not self.flags.parallel_actor and hasattr(self, "actor_local_buffer"):
                    data = (self.actor_local_buffer, initial_actor_state)
                    self.actor_net.train(True)
                    self.train_actor = not self.actor_learner.consume_data(data)
                    self.actor_net.train(False)
                
                if send_buffer and self.flags.parallel_actor:
                    self._refresh_net()

                if self.time:
                    self.timing.time("update actor net weight")

                n += 1
                if self.time and timer() - start_time > 5:
                    self._logger.info(self.timing.summary())
                    start_time = timer()

                fin = True
                if self.train_actor: fin = False                
                if not info["model_status"]["finish"]: fin = False
                if fin: 
                    self._logger.info("Terminating self-play thread %d" % self.rank)
                    self.env.close()                    
                    return True

        except Exception as e:
            self._logger.error(f"Exception detected in self_play: {e}")
            self._logger.error(traceback.format_exc())
        finally:
            return False
    
    def env_step(self, env_out, actor_state):
        actor_out, actor_state = self.actor_net(
                            env_out, actor_state, greedy=False
                        )
        state, reward, done, info = self.env.step(
                action=actor_out.action[0], 
                reset=actor_out.action[1], 
                action_prob=actor_out.action_prob)
        env_out = self.create_env_out(actor_out.action, state, reward, done, info)
        return actor_out, actor_state, env_out, info

    def write_actor_buffer(self, env_out: EnvOut, actor_out: ActorOut, t: int):
        # write to local buffer
        if t == 0:            
            if self.flags.parallel_actor:
                id = self.actor_id
            else:
                id = [self.actor_id[0]]
            out = {"id": id}
            for field in TrainActorOut._fields:
                if field in ["id"]: continue
                if field == "real_stats" and not self.see_real_state: continue
                val = getattr(env_out if field in EnvOut._fields else actor_out, field)
                out[field] = None
                if val is None: continue
                if self.flags.parallel_actor:
                    out[field] = torch.empty(
                        size=(self.flags.actor_unroll_len + 1, self.env_n)
                        + val.shape[2:],
                        dtype=val.dtype,
                        device=self.device,
                    )
                else:
                    out[field] = []
                # each is in the shape of (T x B xdim_1 x dim_2 ...)
            self.actor_local_buffer = TrainActorOut(**out)

        for field in TrainActorOut._fields:
            v = getattr(self.actor_local_buffer, field)
            if v is not None and field not in ["id"]:
                new_val = getattr(
                    env_out if field in EnvOut._fields else actor_out, field
                )[0]
                if self.flags.parallel_actor:
                    v[t] = new_val
                else:
                    v.append(new_val)

        if self.time:
            self.timing.time("write_actor_buffer")

        if t == self.flags.actor_unroll_len:
            # post-processing
            if self.flags.parallel_actor:
                map = lambda x: x.cpu().numpy()
            else:
                map = lambda x: torch.stack(x, dim=0)
            self.actor_local_buffer = util.tuple_map(
                self.actor_local_buffer,  map
            )
        if self.time:
            self.timing.time("move_actor_buffer_to_cpu")

    def init_env_out(self, *args, **kwargs):
        return init_env_out(*args, **kwargs, flags=self.flags)

    def create_env_out(self, *args, **kwargs):
        return create_env_out(*args, **kwargs, flags=self.flags)

    def _load_net(self):
        if self.rank == 0:
            # load the network from preload or load_checkpoint  
            path = None
            if self.flags.ckp:
                path = os.path.join(self.flags.ckpdir, "ckp_actor.tar")
            else:
                if self.flags.preload_actor:
                    path = os.path.join(self.flags.preload_actor, "ckp_actor.tar")
                    shutil.copyfile(path, os.path.join(self.ckpdir, "ckp_actor.tar"))
            if path is not None:
                checkpoint = torch.load(
                    path, map_location=torch.device("cpu")
                )
                self.actor_net.set_weights(
                    checkpoint["actor_net_state_dict"]
                )
                self._logger.info("Loaded actor net from %s" % path)
            if self.flags.parallel_actor:            
                self.actor_param_buffer.set_data.remote(
                    "actor_net", self.actor_net.get_weights()
                )
        else:
            self._refresh_net()
        return
    
    def _refresh_net(self):
        while True:
            weights = ray.get(
                self.actor_param_buffer.get_data.remote("actor_net")
            )  
            if weights is not None:
                self.actor_net.set_weights(weights)
                del weights
                break                
            time.sleep(0.1)  

def init_env_out(state, flags):
        # minimum env_out for actor_net
        num_rewards = 1        
        num_rewards += int(flags.im_cost > 0.0)
        num_rewards += int(flags.cur_cost > 0.0)

        env_n = state["real_states"].shape[0]
        device = state["real_states"].device

        out = {
            "last_pri": torch.zeros(env_n, dtype=torch.long, device=device),
            "last_reset": torch.zeros(env_n, dtype=torch.long, device=device),
            "reward": torch.zeros((env_n, num_rewards), 
                                dtype=torch.float, device=device),
            "done": torch.zeros(env_n, dtype=torch.bool, device=device),
            "step_status": torch.zeros(env_n, dtype=torch.long, device=device),
        }

        for field in EnvOut._fields:    
            if field not in out:
                out[field] = None
            else:
                continue
            if field in state.keys():
                out[field] = state[field]

        for k, v in out.items():
            if v is not None:
                out[k] = torch.unsqueeze(v, dim=0)
        env_out = EnvOut(**out)        
        return env_out
        

def create_env_out(action, state, reward, done, info, flags):
    if flags.wrapper_type == 0:    
        aug_reward = [reward]
        aug_epsoide_return = [info['episode_return']]
        if flags.im_cost > 0:
            aug_reward.append(info["im_reward"])
            aug_epsoide_return.append(info["im_episode_return"])
        if flags.cur_cost > 0:
            aug_reward.append(info["cur_reward"])
            aug_epsoide_return.append(info["cur_episode_return"])
        aug_reward = torch.stack(aug_reward, dim=-1)
        aug_epsoide_return = torch.stack(aug_epsoide_return, dim=-1)
    else:
        aug_reward = torch.unsqueeze(reward, -1)
        aug_epsoide_return = torch.unsqueeze(info['episode_return'], -1)
    out = {"reward": aug_reward, 
            "episode_return": aug_epsoide_return,
            "done": done,
            }
    if not flags.wrapper_type == 1:    
        out["last_pri"] = action[0]
        out["last_reset"] = action[1]
    else:
        out["last_pri"] = action

    for field in EnvOut._fields:    
        if field not in out:
            out[field] = None
        else:
            continue
        if field in state.keys():
            out[field] = state[field]
        if field in info.keys():
            out[field] = info[field]
    
    for k, v in out.items():
        if v is not None:
            out[k] = torch.unsqueeze(v, dim=0)
    env_out = EnvOut(**out)
    return env_out    