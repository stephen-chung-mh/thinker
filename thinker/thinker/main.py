import os
import shutil
import time
from collections import namedtuple
import ray
import numpy as np
import torch
import thinker.util as util
from thinker.buffer import ModelBuffer, SModelBuffer, GeneralBuffer
from thinker.learn_model import ModelLearner, SModelLearner
from thinker.model_net import ModelNet
from thinker.gym_add.asyn_vector_env import AsyncVectorEnv
import thinker.wrapper as wrapper
from thinker.cenv import cModelWrapper, cPerfectWrapper
import gym
#from gym.wrappers import NormalizeObservation

def ray_init(flags=None, **kwargs):
    # initialize resources for Thinker wrapper
    if flags is None:
        flags = util.create_flags(filename='default_thinker.yaml',
                              **kwargs)
        flags.parallel=True

    if not ray.is_initialized(): 
        object_store_memory = int(flags.ray_mem * 1024**3) if flags.ray_mem > 0 else None
        ray.init(num_cpus=flags.ray_cpu if flags.ray_cpu > 0 else None,
                 num_gpus=flags.ray_gpu if flags.ray_gpu > 0 else None,
                 object_store_memory=object_store_memory)
    model_buffer = ModelBuffer.options(num_cpus=1).remote(
            buffer_n = flags.model_buffer_n,
            max_rank = flags.self_play_n,
            batch_size = flags.env_n,
            alpha = flags.priority_alpha,
            warm_up_n = flags.model_warm_up_n,
    )
    param_buffer = GeneralBuffer.options(num_cpus=1).remote()    
    param_buffer.set_data.remote("flags", flags)
    signal_buffer = GeneralBuffer.options(num_cpus=1).remote()   
    ray_obj = {"model_buffer": model_buffer,
               "param_buffer": param_buffer,
               "signal_buffer": signal_buffer}
    return ray_obj

class Env(gym.Wrapper):
    def __init__(self, 
                 name=None, 
                 env_fn=None, 
                 ray_obj=None, 
                 env_n=1, 
                 gpu=True,
                 load_net=True, 
                 timing=False,
                 **kwargs):
        assert name is not None or env_fn is not None, \
            "need either env or env-making function"        
        
        if ray_obj is None:
            self.flags = util.create_flags(filename='default_thinker.yaml',
                              **kwargs)
            if self.flags.parallel:
                ray_obj = ray_init(self.flags)       
        else:
            assert not kwargs, "Unexpected keyword arguments provided"
            self.flags = ray.get(ray_obj["param_buffer"].get_data.remote("flags"))
        
        self._logger = util.logger() 
        self.parallel = self.flags.parallel
                
        self.env_n = env_n
        self.device = torch.device("cuda") if gpu else torch.device("cpu")
        
        if self.parallel:
            self.model_buffer = ray_obj["model_buffer"]
            self.param_buffer = ray_obj["param_buffer"]
            self.signal_buffer = ray_obj["signal_buffer"]
            self.rank = ray.get(self.param_buffer.get_and_increment.remote("rank"))
        else:
            self.rank = 0
        self.counter = 0
        self.ckp_start_time = int(time.strftime("%M")) // 10
        self.ckp_env_path = os.path.join(self.flags.ckpdir, "ckp_env.npz")    

        self._logger.info(
            "Initializing env %d with device %s"
            % (
                self.rank,
                "cuda" if self.device == torch.device("cuda") else "cpu",
            )
        )
        if env_fn is None: env_fn = wrapper.create_env_fn(name, self.flags)
        # initialize a single env to collect env information
        env = env_fn()
        assert len(env.observation_space.shape) in [1, 3], \
            f"env.observation_space should be 1d or 3d, not {env.observation_space.shape}"
        # assert type(env.action_space) in [gym.spaces.discrete.Discrete, gym.spaces.tuple.Tuple], \
        #    f"env.action_space should be Discrete or Tuple, not {type(env.action_space)}"  
        
        if env.observation_space.dtype == 'uint8':
            self.state_dtype = 0
        elif env.observation_space.dtype == 'float32':
            self.state_dtype = 1        
        
        self.real_state_space  = env.observation_space
        self.real_state_shape = env.observation_space.shape

        self.pri_action_space = env.action_space
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(self.pri_action_space)

        if isinstance(self.pri_action_space, gym.spaces.Box):
            assert len(env.action_space.shape) == 1, f"Invalid action space {env.action_space}"

        self._logger.info(f"Init. environment with obs space \033[91m{env.observation_space}\033[0m and action space \033[91m{env.action_space}\033[0m")        
        self.sample = self.flags.sample_n > 0
        self.sample_n = self.flags.sample_n

        if self.sample:
            self.pri_action_shape = (self.env_n,)
            self.action_prob_shape = (self.env_n, self.sample_n,)
        elif self.tuple_action:
            self.pri_action_shape = (self.env_n, self.dim_actions)
            if self.discrete_action:
                self.action_prob_shape = self.pri_action_shape + (self.num_actions,)
            else:
                self.action_prob_shape = self.pri_action_shape + (2,) # mean and var of Gaussian dist
        else:
            self.pri_action_shape = (self.env_n,)
            self.action_prob_shape = (self.env_n, self.num_actions,)

        self.frame_stack_n = env.frame_stack_n if hasattr(env, "frame_stack_n") else 1
        self.frame_ch = env.observation_space.shape[0] // self.frame_stack_n
        self.model_mem_unroll_len = self.flags.model_mem_unroll_len
        self.pre_len = self.frame_stack_n - 1 + self.model_mem_unroll_len
        self.post_len = self.flags.model_unroll_len + self.flags.model_return_n + 1

        if self.rank == 0 and self.frame_stack_n > 1:
            self._logger.info("Detected frame stacking with %d counts" % self.frame_stack_n)
        env.close()

        # initalize model
        self.has_model = self.flags.has_model
        self.train_model = self.has_model and self.flags.train_model 
        self.require_prob = False
        self.sample = self.flags.sample_n > 0
        if self.has_model:
            model_param = {
                "obs_space": self.real_state_space,                
                "action_space": self.pri_action_space, 
                "flags": self.flags,
                "frame_stack_n": self.frame_stack_n
            }
            self.model_net = ModelNet(**model_param)
            if self.rank == 0:
                self._logger.info(
                    "Model network size: %d"
                    % sum(p.numel() for p in self.model_net.parameters())
                )
            if load_net: self._load_net()            
            self.model_net.train(False)
            self.model_net.to(self.device)       
            if self.train_model and self.rank == 0:
                if self.parallel:
                    # init. the model learner thread
                    self.model_learner = ModelLearner.options(
                        num_cpus=1, num_gpus=self.flags.gpu_learn,
                    ).remote(name, ray_obj, model_param, self.flags)
                    # start learning
                    self.r_learner = self.model_learner.learn_data.remote()
                    self.model_buffer.set_frame_stack_n.remote(self.frame_stack_n)
                else:
                    self.model_learner = SModelLearner(name=name, ray_obj=None, model_param=model_param,
                        flags=self.flags, model_net=self.model_net, device=self.device)
                    self.model_buffer = SModelBuffer(
                        buffer_n = self.flags.model_buffer_n,
                        max_rank = self.flags.self_play_n,
                        batch_size = self.flags.env_n,
                        alpha = self.flags.priority_alpha,
                        warm_up_n = self.flags.model_warm_up_n,                        
                    )
                    self.model_buffer.set_frame_stack_n(self.frame_stack_n)
            if self.train_model: self.require_prob = self.flags.require_prob
            
            per_state = self.model_net.initial_state(batch_size=1)
            self.per_state_shape = {k:v.shape[1:] for k, v in per_state.items()}
        else:
            self.model_net = None            
            
        # create batched asyn. environments
        env = AsyncVectorEnv([env_fn for _ in range(env_n)]) 
        env = wrapper.InfoConcat(env)
        env = wrapper.RecordEpisodeStatistics(env)                      
        if self.flags.obs_norm:
            assert env.observation_space.dtype == np.float32  
            env = wrapper.NormalizeObservation(env)
        if self.flags.reward_norm:
            env = wrapper.NormalizeReward(env, gamma=self.flags.discounting)
        if self.flags.obs_clip > 0:
            env = wrapper.TransformObservation(env, lambda obs: np.clip(obs, -self.flags.obs_clip, self.flags.obs_clip))
        if self.flags.reward_clip > 0:
            env = wrapper.TransformReward(env, lambda reward: np.clip(reward, -self.flags.reward_clip, self.flags.reward_clip))

        env.seed([i for i in range(
            self.rank * env_n + self.flags.base_seed, 
            self.rank * env_n + self.flags.base_seed + env_n)])       

        if self.flags.wrapper_type == 0:
            core_wrapper = cModelWrapper
        elif self.flags.wrapper_type == 1:
            core_wrapper = wrapper.DummyWrapper
        elif self.flags.wrapper_type == 2:
            core_wrapper = cPerfectWrapper
        else:
            raise Exception(
                f"wrapper_type can only be [0, 1, 2], not {self.flags.wrapper_type}")

        # wrap the env with core Cython wrapper that runs
        # the core Thinker algorithm
        env = core_wrapper(env=env, 
                        env_n=env_n, 
                        flags=self.flags, 
                        model_net=self.model_net, 
                        device=self.device, 
                        timing=timing)
        
        if self.flags.ckp and os.path.exists(self.ckp_env_path):
            with np.load(self.ckp_env_path, allow_pickle=True) as data:
                env.load_ckp(data)
                    
        # wrap the env with a wrapper that computes episode
        # return and episode step for logging purpose;
        # also clip the reward afterwards if set
        env = wrapper.PostWrapper(env, self.flags, self.device) 
        gym.Wrapper.__init__(self, env)                          

        if self.train_model:
            if self.flags.parallel:
                self.status_ptr = self.model_buffer.get_status.remote()        
                self._update_status()
                self.signal_ptr = self.signal_buffer.get_data.remote("self_play_signals")
            else:
                self.status = self.model_buffer.get_status()
        else:
            self.status = {"processed_n": 0,
                           "warm_up_n": 0,
                           "running": False,
                           "finish": True,
                            }

        
    def _load_net(self):
        if self.rank == 0:
            # load the network from preload or load_checkpoint  
            path = None
            if self.flags.ckp:
                path = os.path.join(self.flags.ckpdir, "ckp_model.tar")
            else:
                if self.flags.preload:
                    path = os.path.join(self.flags.preload, "ckp_model.tar")
                    shutil.copyfile(path, os.path.join(self.flags.ckpdir, "ckp_model.tar"))
            if path is not None:                
                checkpoint = torch.load(
                    path, map_location=torch.device("cpu")
                )
                self.model_net.set_weights(
                    checkpoint["model_net_state_dict"]
                )
                self._logger.info("Loaded model net from %s" % path)
            
            if self.has_model and self.parallel:
                self.param_buffer.set_data.remote(
                    "model_net", self.model_net.get_weights()
                )
        else:
            self._refresh_net()
        return
    
    def _refresh_net(self):
        while True:
            weights = ray.get(
                self.param_buffer.get_data.remote("model_net")
            )  
            if weights is not None:
                self.model_net.set_weights(weights)
                del weights
                break                
            time.sleep(0.1)  
    
    def _update_status(self):
        self.status = ray.get(self.status_ptr)
        self.status_ptr = self.model_buffer.get_status.remote()        

    def reset(self):
        state = self.env.reset(self.model_net)
        if self.sample: self.sampled_action = state["sampled_action"]
        return state

    def step(self, primary_action, reset_action=None, action_prob=None, ignore=False):        

        assert primary_action.shape == self.pri_action_shape, \
                    f"primary_action should have shape {self.pri_action_shape} not {primary_action.shape}"  
        if self.flags.wrapper_type == 1:
            action = primary_action                
        else:
            assert reset_action.shape == (self.env_n,), \
                    f"reset should have shape {(self.env_n,)} not {reset_action.shape}"
            action = (primary_action, reset_action)            
                
        if self.require_prob and not ignore: 
            assert action_prob is not None
            assert action_prob.shape == self.action_prob_shape, \
                    f"action_prob should have shape {self.action_prob_shape} not {action_prob.shape}"
        
        with torch.set_grad_enabled(False):
            state, reward, done, info = self.env.step(action, self.model_net)  
        last_step_real = (info["step_status"] == 0) | (info["step_status"] == 3)
        if self.train_model and not ignore and torch.any(last_step_real): 
            self._write_send_model_buffer(state, reward, done, info, primary_action, action_prob)        
        if self.sample: self.sampled_action = state["sampled_action"] # should refresh sampled_action only after sending model buffer
        if self.train_model:
            if self.parallel:
                if self.counter % 200 == 0: self._refresh_wait()     
            else:
                self.status = self._train_model()
            if self.status["finish"]:                 
                if self.rank == 0 and self.train_model: 
                    self._logger.info("Finish training model")
                self.train_model = False   

        if self.rank == 0 and int(time.strftime("%M")) // 10 != self.ckp_start_time:
            self.save_ckp()
            self.ckp_start_time = int(time.strftime("%M")) // 10
        
        info["model_status"] = self.status
        self.counter += 1
        return state, reward, done, info      

    def _write_send_model_buffer(self, state, reward, done, info, primary_action, action_prob):
        real_step_mask = (info["step_status"] == 0) | (info["step_status"] == 3)
        data = {
                "baseline": info["baseline"][real_step_mask],
                "action": primary_action[real_step_mask],            
                "reward": reward[real_step_mask],
                "done": done[real_step_mask],
                "truncated_done": info["truncated_done"][real_step_mask],
                "real_state": info["real_states_np"][real_step_mask.cpu().numpy()],
            }       

        per_state = info["initial_per_state"] if "initial_per_state"  in info else {}
        for k in per_state.keys():
            if not k.startswith("per"): continue
            data[k] = per_state[k][real_step_mask]   

        if action_prob is not None:
            action_prob = action_prob[real_step_mask]
            if not self.tuple_action: action_prob = action_prob.unsqueeze(-2)        
            data["action_prob"] = action_prob
        
        data = util.dict_map(data, lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x)    
        for k in ["baseline", "reward"]: data[k] = data[k].astype(np.float32)
        if self.frame_stack_n > 1:
            data["real_state"] = data["real_state"][:, -self.frame_ch:]
        idx = np.arange(self.env_n)[real_step_mask.detach().cpu().numpy()]
        self.model_buffer.write.remote(ray.put(data), rank=self.rank, idx=idx, priority=None) 
    
    def to_raw_action(self, sampled_raw_action, action, action_prob):
        B, M, D = sampled_raw_action.shape
        assert M == self.sample_n
        assert D == self.dim_actions

        # Get the selected raw action for each batch instance
        raw_action = sampled_raw_action[torch.arange(B, device=self.device), action] # shape (B, D)
        if action_prob is not None:
            # Compute the probability of selecting each raw action
            raw_action_prob = torch.zeros(B, self.dim_actions, self.num_actions, device=self.device)
            for n in range(self.num_actions):
                mask = (sampled_raw_action == n).float()  # Create a mask where the raw action is n; shape (B, M, D)
                # action_prob has shape (B, M)
                # raw_action_prob has shape (B, D, N)
                raw_action_prob[:, :, n] = torch.sum(mask * action_prob.unsqueeze(-1), dim=1)
        else:
            raw_action_prob = None
        return raw_action, raw_action_prob

    def _refresh_wait(self):
        self._update_status()
        if self.status["running"]: self._refresh_net()
        if self.status["processed_n"] < self.status["warm_up_n"] * 2: return         
        while self.status["replay_ratio"] < self.flags.min_replay_ratio and not self.status["finish"]:
            time.sleep(0.01)
            self._update_status()
        return 

    def _train_model(self):
        with torch.set_grad_enabled(True):
            beta = self.model_learner.compute_beta()
            while True:            
                data = self.model_buffer.read(beta)            
                self.model_learner.init_psteps(data)                  
                if data is None: 
                    self.model_learner.log_preload(self.model_buffer.get_status())
                    break
                self.model_learner.update_real_step(data)                        
                if (self.model_learner.step_per_transition() > 
                    self.flags.max_replay_ratio):
                    break   
                self.model_learner.consume_data(data, model_buffer=self.model_buffer)
            if self.model_learner.real_step >= self.flags.total_steps:
                self.model_buffer.set_finish()
        return self.model_buffer.get_status()

    def normalize(self, x):
        if self.flags.wrapper_type == 1:
            return self.env.normalize(x)
        else:
            return self.model_net.normalize(x)

    def unnormalize(self, x):
        if self.flags.wrapper_type == 1:
            return self.env.unnormalize(x)
        else:
            return self.model_net.unnormalize(x)

    def render(self, *args, **kwargs):  
        return self.env.render(*args, **kwargs)

    def close(self):
        if self.parallel:
            self.model_buffer.set_finish.remote()
        del self.model_net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.env.close()
    
    def decode_tree_reps(self, tree_reps):
        if self.flags.wrapper_type in [3, 4, 5]:
            return self.env.decode_tree_reps(tree_reps)
        return util.decode_tree_reps(
            tree_reps=tree_reps,
            num_actions=self.num_actions if not self.sample else self.sample_n,
            dim_actions=self.dim_actions,
            sample_n=self.flags.sample_n,
            rec_t=self.flags.rec_t,
            enc_type=self.flags.model_enc_type,
            f_type=self.flags.model_enc_f_type,
        )
    
    def get_tree_rep_meaning(self):
        if not hasattr(self, "tree_rep_meaning") or self.tree_rep_meaning is None:
            if self.flags.wrapper_type in [3, 4, 5]:
                self.tree_rep_meaning = self.env.tree_rep_meaning
            elif self.flags.wrapper_type in [0, 2]:
                self.tree_rep_meaning = util.slice_tree_reps(self.num_actions, self.dim_actions, self.flags.sample_n, self.flags.rec_t)        
        return self.tree_rep_meaning
    
    def save_ckp(self):
        data = self.env.save_ckp()
        if len(data) > 0:
            np.savez(self.ckp_env_path, **data)

def make(*args, **kwargs):
    return Env(*args, **kwargs)
