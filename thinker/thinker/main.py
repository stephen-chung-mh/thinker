import os
import shutil
import time
from collections import namedtuple
import ray
import torch
import thinker.util as util
from thinker.buffer import ModelBuffer, SModelBuffer, GeneralBuffer
from thinker.learn_model import ModelLearner, SModelLearner
from thinker.model_net import ModelNet
from thinker.gym_add.asyn_vector_env import AsyncVectorEnv
from thinker.wrapper import PreWrapper, DummyWrapper, PostWrapper
from thinker.cenv import cModelWrapper, cPerfectWrapper
import gym
TrainModelOut = namedtuple(
    "TrainModelOut",
    [
        "real_state",        
        "action",
        "action_prob",
        "reward",
        "done",
        "truncated_done",
        "baseline",
    ],
)

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
    model_buffer = ModelBuffer.options(num_cpus=1).remote(flags)    
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
        self.status = 0 # 0 for model being warmed up; 
                        # 1 for refilling model after loading checkpoint
                        # 2 for model learner running
                        # 3 for model learner finished training

        self._logger.info(
            "Initializing env %d with device %s"
            % (
                self.rank,
                "cuda" if self.device == torch.device("cuda") else "cpu",
            )
        )

        if env_fn is None:
            if name == "Sokoban-v0": import gym_sokoban
            env_fn = lambda: PreWrapper(gym.make(name),  name=name,)            

        # initialize a single env to collect env information
        env = env_fn()
        assert len(env.observation_space.shape) == 3, \
            f"env.observation_space should be 3d, not {env.observation_space.shape}"
        assert type(env.action_space) == gym.spaces.discrete.Discrete, \
            f"env.action_space should be Discrete, not {type(env.action_space)}"          

        self.real_state_space  = env.observation_space
        self.real_state_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.frame_stack_n = env.frame_stack_n if hasattr(env, "frame_stack_n") else 1
        if self.rank == 0 and self.frame_stack_n > 1:
            self._logger.info("Detected frame stacking with %d counts" % self.frame_stack_n)
        env.close()

        # initalize model
        self.has_model = self.flags.wrapper_type != 1
        self.train_model = self.has_model and self.flags.train_model 
        self.require_prob = False
        if self.has_model:
            model_param = {
                "obs_space": self.real_state_space,
                "num_actions": self.num_actions, 
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
                    ).remote(ray_obj, model_param, self.flags)
                    # start learning
                    self.r_learner = self.model_learner.learn_data.remote()
                else:
                    self.model_learner = SModelLearner(ray_obj=None, model_param=model_param,
                        flags=self.flags, model_net=self.model_net, device=self.device)
                    self.model_buffer = SModelBuffer(flags=self.flags)
            if self.train_model: self.require_prob = self.flags.require_prob
        else:
            self.model_net = None            
            
        # create batched asyn. environments
        env = AsyncVectorEnv([env_fn for _ in range(env_n)]) 
        env.seed([i for i in range(
            self.rank * env_n + self.flags.base_seed, 
            self.rank * env_n + self.flags.base_seed + env_n)])       

        if self.flags.wrapper_type == 0:
            core_wrapper = cModelWrapper
        elif self.flags.wrapper_type == 1:
            core_wrapper = DummyWrapper
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
                        time=False)
        
        # wrap the env with a wrapper that computes episode
        # return and episode step for logging purpose;
        # also clip the reward afterwards if set
        env = PostWrapper(env, 
                        reward_clip=self.flags.reward_clip) 
        gym.Wrapper.__init__(self, env)    

        # create local buffer for transitions

        if self.train_model:
            self.model_local_buffer = [
                self._empty_local_buffer(),
                self._empty_local_buffer(),
            ]
            self.model_n = 0
            self.model_t = 0                          

        if self.train_model:
            if self.flags.parallel:
                self.status_ptr = self.model_buffer.get_status.remote()        
                self.status = ray.get(self.status_ptr)
                self.status_ptr = self.model_buffer.get_status.remote()    
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
                    shutil.copyfile(path, os.path.join(self.ckpdir, "ckp_model.tar"))
            if path is not None:                
                checkpoint = torch.load(
                    path, map_location=torch.device("cpu")
                )
                self.model_net.set_weights(
                    checkpoint["model_net_state_dict"]
                )
                self._logger.info("Loaded model net from %s" % path)
            
            if self.train_model and self.parallel:
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
    
    def _empty_local_buffer(self):
        pre_shape = (
            self.flags.buffer_traj_len + 2 * self.flags.model_unroll_len,
            self.env_n,
        )
        return TrainModelOut(
            real_state=torch.zeros(
                pre_shape + self.real_state_shape,
                dtype=torch.uint8,
                device=self.device,
            ),
            action_prob=torch.zeros(
                pre_shape + (self.num_actions,),
                dtype=torch.float32,
                device=self.device,
            ),
            action=torch.zeros(pre_shape, dtype=torch.long, device=self.device),
            reward=torch.zeros(pre_shape, dtype=torch.float, device=self.device),
            done=torch.ones(pre_shape, dtype=torch.bool, device=self.device),
            truncated_done=torch.ones(pre_shape, dtype=torch.bool, device=self.device),
            baseline=torch.zeros(pre_shape, dtype=torch.float, device=self.device),
        )
    
    def _write_single_model_buffer(self, n, t, state, reward, done, info,
                                  action, action_prob):
        
        if not torch.is_tensor(action):
            action = torch.tensor(action, device=self.device)
        if action_prob is not None and not torch.is_tensor(action_prob):
            action_prob = torch.tensor(action_prob, device=self.device)

        self.model_local_buffer[n].real_state[t] = state["real_states"]
        self.model_local_buffer[n].action[t] = action
        if self.flags.require_prob:
            self.model_local_buffer[n].action_prob[t] = action_prob        
        self.model_local_buffer[n].reward[t] = reward
        self.model_local_buffer[n].done[t] = done
        self.model_local_buffer[n].truncated_done[t] = info["truncated_done"]        
        self.model_local_buffer[n].baseline[t] = info["baseline"]

    def _write_send_model_buffer(
        self, state, reward, done, info, action, action_prob
    ):
        n, t, cap_t, k = (
            self.model_n,
            self.model_t,
            self.flags.buffer_traj_len,
            self.flags.model_unroll_len,
        )
        self._write_single_model_buffer(n, t, state, reward, done, info, action, action_prob)

        if t >= cap_t:
            # write the beginning of another buffer
            self._write_single_model_buffer(
                1 - n, t - cap_t, state, reward, done, info, action, action_prob
            )

        if t >= cap_t + 2 * k - 2:
            # finish writing a buffer, send it
            send_model_local_buffer = util.tuple_map(
                self.model_local_buffer[n], lambda x: x.cpu().numpy()
            )
            if self.parallel:
                self.model_buffer.write.remote(ray.put(send_model_local_buffer), self.rank)
            else:
                self.model_buffer.write(send_model_local_buffer, self.rank)
            self.model_local_buffer[n] = self._empty_local_buffer()
            self.model_n = 1 - n
            self.model_t = t - cap_t + 1
        else:
            self.model_t += 1

    def _update_status(self):
        status = ray.get(self.status_ptr)
        self.status_ptr = self.model_buffer.get_status.remote()        
        return status

    def reset(self):
        state = self.env.reset(self.model_net)
        return state

    def step(self, action, reset, action_prob=None, ignore=False):        
        if self.require_prob and not ignore: 
            assert action_prob is not None and action_prob.shape == (self.env_n, self.num_actions), \
                    f"action_prob should have shape f{(self.env_n, self.num_actions)}"
        with torch.set_grad_enabled(False):
            state, reward, done, info = self.env.step((action, reset), self.model_net)        
        if self.train_model and info["step_status"][0] == 0 and not ignore: # assume all transition in same step within a stage
            self._write_send_model_buffer(state, reward, done, info, action, action_prob)        
        if self.train_model:
            if self.parallel:
                if self.counter % 200 == 0: self.status = self._refresh_wait()     
            else:
                self.status = self._train_model()
            if self.status["finish"]:                 
                if self.rank == 0 and self.train_model: 
                    self._logger.info("Finish training model")
                self.train_model = False      
        info["model_status"] = self.status
        self.counter += 1
        return state, reward, done, info       
    
    def _refresh_wait(self):
        status = self._update_status()
        if status["running"]: self._refresh_net()
        signals = ray.get(self.signal_ptr)
        # if model-learning thread is legging behind, need to wait for it to catch up
        self.signal_ptr = self.signal_buffer.get_data.remote("self_play_signals")
        while (signals is not None and "halt" in signals and signals["halt"]):
            time.sleep(0.1)
            signals = ray.get(self.signal_ptr)
            self.signal_ptr = self.signal_buffer.get_data.remote("self_play_signals")
        return status

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
   
    def close(self):
        if self.parallel:
            self.model_buffer.set_finish.remote()
        del self.model_net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.env.close()

    def unnormalize(self, x):
        if self.flags.wrapper_type == 1:
            return self.env.unnormalize(x)
        else:
            return self.model_net.unnormalize(x)

def make(*args, **kwargs):
    return Env(*args, **kwargs)
