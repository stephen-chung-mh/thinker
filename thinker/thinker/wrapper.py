from collections import deque
import numpy as np
import cv2
import torch
import gym
from gym import spaces
import thinker.util as util
import time

class DummyWrapper(gym.Wrapper):
    """DummyWrapper that represents the core wrapper for the real env;
    the only function is to convert returning var into tensor
    and reset the env when it is done.
    """
    def __init__(self, env, env_n, flags, model_net, device=None, timing=False):   
        gym.Wrapper.__init__(self, env)
        self.env_n = env_n
        self.flags = flags
        self.device = torch.device("cpu") if device is None else device 
        self.observation_space = spaces.Dict({
            "real_states": self.env.observation_space,
        })        
        if env.observation_space.dtype == 'uint8':
            self.state_dtype = torch.uint8
        elif env.observation_space.dtype == 'float32':
            self.state_dtype = torch.float32
        else:
            raise Exception(f"Unupported observation sapce", env.observation_space)

        self.train_model = self.flags.train_model
        action_space =  env.action_space[0]
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(action_space)

    def reset(self, model_net):
        obs = self.env.reset()
        obs_py = torch.tensor(obs, dtype=self.state_dtype, device=self.device)                
        if self.train_model: 
            self.per_state = model_net.initial_state(batch_size=self.env_n, device=self.device)
            pri_action = torch.zeros(self.env_n, self.dim_actions, dtype=torch.long, device=self.device)
            done = torch.zeros(self.env_n, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                model_net_out = model_net(
                    env_state=obs_py, 
                    done=done,
                    actions=pri_action.unsqueeze(0), 
                    state=self.per_state,)       
            self.per_state = model_net_out.state
            self.baseline = model_net_out.vs[-1]
        states = {"real_states": obs_py}       
        return states 

    def step(self, action, model_net):  
        # action in shape (B, *) or (B,)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()        

        obs, reward, done, info = self.env.step(action) 
        if np.any(done):
            done_idx = np.arange(self.env_n)[done]
            obs_reset = self.env.reset(idx=done_idx)
            obs[done] = obs_reset
        obs_py = torch.tensor(obs, dtype=self.state_dtype, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)        
        states = {
            "real_states": obs_py,
        }     

        info = util.dict_map(info, lambda x: torch.tensor(x, device=self.device))
        info["step_status"] = torch.full((self.env_n,), fill_value=3, dtype=torch.long, device=self.device)
        info["real_states_np"] = obs
        
        if self.train_model:             
            info["initial_per_state"] = self.per_state
            info["baseline"] = self.baseline
            pri_action = torch.tensor(action, dtype=torch.long, device=self.device)
            if not self.tuple_action: pri_action = pri_action.unsqueeze(-1)          
            with torch.no_grad():
                model_net_out = model_net(
                    env_state=obs_py, 
                    done=done,
                    actions=pri_action.unsqueeze(0), 
                    state=self.per_state,)       
                self.per_state = model_net_out.state
                self.baseline = model_net_out.vs[-1]
        
        return states, reward, done, info
    
class PostWrapper(gym.Wrapper):
    """Wrapper for recording episode return, clipping rewards"""
    def __init__(self, env, flags, device):
        gym.Wrapper.__init__(self, env)
        self.reset_called = False        
        low = torch.tensor(self.env.observation_space["real_states"].low[0])
        high = torch.tensor(self.env.observation_space["real_states"].high[0])
        self.need_norm = torch.isfinite(low).all() and torch.isfinite(high).all()
        self.norm_low = low
        self.norm_high = high

        self.disable_thinker = flags.wrapper_type == 1
        if not self.disable_thinker:
            self.pri_action_space = self.env.action_space[0][0]            
        else:
            self.pri_action_space = self.env.action_space[0]
        self.num_actions, self.dim_actions, self.dim_rep_actions, self.tuple_action, self.discrete_action = \
            util.process_action_space(self.pri_action_space)
        if not self.discrete_action:
            self.action_space_low = torch.tensor(self.pri_action_space.low, dtype=torch.float, device=device)
            self.action_space_high = torch.tensor(self.pri_action_space.high, dtype=torch.float, device=device)
    
    def reset(self, model_net):
        state = self.env.reset(model_net)
        self.device = state["real_states"].device
        self.env_n = state["real_states"].shape[0]

        self.episode_step = torch.zeros(
            self.env_n, dtype=torch.long, device=self.device
        )

        self.episode_return = {}
        for key in ["im", "cur"]:
            self.episode_return[key] = torch.zeros(
                self.env_n, dtype=torch.float, device=self.device
            )
        self.reset_called = True
        return state

    def step(self, action, model_net):
        assert self.reset_called, "need to call reset ONCE before step"

        if not self.discrete_action:            
            if not self.disable_thinker:
                pri_action, reset_action = action
                pri_action = ((pri_action + 1) / 2) * (self.action_space_high - self.action_space_low) + self.action_space_low
                pri_action = torch.clamp(pri_action, -1, +1)
                action = (pri_action, reset_action)
            else:
                action = torch.clamp(action, self.action_space_low, self.action_space_high)

        state, reward, done, info = self.env.step(action, model_net)
        real_done = info["real_done"]        

        for prefix in ["im", "cur"]:
            if prefix+"_reward" in info:
                nan_mask = ~torch.isnan(info[prefix+"_reward"])
                self.episode_return[prefix][nan_mask] += info[prefix+"_reward"][nan_mask]
                info[prefix + "_episode_return"] = self.episode_return[prefix].clone()
                self.episode_return[prefix][real_done] = 0.
                if prefix == "im":
                    self.episode_return[prefix][info["step_status"] == 0] = 0.        
        return state, reward, done, info
    
    def render(self, *args, **kwargs):  
        return self.env.render(*args, **kwargs)    
    
    def unnormalize(self, x):
        assert x.dtype == torch.float or x.dtype == torch.float32
        if self.need_norm:
            ch = x.shape[-3]
            x = torch.clamp(x, 0, 1)
            x = x * (self.norm_high[-ch:] -  self.norm_low[-ch:]) + self.norm_low[-ch:]
        return x
    
    def normalize(self, x):
        if self.need_norm:    
            if self.norm_low.device != x.device or self.norm_high.device != x.device:
                self.norm_low = self.norm_low.to(x.device)
                self.norm_high = self.norm_high.to(x.device)
            x = (x.float() - self.norm_low) / (self.norm_high -  self.norm_low)
        return x

def PreWrapper(env, name, flags):
    grayscale = flags.grayscale
    discrete_k = flags.discrete_k 
    repeat_action_n = flags.repeat_action_n
    rand_action_eps = flags.rand_action_eps
    sokoban_pomdp = flags.sokoban_pomdp
    atari = flags.atari
    
    if sokoban_pomdp: env = Sokoban_POMDP(env)
    if discrete_k > 0: env = DiscretizeActionWrapper(env, K=discrete_k)
    if repeat_action_n > 0: env = RepeatActionWrapper(env, repeat_action_n=repeat_action_n)      
    if rand_action_eps > 0.: env = RandomZeroActionWrapper(env, eps=rand_action_eps)
    if "NoFrameskip" in name and not atari: 
        raise Exception(f"{name} is likely an Atari game but flags.Atari is False")
    if atari: 
        # atari
        env = StateWrapper(env)
        env = TimeLimit_(env, max_episode_steps=108000)
        env = NoopResetEnv(env, noop_max=30)
        if "NoFrameskip" in name:
            env = MaxAndSkipEnv(env, skip=4)
        env = wrap_deepmind(
            env,
            episode_life=True,
            clip_rewards=False,
            frame_stack=True,
            scale=False,
            grayscale=grayscale,
            frame_wh=96,
        )
    if env.observation_space.dtype == np.float64:
        env = ScaledFloatFrame(env)
    if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 3:
        #old_env_obs_space = env.observation_space.shape        
        # 3d input, need transpose
        env = TransposeWrap(env)      
        #new_env_obs_space = env.observation_space.shape  
        #print(f"Added transpose wrapper for {old_env_obs_space} => {new_env_obs_space}")
    return env

def create_env_fn(name, flags):
    if "Sokoban" in name:
        import gym_sokoban
        fn = gym.make
        args = {"id": name}
    else:
        fn = gym.make
        args = {"id": name}

    env_fn = lambda: PreWrapper(
        fn(**args), 
        name=name, 
        flags=flags,
    )
    return env_fn
    

# Standard wrappers

class TransposeWrap(gym.ObservationWrapper):
    """Image shape to channels x weight x height"""

    def __init__(self, env):
        super(TransposeWrap, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.transpose(2, 0, 1),
            high=self.observation_space.high.transpose(2, 0, 1),
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))

class NoopWrapper(gym.Wrapper):
    def __init__(self, env, cost=0.0):
        gym.Wrapper.__init__(self, env)
        env.action_space.n += 1
        self.cost = cost

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # obs = obs[np.newaxis, :, :, :]
        self.last_obs = obs
        return obs

    def step(self, action):
        if action == 0:
            return self.last_obs, self.cost, False, {}
        else:
            obs, reward, done, info = self.env.step(action - 1)
            # obs = obs[np.newaxis, :, :, :]
            self.last_obs = obs
            return obs, reward, done, info

    def get_action_meanings(self):
        return [
            "NOOP",
        ] + self.env.get_action_meanings()

    def clone_state(self):
        state = self.env.clone_state()
        state["noop_last_obs"] = np.copy(self.last_obs)
        return state

    def restore_state(self, state):
        self.last_obs = np.copy(state["noop_last_obs"])
        self.env.restore_state(state)
        return

class TimeLimit_(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit_, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["truncated_done"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def clone_state(self):
        state = self.env.clone_state()
        state["timeLimit_elapsed_steps"] = self._elapsed_steps
        return state

    def restore_state(self, state):
        self._elapsed_steps = state["timeLimit_elapsed_steps"]
        self.env.restore_state(state)
        return

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=False, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

# Atari-related wrapped (taken from torchbeast)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = False
        self.was_done = False
        self.init = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        info["real_done"] = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        self.was_done = done
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done or not self.was_done:
            obs = self.env.reset(**kwargs)
            if not self.was_done and self.init:
                #print("Warning: Resetting when episode is not done.")
                pass
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        self.init = True
        return obs

    def clone_state(self):
        state = self.env.clone_state()
        state["eps_life_vars"] = [self.lives, self.was_real_done]
        return state

    def restore_state(self, state):
        self.lives, self.was_real_done = state["eps_life_vars"]
        self.env.restore_state(state)
        return state

class DoneEnv(gym.Wrapper):
    def __init__(self, env):
        """Always done=True"""
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, True, info

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype,
        )
        self.frame_stack_n = k

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        # return np.concatenate(list(self.frames), axis=-1)
        return LazyFrames(list(self.frames))

    def clone_state(self):
        state = self.env.clone_state()
        state["frameStack"] = [np.copy(i) for i in self.frames]
        return state

    def restore_state(self, state):
        for i in state["frameStack"]:
            self.frames.append(i)
        self.env.restore_state(state)

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

class StateWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def clone_state(self):
        state = self.env.clone_state()
        # state = self.env.clone_state(include_rng=True)
        return {"ale_state": state}

    def restore_state(self, state):
        # self.env.restore_state(state["ale_state"])
        self.env.restore_state(state["ale_state"])

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):        
        """
        An environment wrapper that scales observations from uint8 to float32 and
        normalizes them if they are uint8. If observations are float64, it converts them to float32 without normalization.
        """
        super(ScaledFloatFrame, self).__init__(env)
        assert self.env.observation_space.dtype in [np.uint8, np.float64]

        # Determine if the original observation space is uint8
        self.is_uint8 = self.env.observation_space.dtype == np.uint8
        # Adjust the observation space to reflect the change in dtype
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low if not self.is_uint8 else 0,
            high=self.env.observation_space.high if not self.is_uint8 else 1,
            shape=self.env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        # Convert observation to float32
        observation = np.array(observation, dtype=np.float32)        
        # Normalize only if the original observation space was uint8
        if self.is_uint8: observation = observation / 255.0
        return observation
    
class RandomZeroActionWrapper(gym.ActionWrapper):
    def __init__(self, env, eps=0.05):
        super().__init__(env)
        self.eps = eps

    def action(self, action):
        # Check if we should randomize the action
        if np.random.rand() < self.eps:
            if isinstance(self.action_space, gym.spaces.Discrete):
                return 0  # For discrete action space, action 0
            elif isinstance(self.action_space, gym.spaces.Box):
                return np.zeros(self.action_space.shape)  # For continuous action space, vector of zeros
            else:
                raise NotImplementedError("Unsupported action space for randomization")
        return action  
    
class Sokoban_POMDP(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):        
        int_state = self.env.clone_state()
        room_status = int_state['sokoban_room_status'].reshape(10, 10)
        agent_y, agent_x = np.nonzero((room_status == 4) | (room_status == 5) | (room_status == 9))
        #if len(agent_y) == 0: print(room_status)
        agent_y, agent_x = agent_y[0], agent_x[0]
        rand_actions = [0]
        if room_status[np.clip(agent_y-1, 0, 9), np.clip(agent_x, 0, 9)] not in [0, 2, 3]:
            rand_actions += [1]
        if room_status[np.clip(agent_y+1, 0, 9), np.clip(agent_x, 0, 9)] not in [0, 2, 3]:    
            rand_actions += [2]
        if room_status[np.clip(agent_y, 0, 9), np.clip(agent_x-1, 0, 9)] not in [0, 2, 3]:    
            rand_actions += [3]    
        if room_status[np.clip(agent_y, 0, 9), np.clip(agent_x+1, 0, 9)] not in [0, 2, 3]:    
            rand_actions += [4]    
        rand_action = rand_actions[np.random.randint(len(rand_actions))]
        if rand_action != 0:
            obs_, _, done, _ = self.env.step(rand_action)
            if not done: obs = obs_
            self.env.restore_state(int_state)
        return obs    

class RepeatActionWrapper(gym.Wrapper):
    def __init__(self, env, repeat_action_n):
        super().__init__(env)
        self.repeat_action_n = repeat_action_n

        # Adjust observation space for stacked observations
        orig_obs_space = env.observation_space
        self.obs_shape = orig_obs_space.shape
        new_shape = (*self.obs_shape[:-1], self.obs_shape[-1] * repeat_action_n)
        self.observation_space = gym.spaces.Box(
            low=np.tile(orig_obs_space.low, repeat_action_n),
            high=np.tile(orig_obs_space.high, repeat_action_n),
            shape=new_shape,
            dtype=orig_obs_space.dtype
        )

        self.stacked_obs = np.zeros(new_shape, dtype=orig_obs_space.dtype)

    def reset(self):
        initial_obs = self.env.reset()
        self.stacked_obs = np.tile(initial_obs, self.repeat_action_n)
        return self.stacked_obs

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        for i in range(self.repeat_action_n):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            # Update the stacked observation
            start_index = i * self.obs_shape[-1]
            end_index = start_index + self.obs_shape[-1]
            self.stacked_obs[..., start_index:end_index] = obs

            if done:
                # Fill the remaining slots with the last observation if done
                for j in range(i + 1, self.repeat_action_n):
                    start_index = j * self.obs_shape[-1]
                    end_index = start_index + self.obs_shape[-1]
                    self.stacked_obs[..., start_index:end_index] = obs
                break

        return self.stacked_obs, total_reward, done, info

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, K=11):
        super().__init__(env)
        self.K = K  # Number of bins for discretization

        # Infer min and max actions from the original environment
        self.min_action = self.env.action_space.low
        self.max_action = self.env.action_space.high

        # Ensure the original action space is a Box
        assert isinstance(env.action_space, gym.spaces.Box), "The action space must be of type gym.spaces.Box"

        # Define the new action space as a tuple of Discrete(K) spaces
        self.action_space = spaces.Tuple([spaces.Discrete(K) for _ in range(env.action_space.shape[0])])

    def action(self, action):
        # Convert the discrete action to continuous action using vectorized operations
        action = np.array(action)  # Ensure action is a NumPy array
        discrete_to_cont = (action / (self.K - 1)) * (self.max_action - self.min_action) + self.min_action
        return discrete_to_cont


# wrapper for normalization

class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = util.RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = util.RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action, **kwargs):
        obs, rews, dones, infos = self.env.step(action, **kwargs)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, dones, infos    

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs)
        else:
            return self.normalize(np.array([obs]))[0]

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)   
    
    def clone_state(self, idx=None):
        state = self.env.clone_state(idx)
        if idx is None: idx = range(self.num_envs)
        for n, i in enumerate(idx):
            state[n]["obs_mean"] = self.obs_rms.mean
            state[n]["obs_var"] = self.obs_rms.var
            state[n]["obs_count"] = self.obs_rms.count
        return state
    
    def restore_state(self, state, idx=None):
        self.env.restore_state(state, idx)
        if idx is None: idx = range(self.num_envs)        
        self.obs_rms.mean = state[0]["obs_mean"]
        self.obs_rms.var = state[0]["obs_var"]
        self.obs_rms.count = state[0]["obs_count"]

    def load_ckp(self, data):
        self.obs_rms.mean = data['obs_mean']
        self.obs_rms.var = data['obs_var']
        self.obs_rms.count = data['obs_count']
        self.env.load_ckp(data)

    def save_ckp(self):
        data = self.env.save_ckp()
        data["obs_mean"] = self.obs_rms.mean
        data["obs_var"] = self.obs_rms.var
        data["obs_count"] = self.obs_rms.count
        return data

class NormalizeReward(gym.core.Wrapper):
    def __init__(
        self,
        env,
        gamma=0.99,
        epsilon=1e-8,
    ):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = util.RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action, **kwargs):
        obs, rews, dones, infos = self.env.step(action, **kwargs)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, dones, infos

    def normalize(self, rews):
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
    
    def clone_state(self, idx=None):
        state = self.env.clone_state(idx)
        if idx is None: idx = range(self.num_envs)
        for n, i in enumerate(idx):
            state[n]["return_mean"] = self.return_rms.mean
            state[n]["return_var"] = self.return_rms.var
            state[n]["return_count"] = self.return_rms.count
            state[n]["return_cur"] = self.returns[i]
        return state
    
    def restore_state(self, state, idx=None):
        self.env.restore_state(state, idx)
        if idx is None: idx = range(self.num_envs)        
        self.return_rms.mean = state[0]["return_mean"]
        self.return_rms.var = state[0]["return_var"]
        self.return_rms.count = state[0]["return_count"]
        for n, i in enumerate(idx):
            self.returns[i] = state[n]["return_cur"]

    def load_ckp(self, data):
        self.return_rms.mean = data['return_mean']
        self.return_rms.var = data['return_var']
        self.return_rms.count = data['return_count']
        self.env.load_ckp(data)

    def save_ckp(self):
        data = self.env.save_ckp()
        data["return_mean"] = self.return_rms.mean
        data["return_var"] = self.return_rms.var
        data["return_count"] = self.return_rms.count
        return data

    
class InfoConcat(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.is_vector_env
        self.num_envs = getattr(env, "num_envs", 1)        

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs) 
        real_done = np.array([m["real_done"] if "real_done" in m else done[n] for n, m in enumerate(info)], dtype=np.bool_)
        truncated_done = np.array([m["truncated_done"] if "truncated_done" in m else False for n, m in enumerate(info)], dtype=np.bool_)
        cost = np.array([m["cost"] if "cost" in m else False for n, m in enumerate(info)], dtype=np.bool_)
        info = {
            "real_done": real_done,
            "truncated_done": truncated_done,
            "cost": cost,
        }
        return obs, reward, done, info
    
    def default_info(self):
        info = {
            "real_done": np.zeros(self.num_envs, dtype=np.bool_),
            "truncated_done": np.zeros(self.num_envs, dtype=np.bool_),
            "cost": np.zeros(self.num_envs, dtype=np.bool_),
        }
        return info
    
    def clone_state(self, idx=None):
        return self.env.clone_state(idx)
    
    def restore_state(self, state, idx=None):
        return self.env.restore_state(state, idx)
    
    def load_ckp(self, data):
        return 
    
    def save_ckp(self):
        return {}

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)        
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_step = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, **kwargs):
        idx = kwargs.get("idx", None)     
        reset_stat = kwargs.get("reset_stat", False)  
        if "reset_stat" in kwargs: kwargs.pop("reset_stat")
        if reset_stat:
            if idx is None:
                self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
                self.episode_step = np.zeros(self.num_envs, dtype=np.int64)
            else:
                self.episode_return[idx] = 0.
                self.episode_step[idx] = 0

        return self.env.reset(**kwargs)

    def step(self, action, **kwargs):        
        idx = kwargs.get("idx", None)     
        obs, reward, done, info = self.env.step(action, **kwargs)
        real_done = info["real_done"]
        if idx is None:
            self.episode_return = self.episode_return + reward
            self.episode_step = self.episode_step + 1
        else:
            self.episode_return[idx] = self.episode_return[idx] + reward
            self.episode_step[idx] = self.episode_step[idx] + 1
        episode_return = self.episode_return
        episode_step = self.episode_step

        if np.any(real_done):
            episode_return = np.copy(episode_return)
            episode_step = np.copy(episode_step)

            if idx is None:    
                self.episode_return[real_done] = 0.
                self.episode_step[real_done] = 0
            else:
                idx_b = np.zeros(self.num_envs, np.bool_)
                idx_b[idx] = real_done
                self.episode_return[idx_b] = 0.
                self.episode_step[idx_b] = 0
                
        info["episode_return"] = episode_return[idx] if idx is not None else episode_return
        info["episode_step"] = episode_step[idx] if idx is not None else episode_return
        return obs, reward, done, info
    
    def default_info(self):
        info = self.env.default_info()
        info["episode_return"] =  np.zeros(self.num_envs, dtype=np.float32)
        info["episode_step"] =  np.zeros(self.num_envs, dtype=np.int64)
        return info
    
    def clone_state(self, idx=None):
        state = self.env.clone_state(idx)
        if idx is None: idx = range(self.num_envs)
        for n, i in enumerate(idx):
            state[n]["episode_return"] = self.episode_return[i]
            state[n]["episode_step"] = self.episode_step[i]
        return state
    
    def restore_state(self, state, idx=None):
        self.env.restore_state(state, idx)
        if idx is None: idx = range(self.num_envs)
        for n, i in enumerate(idx):
            self.episode_return[i] = state[n]["episode_return"]
            self.episode_step[i] = state[n]["episode_step"]

class TransformReward(gym.Wrapper):
    r"""Transform the reward via an arbitrary function."""
    def __init__(self, env, f):
        super().__init__(env)
        assert callable(f)
        self.f = f
    
    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        return observation, self.f(reward), done, info

class TransformObservation(gym.Wrapper):
    r"""Transform the reward via an arbitrary function."""
    def __init__(self, env, f):
        super().__init__(env)
        assert callable(f)
        self.f = f
    
    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        return self.f(observation), reward, done, info

def wrap_deepmind(
    env,
    episode_life=True,
    clip_rewards=True,
    frame_stack=False,
    scale=False,
    grayscale=False,
    frame_wh=96,
):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=frame_wh, height=frame_wh, grayscale=grayscale)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env