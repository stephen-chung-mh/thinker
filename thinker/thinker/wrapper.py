from collections import deque
import numpy as np
import cv2
import torch
import gym
from gym import spaces

class DummyWrapper(gym.Wrapper):
    """DummyWrapper that represents the core wrapper for the real env;
    the only function is to convert returning var into tensor
    and reset the env when it is done.
    """
    def __init__(self, env, env_n, flags, model_net, device=None, time=False):   
        gym.Wrapper.__init__(self, env)
        self.env_n = env_n
        self.flags = flags
        self.device = torch.device("cpu") if device is None else device 
        self.observation_space = spaces.Dict({
            "real_states": self.env.observation_space,
        })        
        self.norm_low = self.env.observation_space.low[0]
        self.norm_high = self.env.observation_space.high[0]

    def reset(self, model_net):
        obs = self.env.reset()
        obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
        states = {"real_states": obs_py,
                  }       
        if self.flags.return_x:
            with torch.set_grad_enabled(False):
                states["xs"] = (obs_py.float() - self.norm_low) / \
                            (self.norm_high -  self.norm_low)
        return states 

    def step(self, action, model_net):  
        # action in shape (B, *) or (B,)
        if len(action.shape) > 1:
            action = action[:, 0]
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()

        obs, reward, done, info = self.env.step(action) 
        if np.any(done):
            done_idx = np.arange(self.env_n)[done]
            obs_reset = self.env.reset(done_idx)
            
        real_done = [m["real_done"] if "real_done" in m else done[n] for n, m in enumerate(info)]
        truncated_done = [m["truncated_done"] if "truncated_done" in m else False for n, m in enumerate(info)]
        obs_py = torch.tensor(obs, dtype=torch.uint8, device=self.device)
        if np.any(done):
            obs_py[done] = torch.tensor(obs_reset, dtype=torch.uint8, device=self.device)

        states = {
            "real_states": obs_py,
        }     
  
        info = {"real_done": torch.tensor(real_done, dtype=torch.bool, device=self.device),
                "truncated_done": torch.tensor(truncated_done, dtype=torch.bool, device=self.device),                
                "step_status": torch.zeros(self.env_n, dtype=torch.long, device=self.device),
                }
        if self.flags.return_x:
            with torch.set_grad_enabled(False):
                states["xs"] = (obs_py.float() - self.norm_low) / \
                            (self.norm_high -  self.norm_low)
        
        return (states, 
                torch.tensor(reward, dtype=torch.float32, device=self.device), 
                torch.tensor(done, dtype=torch.bool, device=self.device), 
                info)
    
    def unnormalize(self, x):
        assert x.dtype == torch.float or x.dtype == torch.float32
        ch = x.shape[-3]
        x = torch.clamp(x, 0, 1)
        x = x * (self.norm_high[-ch:] -  self.norm_low[-ch:]) + self.norm_low[-ch:]
        return x
    
class PostWrapper(gym.Wrapper):
    """Wrapper for recording episode return, clipping rewards"""
    def __init__(self, env, reward_clip):
        gym.Wrapper.__init__(self, env)
        self.reset_called = False
        self.reward_clip = reward_clip
    
    def reset(self, model_net):
        state = self.env.reset(model_net)
        self.device = state["real_states"].device
        self.env_n = state["real_states"].shape[0]

        self.episode_step = torch.zeros(
            self.env_n, dtype=torch.long, device=self.device
        )

        self.episode_return = {}
        for key in ["re", "im", "cur"]:
            self.episode_return[key] = torch.zeros(
                self.env_n, dtype=torch.float, device=self.device
            )
        self.reset_called = True
        return state

    def step(self, action, model_net):
        assert self.reset_called, "need to call reset ONCE before step"
        state, reward, done, info = self.env.step(action, model_net)
        real_done = info["real_done"]

        self.episode_step += 1
        info["episode_step"] = self.episode_step.clone()
        self.episode_step[real_done] = 0

        self.episode_return["re"] += reward
        info["episode_return"] = self.episode_return["re"].clone()
        self.episode_return["re"][real_done] = 0.

        for prefix in ["im", "cur"]:
            if prefix+"_reward" in info:
                self.episode_return[prefix] += info[prefix+"_reward"]
                info[prefix + "_episode_return"] = self.episode_return[prefix].clone()
                self.episode_return[prefix][real_done] = 0.
                if prefix == "im":
                    self.episode_return[prefix][info["step_status"] == 0] = 0.
        
        if self.reward_clip > 0.:
            reward = torch.clamp(reward, -self.reward_clip, +self.reward_clip)
        return state, reward, done, info

def PreWrapper(env, name, grayscale=False, frame_wh=96):
    if "Sokoban" in name:
        env = TransposeWrap(env)
    else:
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
            frame_wh=frame_wh,
        )
        env = TransposeWrap(env)
    return env

# Standard wrappers

class TransposeWrap(gym.ObservationWrapper):
    """Image shape to channels x weight x height"""

    def __init__(self, env):
        super(TransposeWrap, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
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
        self.was_real_done = True

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
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
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

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

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
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

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
