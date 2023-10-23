import gym
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from .csokoban import cSokoban
import numpy as np
import pkg_resources
import os 

class SokobanEnv(gym.Env):
    def __init__(self, difficulty='unfiltered', small=True, seed=0):
        if difficulty == 'unfiltered': 
            level_num = 900000                      
            path = '/'.join(('boxoban-levels', difficulty, 'train'))
        elif difficulty == 'test':     
            level_num = 1000                  
            path = '/'.join(('boxoban-levels', 'unfiltered', 'test'))
        elif difficulty == 'medium':            
            level_num = 50000           
            path = '/'.join(('boxoban-levels', difficulty, 'valid'))
        elif difficulty == 'hard': 
            level_num = 3332         
            path = '/'.join(('boxoban-levels', difficulty))
        else:
            raise Exception(f"difficulty {difficulty} not accepted.")

        level_dir = pkg_resources.resource_filename(__name__, path)
        img_dir = pkg_resources.resource_filename(__name__, 'surface')
        
        self.sokoban = cSokoban(small=small, 
                                level_dir=level_dir.encode('UTF-8'), 
                                img_dir=img_dir.encode('UTF-8'), 
                                level_num=level_num, 
                                seed=seed)
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=255, shape=(self.sokoban.obs_x, self.sokoban.obs_y, 3), dtype=np.uint8)
        # self.sokoban.reset()

    def step(self, action):
        obs, reward, done, info = self.sokoban.step(action)
        reward = round(reward, 2)
        return obs, reward, done, info

    def reset(self, room_id=None):
        if room_id is None:
            return self.sokoban.reset()
        else:
            return self.sokoban.reset_level(room_id)   
        
    def clone_state(self):
        return self.sokoban.clone_state()

    def restore_state(self, state):
        return self.sokoban.restore_state(state)    

    def seed(self, seed): 
        self.sokoban.seed(seed)    

    @property
    def step_n(self):
        return self.sokoban.step_n

    @step_n.setter
    def step_n(self, step_n):
        self.sokoban.step_n = step_n  