import matplotlib.pyplot as plt
import numpy as np
plt.imshow(np.random.rand(10,10,3))

import gym
import gym_csokoban

env = gym.make("cSokoban-v0")
obs = env.reset()
obs, reward, done, _ = env.step(2)

plt.imshow(obs)
plt.show()
