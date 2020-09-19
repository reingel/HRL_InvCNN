import gym
import numpy as np

env = gym.make('Taxi-v3')
s = env.reset()

for i in range(1000):
    a = env.action_space.sample()
    s, r, done, _ = env.step(a)
    if i % 10 == 0:
        env.render()
        print(r)
    if done:
        break
