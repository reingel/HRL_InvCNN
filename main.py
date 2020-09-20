import gym
import numpy as np
import numpy.random as rd
from agent_dqn import AgentDqn, Sample

env = gym.make('Taxi-v3').unwrapped
action_space = range(env.action_space.n)

agt = AgentDqn(action_space=action_space, train_interval=100, n_minibatch=10, alpha=0.01, gamma=0.99, lr=0.01)

for i in range(1000):
    s = env.reset()
    env.render()

    R = 0
    step = 0
    for j in range(1000):
        state = list(env.decode(s))
        action = agt(state)

        s1, reward, done, _ = env.step(action)
        next_state = list(env.decode(s1))

        sample = Sample(state, action, reward, next_state, done)

        agt.train(sample, i)

        R += reward
        s = s1
        step += 1

        if done:
            break

    env.render()
    env.close()
    print(f'i = {i}, R = {R}, step = {step}')
