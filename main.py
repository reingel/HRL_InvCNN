import gym
import numpy as np
import numpy.random as rd
from agent_dnn import AgentDnn

env = gym.make('Taxi-v3').unwrapped
actions = range(env.action_space.n)

agt = AgentDnn()

epsilon = 0.1

states = []
actions = []
rewards = []
next_states = []
dones = []

for i in range(1000):
    s = env.reset()
    env.render()

    R = 0
    step = 0
    for j in range(1000):
        state = list(env.decode(s))
        if rd.rand() < epsilon:
            action = env.action_space.sample()
        else:
            Q = agt(state)
            action = np.argmax(Q.detach().numpy())

        s1, reward, done, _ = env.step(action)
        next_state = list(env.decode(s1))

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

        if done or step % 10 == 0:
            agt.train(states, actions, rewards, next_states, dones)
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

        R += reward
        s = s1
        step += 1

        if done:
            break

    env.render()
    env.close()
    print(i, R, step)
