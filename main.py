import gym
import numpy as np
import numpy.random as rd
from agent_dqn import AgentDqn, Sample
from tensorboardX import SummaryWriter
from summary_dir import summary_dir

logdir = summary_dir()
writer = SummaryWriter(logdir=logdir)
print(f"Data will be logged in '{logdir}'")

env = gym.make('Taxi-v3').unwrapped
action_space = range(env.action_space.n)
def decode(s):
    return list(env.decode(s))

agt = AgentDqn(action_space=action_space, train_interval=100, n_minibatch=10, alpha=0.01, gamma=0.99, lr=0.01)

for i in range(10000):
    s = env.reset()
    env.render()

    R = 0
    step = 0
    for j in range(3000):
        state = decode(s)
        action = agt(state)

        s1, reward, done, _ = env.step(action)
        next_state = decode(s1)

        sample = Sample(state, action, reward, next_state, done)
        agt.train(sample)

        R += reward
        s = s1
        step += 1

        if done:
            break

    env.render()
    env.close()
    print(f'i = {i}, R = {R}, step = {step}')
    writer.add_scalar('status/R', R, i)
    writer.add_scalar('status/step', step, i)
