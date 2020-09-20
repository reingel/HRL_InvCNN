import numpy as np
import numpy.random as rd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


Sample = namedtuple('Sample', 'state, action, reward, next_state, done')

class EpsilonGreedy():
    def __init__(self, action_space, iv=1.0, fac=0.99, mv=0.05):
        self.action_space = action_space
        self.value = iv
        self.factor = fac
        self.min_value = mv
    
    def __call__(self, action):
        if self.value < rd.rand():
            chosen_action = rd.choice(self.action_space)
            return chosen_action
        else:
            return action

        self.value = max(self.value * self.factor, self.min_value)


class ReplayMemory():
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []

    def reset(self):
        self.memory = []
    
    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, i):
        return self.memory[i]
    
    def append(self, sample):
        self.memory.append(sample)
    
    def sample(self, n=100):
        return random.sample(self.memory, n)


class Qnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.lay1 = nn.Linear(4, 32)
        self.lay2 = nn.Linear(32, 128)
        self.lay3 = nn.Linear(128, 6)
    
    def forward(self, s):
        u = torch.FloatTensor(s)

        x = torch.tanh(self.lay1(u))
        x = torch.tanh(self.lay2(x))
        Q = self.lay3(x)

        return Q

class AgentDqn():
    def __init__(self, action_space=range(5), train_interval = 1000, n_minibatch=100, alpha=0.01, gamma=0.99, lr=0.01):
        self.train_interval = train_interval
        self.n_minibatch = n_minibatch
        self.alpha = alpha
        self.gamma = gamma

        self.target_network = Qnet()
        self.behavior_network = Qnet()
        self.optimizer = torch.optim.SGD(self.behavior_network.parameters(), lr=lr)

        self.buffer = ReplayMemory(capacity=10000)
        self.explorer = EpsilonGreedy(action_space, iv=1.0, fac=0.999, mv=0.05)
    
    def reset(self):
        self.buffer.reset()

    def __call__(self, state):
        Q = self.behavior_network(state)
        action = np.argmax(Q.detach().numpy())
        chosen = self.explorer(action)

        return chosen
    
    def train(self, sample, i_episode):
        self.buffer.append(sample)

        if i_episode + 1 % self.train_interval == 0:
            samples = random.sample(self.buffer, self.n_minibatch)

            for sample in samples:
                s, a, r, s1, done = sample
                print(s,a,r,s1,done)

                U = r
                if not done:
                    U += self.gamma * torch.max(self.target_network(s1).detach())

                Q = self.behavior_network(s)[a]
                loss = (U - Q)**2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            for bparam, tparam in zip(self.behavior_network.parameters(), self.target_network.parameters()):
                tparam.data.copy_(self.alpha * bparam.data + (1 - self.alpha) * tparam.data)
            

if __name__ == '__main__':
    agent = AgentDqn()

    s = (1, 1, 0, 1)

    a = agent(s)

    print(a)