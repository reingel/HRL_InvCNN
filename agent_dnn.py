import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class AgentDnn():
    def __init__(self):
        self.net = Qnet()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001)
        self.gamma = 0.99
    
    def __call__(self, s):
        return self.net(s)
    
    def train(self, states, actions, rewards, next_states, dones):
        for s, a, r, s1, done in zip(states, actions, rewards, next_states, dones):
            print(s,a,r,s1,done)
            Q = self.net(s)[a]
            Q1max = torch.max(self.net(s1)).detach()
            if done:
                loss = (r - Q)**2
            else:
                loss = (r + self.gamma * Q1max - Q)**2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

if __name__ == '__main__':
    agent = AgentDnn()

    s = (1, 1, 0, 1)

    a = agent(s)

    print(a)