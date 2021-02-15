import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import gym
import math 
import random


class PolicyNetwork(nn.Module):
    def __init__(self, inputShape, numActions, lr=0.0005, device="cuda:0"):
        super(PolicyNetwork, self).__init__()
        self.inputShape = inputShape
        self.numActions = numActions

        self.fc = nn.Linear(*self.inputShape, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, self.numActions)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0")
        self.to(self.device)

    def forward(self, x):
        # print(x.shape,self.inputShape)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class PolicyAgent(object):
    def __init__(self, inputShape, numActions, lr, gamma):
        self.inputShape = inputShape
        self.numActions = numActions
        self.lr = lr
        self.gamma = gamma

        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(inputShape, numActions, device="cuda:0")

    def choose_action(self, x):
        x = torch.Tensor([x]).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(x))
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.action_memory.append(log_prob)

        return action.item()

    def learn(self, player):
        self.policy.optimizer.zero_grad()
        # Steps for REINFORCE: 
        # First we have to initialize a G with the cumsum of all the rewards
        # Use the zip of the logprobs and G to update loss 
        # step optimizer 

        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for i in range(len(self.reward_memory)):
            g_sum = 0
            discount = 1
            for k in range(i, len(self.reward_memory)):
                g_sum += self.reward_memory[k]*discount
                discount *= self.gamma

            G[i] = g_sum

        G = torch.tensor(G).to(torch.float).to(self.policy.device)

        loss = 0
        for g, log_prob in zip(G, self.action_memory):
            loss += -g*log_prob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory= []
        self.reward_memory= []


    def save_rewards(self, reward):
        self.reward_memory.append(reward)


a=[i for i in range(n_games)]
plt.plot(a, avg_scores)
plt.xlabel("Game", color="C0")
plt.ylabel("Avg_scores", color="C0")
plt.savefig("AlphaGoPhuqUrself.png")
plt.show()

