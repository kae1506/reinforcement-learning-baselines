"""
This is the PER attempt with making recent memories much more important.
"""

import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import gym
import math
from npreplaymemory import ReplayBuffer
import matplotlib.pyplot as plt
from utils import plotLearning
import numpy as np

class Network(nn.Module):
    def __init__(self, alpha, inputShape, numActions):
        super().__init__()
        self.inputShape = inputShape
        self.numActions = numActions
        self.fc1Dims = 1024
        self.fc2Dims = 512

        self.fc1 = nn.Linear(*self.inputShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Agent():
    def __init__(self, lr, input_shape, n_actions, eps_dec=0.001, eps_min=0.001):
        self.lr = lr
        self.gamma = 0.99
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.surprise = 0.5

        self.learn_cntr = 0
        self.replace = 100

        self.eps = 1
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        self.memory = ReplayBuffer(500_000, self.input_shape)
        self.model = Network(lr, self.input_shape, self.n_actions)
        self.target = Network(lr, self.input_shape, self.n_actions)


    def choose_action(self, state):
        if np.random.random() > self.eps:
            state = T.Tensor(state).to(self.model.device)
            states = state.unsqueeze(0)
            actions = self.model(states)
            action = T.argmax(actions).item()
        else:
            action = env.action_space.sample()

        return action

    def replace_ntwrk(self):
        self.target.load_state_dict(self.model.state_dict())

    def learn(self, batchSize, priority_scale=1.1):
        if self.memory.memCount < batchSize:
            return

        self.model.optimizer.zero_grad()

        if self.learn_cntr % self.replace == 0:
            self.replace_ntwrk()

        weights = self.memory.stepMemory
        #print(len(weights))
        weights = np.array([i**priority_scale for i in weights])
        states, actions, rewards, states_, dones = self.memory.sample(batchSize, weights=weights)

        states  = T.Tensor(states).to(T.float32 ).to(self.model.device)
        actions = T.Tensor(actions).to(T.int64   ).to(self.model.device)
        rewards = T.Tensor(rewards).to(T.float32 ).to(self.model.device)
        states_ = T.Tensor(states_).to(T.float32 ).to(self.model.device)
        dones   = T.Tensor(dones).to(T.bool    ).to(self.model.device)

        batch_indices = np.arange(batchSize, dtype=np.int64)
        qValue = self.model(states)[batch_indices, actions]

        qValues_ = self.target(states_)
        qValue_ = T.max(qValues_, dim=1)[0]
        qValue_[dones] = 0.0

        td = rewards + self.gamma * qValue_
        loss = T.square(td-qValue)
        loss = loss.mean()
        loss.backward()
        self.model.optimizer.step()


        self.eps -= self.eps_dec
        if self.eps < self.eps_min:
            self.eps = self.eps_min

        self.learn_cntr += 1

if __name__ == '__main__':
    BATCH_SIZE = 64
    n_games = 250
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(lr=0.001, input_shape=(4,), n_actions=2)
    Avg_scores = []

    scores = []
    highscore = -math.inf
    for i in range(n_games):
        state = env.reset()
        done=False

        score = 0
        frame = 0
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            agent.memory.storeMemory(state, action, reward, state_, done)
            agent.learn(BATCH_SIZE)

            score += reward
            frame += 1
            state = state_

        scores.append(score)
        highscore = max(highscore, score)

        print(( "ep {}: high-score {:12.3f}, "
                "score {:12.3f}, last-episode-time {:4d}").format(
            i, highscore, score, frame))

        avg_score = np.mean(scores[-100:])
        Avg_scores.append(avg_score)

    plotLearning([i for i in range(n_games)], Avg_scores, scores, "per_time_step.png")
    plt.show()
