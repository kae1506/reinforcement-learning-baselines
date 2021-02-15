import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import gym
import math
import time
import numpy as np
import torch.autograd as autograd
from npreplaymemory import ReplayBuffer
import matplotlib.pyplot as plt
from utils import plotLearning
import torch

np.random.seed(0)

class NoisyLayer(nn.Module):
    def __init__(self, in_f, out):
        super().__init__()
        self.in_f = in_f
        self.out  = out
        self.device = torch.device("cuda:0")

        self.weight_mu      = nn.Parameter(torch.Tensor(out, in_f).to(self.device))
        self.weight_sigma   = nn.Parameter(torch.Tensor(out, in_f).to(self.device))

        self.bias_mu        = nn.Parameter(torch.Tensor(out).to(self.device))
        self.bias_sigma     = nn.Parameter(torch.Tensor(out).to(self.device))

        self.init_params()

    def init_params(self):
        mu_range = math.sqrt(3/self.in_f)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.017)

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.017)

    def forward(self, x):
        weight_epsilon = torch.Tensor(np.random.normal(size=(self.out, self.in_f))).to(self.device)
        bias_epsilon = torch.Tensor(np.random.normal(size=self.out)).to(self.device)

        weights = self.weight_mu + (self.weight_sigma.mul(autograd.Variable(weight_epsilon).cuda()))
        bias = self.bias_mu + (self.bias_sigma.mul(autograd.Variable(bias_epsilon).cuda()))

        return F.linear(x, weights, bias)

class Network(nn.Module):
    def __init__(self, alpha, inputShape, numActions):
        super().__init__()
        self.inputShape = inputShape
        self.numActions = numActions
        self.fc1Dims = 1024
        self.fc2Dims = 512

        self.fc1 = nn.Linear(*self.inputShape, self.fc1Dims)
        self.fc2 = NoisyLayer(self.fc1Dims, self.fc2Dims)
        self.fc3 = NoisyLayer(self.fc2Dims, numActions)

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

        # self.eps = 1
        # self.eps_dec = eps_dec
        # self.eps_min = eps_min

        self.model = Network(lr, self.input_shape, self.n_actions)
        self.target = Network(lr, self.input_shape, self.n_actions)
        self.memory = ReplayBuffer(1_000_000, env_dict={
            "obs"     : {"shape": self.input_shape},
            "act"     : {"shape": self.n_actions  },
            "rew"     : {                         },
            "next_obs": {"shape": self.input_shape},
            "done"    : {                         }
        })


    def choose_action(self, state):
        # if np.random.random() > self.eps:
        state = T.Tensor(state).to(self.model.device)
        states = state.unsqueeze(0)
        actions = self.model(states)
        action = T.argmax(actions).item()
        # else:
        #     action = env.action_space.sample()

        return action

    def replace_ntwrk(self):
        self.target.load_state_dict(self.model.state_dict())

    def learn(self, batchSize):
        if self.memory.memCount < batchSize:
            return

        self.model.optimizer.zero_grad()

        if self.learn_cntr % self.replace == 0:
            self.replace_ntwrk()

        state, action, reward, state_, done = self.memory.sample(batchSize)

        states  = T.Tensor(state).to(T.float32 ).to(self.model.device)
        actions = T.Tensor(action).to(T.int64   ).to(self.model.device)
        rewards = T.Tensor(reward).to(T.float32 ).to(self.model.device)
        states_ = T.Tensor(state_).to(T.float32 ).to(self.model.device)
        dones   = T.Tensor(done).to(T.bool    ).to(self.model.device)

        batch_indices = np.arange(batchSize, dtype=np.int64)
        qValue = self.model(states)[batch_indices, actions]

        qValues_ = self.target(states_)
        qValue_ = T.max(qValues_, dim=1)[0]
        qValue_[dones] = 0.0

        td = rewards + self.gamma * qValue_
        loss = self.model.loss(td, qValue)
        loss.backward()
        self.model.optimizer.step()

        #   PER
        error = td - qValue

        #
        # self.eps -= self.eps_dec
        # if self.eps < self.eps_min:
        #     self.eps = self.eps_min

        self.learn_cntr += 1


if __name__ == '__main__':
    BATCH_SIZE = 64
    n_games = 150
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(lr=0.001, input_shape=(4,), n_actions=2)
    memory = []
    scores = []
    Avg_scores = []
    highscore = -math.inf

    start_time =  time.process_time()
    for i in range(n_games):
        state = env.reset()
        done=False

        score = 0
        frame = 0
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            agent.memory.add(obs=state, act=action, rew=reward, next_obs=state_, done=done)
            agent.learn(BATCH_SIZE)

            score += reward
            frame += 1
            state = state_

        scores.append(score)
        highscore = max(highscore, score)

        avg_score = np.mean(scores[-100:])
        Avg_scores.append(avg_score)

        print(( "ep {}: high-score {:12.3f}, "
                "score {:12.3f}, average_score {:12.3f}").format(
            i, highscore, score, avg_score))

    print(time.process_time() - start_time)
    plotLearning([i for i in range(n_games)], Avg_scores, scores, "del.png")
    plt.show()
