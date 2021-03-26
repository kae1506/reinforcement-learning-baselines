import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import gym
import math
import time
import matplotlib.pyplot as plt

import numpy as np

env = gym.make('CartPole-v1').unwrapped
n_games = 150

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
    def __init__(self, lr, input_shape, n_actions, eps_dec=10e-4, eps_end=0.001):
        self.lr = lr
        #self.gamma = gamma
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.eps = 1
        self.eps_dec = eps_dec
        self.eps_end = eps_end

        self.model = Network(lr, self.input_shape, self.n_actions)

    def choose_action(self, state):

        state = T.Tensor(state).float().detach()
        state = state.to(self.model.device)
        #print(state.shape)
        states = state.unsqueeze(0)
        #print(state.shape)

        # print(T.argmax(self.model.forward(state)))

        # if np.random.random() > self.eps:
        #     start = time.process_time()
        #     actions = self.model.forward(states)
        #     action = T.argmax(actions).item()
        #     print("time to predict is: ", time.process_time()-start)
        # # print(action)
        # #
        # else:
        #     action = env.action_space.sample()
        #
        # return action

        q_values = self.model(states)
        probs = q_values.abs()
        weighted_probs = F.softmax(probs)
        dist = T.distributions.Categorical(weighted_probs)
        action = dist.sample()
        return action.item()

    def learn(self, memory, batchSize):
        if len(memory) < batchSize:
            return


        self.model.optimizer.zero_grad()

        randomMemories = random.choices(memory, k=batchSize)  # randomly choose some memories

        # this bullshit just puts all the memories into their own seperate numpy arrays
        # # I encourage you to print out each line to see what's going on
        memories = np.stack(randomMemories)
        states, actions, rewards, states_, dones = memories.T
        states, actions, rewards, states_, dones = \
            np.stack(states), np.stack(actions), np.stack(rewards), np.stack(states_), np.stack(dones)

        # start_ = time.process_time()
        states = T.Tensor(states).float().to(self.model.device)
        actions = T.Tensor(actions).to(T.int64).to(self.model.device)
        rewards = T.Tensor(rewards).float().to(self.model.device)
        states_ = T.Tensor(states_).float().to(self.model.device)
        dones = T.Tensor(dones).to(T.bool).to(self.model.device)
        # print("time for converting into tensors: ", time.process_time()-start_)

        # print(states.shape)
        batch_indices = np.arange(batchSize, dtype=np.int64)
        q_now = self.model(states)[batch_indices, actions]
        # print(q_now.shape)
        q_next = T.max(self.model(states_), dim=1)[0]
        # print("q_next: ",  q_next.shape)
        q_next[dones] = 0.0

        q_target = reward + q_next

        loss = self.model.loss(q_target, q_now)
        print(loss.shape)

        loss.backward()
        self.model.optimizer.step()

        self.eps = self.eps-self.eps_dec if self.eps > self.eps_end else self.eps_end

if __name__ == '__main__':

    #n_games = 500
    agent = Agent(lr=0.0001, input_shape=(4,), n_actions=2)

    highscore = -math.inf
    memory = []
    scores = []
    Avg_scores = []

    for i in range(n_games):
        state = env.reset()
        done=False

        score = 0
        frame = 0
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            memory.append([state, action, reward, state_, done])
            agent.learn(memory, 64)
            env.render()
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

    plt.plot(Avg_scores)
    plt.show()
