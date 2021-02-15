import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import gym
import math
import time
import numpy as np
import matplotlib.pyplot as plt

torch = T

class ReplayBuffer():
    def __init__(self, max_size, input_shape, hidden_size=64):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.hidden_memory = np.zeros((self.mem_size, hidden_size), 
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done, hidden):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.hidden_memory[index] = hidden
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        hidden = self.hidden_memory[batch]

        return states, actions, rewards, states_, terminal, hidden


class Network(nn.Module):
    def __init__(self, alpha, inputShape, numActions):
        super().__init__()
        self.inputShape = inputShape
        self.numActions = numActions
        self.fc1Dims = 1024
        self.fc2Dims = 512
        self.hidden_shape = 64

        self.fc1 = nn.Linear(self.inputShape[0] + self.hidden_shape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, numActions)
        self.hidden = nn.Linear(self.fc2Dims, self.hidden_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, x, hidden):
        x = x.to(self.device)
        hidden = hidden.to(self.device)
        x = F.relu(self.fc1(torch.cat((x, hidden), dim=1)))
        x = F.relu(self.fc2(x))
        preds = self.fc3(x)
        hidden = self.hidden(x)

        return preds, hidden 

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

        self.memory = ReplayBuffer(100000, self.input_shape)

        self.model = Network(lr, self.input_shape, self.n_actions)
        self.target = Network(lr, self.input_shape, self.n_actions)


    def choose_action(self, state, hidden): 
        state = T.Tensor(state).to(self.model.device)
        states = state.unsqueeze(0)
        
        if np.random.random() > self.eps:
            actions, hidden_preds = self.model(states, hidden)
            action = T.argmax(actions).item()
        else:
            action = env.action_space.sample()
            _, hidden_preds = self.model(states, hidden)

        return action, hidden_preds

    def hidden_states(self):
        hidden = T.zeros((1, self.model.hidden_shape)).to(self.model.device)
        return hidden

    def replace_ntwrk(self):
        self.target.load_state_dict(self.model.state_dict())

    def learn(self, batchSize):
        if self.memory.mem_cntr < batchSize:
            return

        self.model.optimizer.zero_grad()

        if self.learn_cntr % self.replace == 0:
            self.replace_ntwrk()

        states, actions, rewards, states_, dones, hidden = self.memory.sample_buffer(batchSize)

        states  = T.Tensor(states ).to(T.float32 ).to(self.model.device)
        actions = T.Tensor(actions).to(T.int64   ).to(self.model.device)
        rewards = T.Tensor(rewards).to(T.float32 ).to(self.model.device)
        states_ = T.Tensor(states_).to(T.float32 ).to(self.model.device)
        dones   = T.Tensor(dones  ).to(T.bool    ).to(self.model.device)
        hidden  = T.Tensor(hidden ).to(T.float32 ).to(self.model.device)

        batch_indices = np.arange(batchSize, dtype=np.int64)
        qValue, _ = self.model(states, hidden)

        qValue = qValue[batch_indices, actions]

        qValues_, _ = self.target(states_, hidden)
        qValue_ = T.max(qValues_, dim=1)[0]
        qValue_[dones] = 0.0

        td = rewards + self.gamma * qValue_
        loss = self.model.loss(td, qValue)
        loss.backward()
        self.model.optimizer.step()

        #   PER
        error = td - qValue


        self.eps -= self.eps_dec
        if self.eps < self.eps_min:
            self.eps = self.eps_min

        self.learn_cntr += 1


if __name__ == '__main__':
    BATCH_SIZE = 64
    n_games = 150
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(lr=0.001, input_shape=(4,), n_actions=2)


    start_time = time.process_time()

    scores = []
    Avg_scores = []
    highscore = -math.inf
    for i in range(n_games):
        state = env.reset()
        done=False

        score = 0
        frame = 0
        hidden = agent.hidden_states()
        while not done:
            action, hidden_ = agent.choose_action(state, hidden)
            state_, reward, done, info = env.step(action)
            hidden_n = hidden.cpu().detach().numpy()

            agent.memory.store_transition(state, action, reward, state_, done, hidden_n)
            agent.learn(BATCH_SIZE)
            
            hidden = hidden_

            score += reward
            frame += 1
            state = state_

        scores.append(score)
        highscore = max(highscore, score)

        avg_score = np.mean(scores[-100:])
        Avg_scores.append(avg_score)

        print(( "ep {}: high-score {:12.3f}, "
                "score {:12.3f}, avg {:12.3f}").format(
            i, highscore, score, avg_score))

    print(time.process_time()-start_time)
    plt.plot(Avg_scores)
    plt.show()

