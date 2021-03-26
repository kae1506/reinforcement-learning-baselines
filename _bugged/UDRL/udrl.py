import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import gym
import matplotlib.pyplot as plt


class BehaviouralFunction(nn.Module):
    def __init__(self, input_shape, action_shape, command_shape=2):
        super().__init__()

        self.input_shape = input_shape
        self.action_shape = action_shape
        self.command_shape = command_shape
        self.hidden_dim = 256
        self.hidden_dim2 = 512

        self.input_tail = nn.Linear(input_shape, self.hidden_dim)
        self.command_tail = nn.Linear(self.command_shape, self.hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim2),   nn.ReLU(),
            nn.Linear(self.hidden_dim2, self.hidden_dim2),    nn.ReLU(),
            nn.Linear(self.hidden_dim2, action_shape),
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.to(self.device)

    def forward(self, state, horizon, reward):
        state = state.to(self.device)
        horizon = horizon.to(self.device)
        reward = reward.to(self.device)

        inputs = self.input_tail(state)
        catted = torch.cat([horizon, reward], dim=1)
        command = self.command_tail(catted)


        total_inputs = torch.tanh(torch.add(inputs, command))
        outputs = self.fc(total_inputs)
        outputs = torch.softmax(outputs, dim=-1)

        return outputs

class ReplayBuffer():
    def __init__(self, mem_size, state_shape, n_actions):
        self.state_shape = state_shape
        self.mem_size = mem_size
        self.mem_count = 0

        self.states  = np.zeros((mem_size, state_shape), dtype=np.float32)
        self.actions = np.zeros((mem_size,   n_actions), dtype=np.int64  )
        self.rewards = np.zeros(mem_size,                dtype=np.int32  )

    def sample_buffer(self, batch_size, device):
        limit = min(self.mem_count, self.mem_size)
        idx = np.random.choice(limit, batch_size, replace=False)

        states = torch.tensor(self.states[idx, :]).float().to(device)
        actions = torch.tensor(self.actions[idx, :]).float().to(device)
        rewards = torch.tensor(self.rewards[idx]).float().to(device)

        return states, actions, rewards

    def store_transition(self, state, action, reward):
        idx = self.mem_count % self.mem_size

        self.states[idx, :] = state
        self.actions[idx, :] = action
        self.rewards[idx] = reward

        self.mem_count += 1

class Agent():
    def __init__(self,
                 state_shape,
                 num_actions,
                 warm_up=100,
                 eps=1.0,
                 eps_dec=0.001,
                 eps_min=0.01,
                 batch_size=64):

        self.state_shape = state_shape
        self.action_shape = [i for i in range(num_actions)]

        self.warm_up = warm_up
        self.warm_up_step = 0
        self.batch_size = batch_size

        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        self.model = BehaviouralFunction(state_shape, num_actions)
        self.memory = ReplayBuffer(1000000, state_shape, num_actions)

    def choose_action(self, x):
        if self.warm_up_step < self.warm_up:
            self.warm_up_step += 1
            action = np.random.choice(self.action_shape)
            expected_vals = np.array([0.0 for _ in range(len(self.action_shape))])
            expected_vals[action] = 1.0
            return action, expected_vals

        if np.random.random() > self.eps:
            state = torch.tensor([x]).float().to(self.model.device)
            reward_mean = np.mean(self.memory.rewards)
            expected_reward = torch.distributions.Normal(reward_mean, 0.1).sample().reshape(1, 1)
            horizon = torch.tensor(1).expand_as(expected_reward)
            expected_vals = self.model(state, horizon, expected_reward)
            action = torch.distributions.Categorical(expected_vals).sample()
            return action.item(), expected_vals.detach().cpu().numpy()
        else:
            action = np.random.choice(self.action_shape)
            expected_vals = np.array([0.0 for _ in range(len(self.action_shape))])
            expected_vals[action] = 1.0
            return action, expected_vals


    def learn(self):
        if self.memory.mem_count < self.warm_up:
            return

        self.model.optimizer.zero_grad()

        states, actions, rewards = self.memory.sample_buffer(self.batch_size, self.model.device)
        horizon = torch.tensor([[self.batch_size]]).to(self.model.device)
        reward_mean = rewards.mean(dim=0)
        dists = torch.distributions.Normal(reward_mean, 0.1)
        reward_samples = torch.tensor([dists.sample() for _ in range(self.batch_size)])
        expected_reward = torch.sum(reward_samples).reshape(1,1).to(self.model.device)
        pred_actions = self.model(states, horizon, expected_reward).float()

        loss = F.mse_loss(pred_actions, actions)
        loss.backward()

        self.model.optimizer.step()

        self.eps -= self.eps_dec if self.eps > self.eps_min else self.eps

scores, avgs = [], []
env = gym.make('CartPole-v1').unwrapped
agent = Agent(4, 2)
high_score = 0

for i in range(500):
    done = False
    score = 0
    state = env.reset()
    while not done:
        action, expected_vals = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        score += reward
        agent.memory.store_transition(state, expected_vals, reward)

        agent.learn()

    scores.append(score)
    avgs.append(np.mean(scores))
    high_score = max(score, high_score)

    print(f'episode {i}, score: {score}, avg: {np.mean(scores)}, high score: {high_score}')


plt.plot(avgs)
plt.show()