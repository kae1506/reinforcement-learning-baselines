import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import math
import time
import numpy as np
import matplotlib.pyplot as plt

class ICM(nn.Module):
    '''My implementation of the Paper "Curiosity Driven Reinforcement Learning"'''
    def __init__(self, state_shape, num_actions, encoding_shape):
        super().__init__()

        self.fc1 = 512
        self.fc2 = 256

        self.encoder = nn.Sequential(
            nn.Linear(*state_shape,    self.fc1), nn.ReLU(),
            nn.Linear(self.fc1,        self.fc2), nn.ReLU(),
            nn.Linear(self.fc2,  encoding_shape)
        )

        enc_shape = encoding_shape + num_actions

        self.encoder_predictor = nn.Sequential(
            nn.Linear(enc_shape,      self.fc1), nn.ReLU(),
            nn.Linear(self.fc1,       self.fc2), nn.ReLU(),
            nn.Linear(self.fc2, encoding_shape)
        )

        apred_shape = encoding_shape*2
        self.action_predictor = nn.Sequential(
            nn.Linear(apred_shape,    self.fc1), nn.ReLU(),
            nn.Linear(self.fc1,       self.fc2), nn.ReLU(),
            nn.Linear(self.fc2,    num_actions)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.000001)

    def get_preds(self, state, next_state, actions):
        state = torch.tensor(state).float().to(self.device) if type(state) is not torch.Tensor else state
        next_state = torch.tensor(next_state).float().to(self.device) if type(next_state) is not torch.Tensor else next_state
        actions = torch.tensor(actions).to(self.device)

        state_encodings = self.encoder(state)
        next_state_encodings = self.encoder(next_state)

        #print(state_encodings.shape, action.shape)
        encoder_predictor_inputs = torch.cat([state_encodings, actions], dim=1)
        encoder_predictor_outputs = self.encoder_predictor(encoder_predictor_inputs)

        action_predictor_inputs = torch.cat([state_encodings, next_state_encodings], dim=1)
        action_predictor_outputs =  self.action_predictor(action_predictor_inputs)

        return next_state_encodings, encoder_predictor_outputs, action_predictor_outputs


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_actions = n_actions
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        action_one_hot = np.zeros(self.n_actions, dtype=np.int64)
        action_one_hot[action] = 1

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action_one_hot
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

        return states, actions, rewards, states_, terminal


class Network(nn.Module):
    def __init__(self, alpha, inputShape, numActions):
        super().__init__()
        self.inputShape = inputShape
        self.numActions = numActions
        self.fc1Dims = 1024
        self.fc2Dims = 512

        self.fc1 = nn.Linear(self.inputShape[0], self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        preds = self.fc3(x)

        return preds


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

        self.memory = ReplayBuffer(1000000, self.input_shape, self.n_actions)

        self.model = Network(lr, self.input_shape, self.n_actions)
        self.target = Network(lr, self.input_shape, self.n_actions)
        self.icm = ICM(self.input_shape, self.n_actions, 64)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.icm.parameters()), lr=3e-4
        )

    def choose_action(self, state):
        state = torch.Tensor(state).to(self.model.device)
        states = state.unsqueeze(0)

        if np.random.random() > self.eps:
            actions = self.model(states)
            action = torch.argmax(actions).item()
        else:
            action = env.action_space.sample()

        return action

    def replace_ntwrk(self):
        self.target.load_state_dict(self.model.state_dict())

    def get_icm_loss(self, states, states_, actions):
        batch_size = actions.shape[0]
        next_state_encodings, predicted_next_state_encodings, predicted_actions = self.icm.get_preds(states, states_, actions)

        forward_loss = (next_state_encodings - predicted_next_state_encodings).sum(dim=1)**2
        inverse_loss = (actions - predicted_actions).sum(dim=1)**2

        intrinsic_reward = torch.clamp(forward_loss, -1.0, 1.0).detach().view(batch_size)

        forward_loss = forward_loss.mean()
        inverse_loss = inverse_loss.mean()

        return forward_loss, inverse_loss, intrinsic_reward

    def learn(self, batchSize):
        if self.memory.mem_cntr < batchSize:
            return

        self.optimizer.zero_grad()

        if self.learn_cntr % self.replace == 0:
            self.replace_ntwrk()

        states, actions, rewards, states_, dones = self.memory.sample_buffer(batchSize)

        states = torch.Tensor(states).to(torch.float32).to(self.model.device)
        actions = torch.Tensor(actions).to(torch.int64).to(self.model.device)
        rewards = torch.Tensor(rewards).to(torch.float32).to(self.model.device)
        states_ = torch.Tensor(states_).to(torch.float32).to(self.model.device)
        dones = torch.Tensor(dones).to(torch.bool).to(self.model.device)

        forward_loss, inverse_loss, intrinsic_reward = self.get_icm_loss(states, states_, actions)

        rewards = intrinsic_reward
        actions_from_one_hot = torch.argmax(actions, dim=1)
        batch_indices = np.arange(batchSize, dtype=np.int64)
        qValue = self.model(states)

        qValue = qValue[batch_indices, actions_from_one_hot]

        qValues_ = self.target(states_)
        policyQValues_ = self.model(states_)
        actions_ = torch.max(policyQValues_, dim=1)[1]
        qValue_ = qValues_[batch_indices, actions_]
        qValue_[dones] = 0.0

        td = rewards + self.gamma * qValue_
        q_learning_loss = (td-qValue).pow(2).mean()

        loss = q_learning_loss + forward_loss + inverse_loss

        loss.backward()
        self.optimizer.step()

        self.eps -= self.eps_dec
        if self.eps < self.eps_min:
            self.eps = self.eps_min

        self.learn_cntr += 1


if __name__ == '__main__':
    BATCH_SIZE = 64
    n_games = 200
    env = gym.make('LunarLander-v2')
    agent = Agent(lr=0.00001, input_shape=(8,), n_actions=4)

    start_time = time.process_time()

    scores = []
    Avg_scores = []
    highscore = -math.inf
    i = 0
    while True:
        state = env.reset()
        done = False

        score = 0
        frame = 0
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)

            agent.memory.store_transition(state, action, reward, state_, done)
            agent.learn(BATCH_SIZE)

            score += reward
            frame += 1
            state = state_

        scores.append(score)
        if score >= 1000 or i > 999:
            print(f'score: {score}')
            print(f'solved in --> {i} episodes')
            break
        highscore = max(highscore, score)

        avg_score = np.mean(scores)#[-100:])
        Avg_scores.append(avg_score)

        print(("ep {}: high-score {:12.3f}, "
               "score {:12.3f}, avg {:12.3f}, avg 100 {:12.3f}").format(
            i, highscore, score, avg_score, np.mean(scores[-100:])))

        i += 1

    print(time.process_time() - start_time)
    plt.plot(Avg_scores)
    plt.show()
