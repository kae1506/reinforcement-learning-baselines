"""
This contains the network code for TD3
4-11-2020, Kae.
MIT License
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
import numpy


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, fc1_dims, fc2_dims, name, lr=0.001):
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.filename = r"models/" + str(
            name) + "_td3.weights"

        self.fc1 = nn.Linear(self.input_shape[0]+self.n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)

        self.device = torch.device("cuda:0")
        self.to(self.device)

    def forward(self, state, action):
        #print(state.shape, action.shape)
        x = self.fc1(torch.cat([state,action], dim=1))
        x = F.relu(x)
        x = self.fc2(x)

        x = F.relu(x)
        q = self.fc3(x)

        return q

    def save_checkpoint(self):
        print("_saving_")
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        print("_loading_")
        self.load_state_dict(torch.load(self.filename))


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, fc1_dims, fc2_dims, name, lr=0.0001):
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.filename = r"models/" + str(
            name) + "_td3.weights"

        self.fc1 = nn.Linear(*input_shape, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device("cuda:0")
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)

        return x

    def save_checkpoint(self):
        print("_saving_")
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        print("_loading_")
        self.load_state_dict(torch.load(self.filename))