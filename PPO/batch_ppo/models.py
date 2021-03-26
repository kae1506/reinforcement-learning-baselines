import math
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(torch.nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()

        self.data_shape = config['state_shape']
        self.action_shape = config['action_shape']

        self.std = config['std']

        self.fc1_dims = config['fc1_dims']
        self.fc2_dims = config['fc2_dims']

        self.net = nn.Sequential(
            nn.Linear(  *self.data_shape,    self.fc1_dims      ),  nn.ReLU(),
            nn.Linear(   self.fc1_dims,      self.fc2_dims      ),  nn.ReLU(),
            nn.Linear(   self.fc2_dims,     *self.action_shape  ),
        )

        self.log_std = nn.Parameter(torch.ones(1, *self.action_shape) * self.std)

    def forward(self, x):
        mu    = self.net(x)
        std   = self.log_std.exp().expand_as(mu)
        policy_dist  = torch.distributions.Normal(mu, std)
        return policy_dist

    def forward_deterministic(self, x):
        mu = self.net(x)
        return mu

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()

        self.data_shape = config['state_shape']

        self.fc1_dims = config['fc1_dims']
        self.fc2_dims = config['fc2_dims']

        self.net = nn.Sequential(
            nn.Linear(  *self.data_shape,   self.fc1_dims   ),  nn.ReLU(),
            nn.Linear(   self.fc1_dims,     self.fc2_dims   ),  nn.ReLU(),
            nn.Linear(   self.fc2_dims,     1               ),
        )

    def forward(self, x):
        value = self.net(x)
        return value
