import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, fc1_dims, fc2_dims, name, lr=0.001):
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.filename = r"C:/Users/Lenovo/Documents/Deep Q Learning/PolicyGradientMethods/DDPG/models/" + str(name) + "_ddpg.weights"
        
        self.fc1 = nn.Linear(*input_shape, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.batchNorm1 = nn.LayerNorm(fc1_dims)   # Layer Norm because it does singularly each
        self.batchNorm2 = nn.LayerNorm(fc2_dims)

        self.action_input = nn.Linear(n_actions, fc2_dims)

        self.q = nn.Linear(fc2_dims, 1)

        # Setting the weights to their respective ranges
        fc1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-fc1, fc1)
        self.fc1.bias.data.uniform_(-fc1, fc1)

        fc2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-fc2, fc2)
        self.fc2.bias.data.uniform_(-fc2, fc2)

        ac1 = 1./np.sqrt(self.action_input.weight.data.size()[0])
        self.action_input.weight.data.uniform_(-ac1, ac1)
        self.action_input.bias.data.uniform_(-ac1, ac1)

        #Output has to be different
        q = 0.003                         # Told by the paper.
        self.q.weight.data.uniform_(-q, q)
        self.q.bias.data.uniform_(-q, q)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.batchNorm2(x)

        ac = self.action_input(action)
        x = F.relu(torch.add(x,ac))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        print("_saving_")
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        print("_loading_")
        self.load_state_dict(torch.load(self.filename))
 
class ActorNetwork(nn.Module):
    def __init__(self, input_shape, n_actions,fc1_dims, fc2_dims, name,lr=0.0001):
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.filename = r"C:/Users/Lenovo/Documents/Deep Q Learning/PolicyGradientMethods/DDPG/models/" + str(name) + "_ddpg.weights"
        
        self.fc1 = nn.Linear(*input_shape, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.batchNorm1 = nn.LayerNorm(fc1_dims)   # Layer Norm because it does singularly each
        self.batchNorm2 = nn.LayerNorm(fc2_dims)

        self.mu = nn.Linear(fc2_dims, n_actions)

        # Setting the weights to their respective ranges
        fc1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-fc1, fc1)
        self.fc1.bias.data.uniform_(-fc1, fc1)

        fc2 = 1./np.sqrt(self.fc2.weight.data.size()[0]) 
        self.fc2.weight.data.uniform_(-fc2, fc2)
        self.fc2.bias.data.uniform_(-fc2, fc2)

        #Output has to be different
        q = 0.003                         # Told by the paper.
        self.mu.weight.data.uniform_(-q, q)
        self.mu.bias.data.uniform_(-q, q)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.mu(x)
        x = F.tanh(x)

        return x

    def save_checkpoint(self):
        print("_saving_")
        torch.save(self.state_dict(), self.filename)

    def load_checkpoint(self):
        print("_loading_")
        self.load_state_dict(torch.load(self.filename))