# This file is going to have all the mcts-related stuff
# Including the node class, and all the mcts functions
# OOP is only going to be used only for placeholders
# in general, this core algorithm is only for atari

import math
import random
import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Node():
    def __init__(self, prior):
        self.prior = prior
        self.value_sum = 0    # backpropogated value sum
        self.visits = 0
        self.children = {}
        self.parent = None
        self.reward = 0   # reward gotten, by the dynamics function
        self.hidden_state = None

    def expanded(self):
        return len(self.children.items()) > 0

    def value(self):
        if self.visits <= 0:
            return math.inf
        return self.value_sum/self.visits

def mcts(start_node, network, action_space): 
    # this will be called by the play_game function.
    # while expanded, traverse
    # expand
    # backpropogate with the latest reward

    for _ in range(150):
        current_node = start_node
        action_history = []
        search_path = [start_node]

        while current_node.expanded():
            action, new_node = select_node(current_node)
            action_history.append(action)
            print(action)
            search_path.append(new_node)


        # recurrent inference.

        network_output = network.recurrent_inference(current_node.parent.hidden_state, torch.tensor(action_history[-1]).unsqueeze(0))

        #expand
        expand_node(current_node, action_space, network_output)
        #backprop
        backpropogate(search_path, network_output.value)

def expand_node(node, action_space, network_output):
    # here we can set the reward that we got as well as the hidden state
    # then we can take the policy that we got from the second network output 
    # and add in all the children

    # also if its a turn based game then you need to fix the to_play and actions list
    node.reward = network_output.reward
    node.hidden_state = network_output.hidden_state

    if type(network_output.policy_logits.cpu().detach().numpy().tolist()) == list:
        network_output.policy_logits = network_output.policy_logits.squeeze(0)
    policies = {a: network_output.policy_logits[a].item() for a in range(action_space)}
    policy_sum = sum(policies.values())
    for action, p in policies.items():
        new_node = Node(p/policy_sum)
        new_node.parent = node
        node.children[action] = new_node

def ucb(node, child_node, base = 19652, init = 1.25):
    # Vnorm + (P * sqrt(N)/n+1 * log(N+base+1/base) + init)

    u = child_node.value() + ((
        (child_node.prior * math.sqrt(node.visits)/(child_node.visits + 1)) * 
        (math.log(
            (node.visits+base+1)/base)
        )
        +init)        
    )

    if type(u) == torch.Tensor:

        return u.detach().cpu()
    return u

def select_node(node):
    # basically the max of the ucb scores
    # children is going to be a dict with the key being the index 
    # and the value being the actual child node

    _, action, child = max(
        (ucb(node,child_node), 
        action, 
        child_node) for action, child_node in node.children.items()
    )

    return action, child

def backpropogate(search_path, value, discount=0.98):
    for node in search_path:
        node.value_sum += value
        node.visits += 1

        value = node.reward + discount * value

class NetworkOutput:
    value = None;
    reward = None;
    hidden_state = None;
    policy_logits = None;

    def __init__(self, value, reward, hidden_state, policy_logits):
        self.value = value
        self.reward = reward 
        self.hidden_state = hidden_state
        self.policy_logits = policy_logits

class HNetwork(nn.Module):
    def __init__(self, state_shape, hidden_shape, device):
        super().__init__()
        self.fc1 = nn.Linear(state_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, hidden_shape)

        self.optim = optim.Adam(self.parameters(), lr=3e-5)
        self.device = device
        self.to(device)

    def forward(self, inp):
        tensor = torch.tensor(inp).to(self.device)
        output = torch.relu(self.fc1(tensor))
        output = torch.relu(self.fc2(output))
        return torch.relu(self.fc3(output))

class GNetwork(nn.Module):
    def __init__(self, hidden_shape, action_shape, device):
        super().__init__()
        self.fc_i1 = nn.Linear(hidden_shape, 256)
        self.fc_i2 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(512, 128)
        self.r = nn.Linear(128, 1)
        self.s = nn.Linear(128, hidden_shape)

        self.optim = optim.Adam(self.parameters(), lr=3e-5)

        self.device = device
        self.to(device)

    def forward(self, state, action):
        state = torch.tensor(state).to(self.device)
        action = torch.tensor(action).float().to(self.device) if type(action) != torch.Tensor else action.float().to(self.device)

        hidden_1 = torch.relu(self.fc_i1(state))
        hidden_2 = torch.relu(self.fc_i2(action))
        
        output = torch.relu(self.fc2(torch.cat([hidden_1, hidden_2], dim=1)))
        return torch.relu(self.r(output)), torch.relu(self.s(output))

class FNetwork(nn.Module):
    def __init__(self, hidden_shape, policy_shape, device):
        super().__init__()
        self.fc1 = nn.Linear(hidden_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.a = nn.Linear(128, policy_shape)
        self.r = nn.Linear(128, 1)

        self.optim = optim.Adam(self.parameters(), lr=3e-5)

        self.device = device
        self.to(device)

    def forward(self, inp):
        tensor = torch.tensor(inp).to(self.device)
        output = torch.relu(self.fc1(tensor))
        output = torch.relu(self.fc2(output))
        return self.a(output), self.r(output)

class Agent():
    def __init__(self, hidden_shape, game_length, action_space, state_shape, num_games):
        self.hidden_shape = hidden_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.h_network = HNetwork(state_shape, self.hidden_shape, self.device)
        self.g_network = GNetwork(self.hidden_shape, action_space, self.device)
        self.f_network = FNetwork(self.hidden_shape, action_space, self.device)
        
        self.rollout_length = 5
        self.game_length = game_length
        self.games = NUM_GAMES
        self.batch_size = 4      
        self.td_steps = 5
        self.action_shape = action_space

        self.rewards = np.random.random((self.games, game_length, 1))
        self.root_values = np.random.random((self.games, game_length, 1))
        self.child_ucbs = np.random.random((self.games, game_length, action_space))
        self.actions = np.random.random((self.games, game_length, 1))
        self.states = np.random.random((self.games, game_length, 4))
        self.mem_cntr = 0

    def flush_memory(self):
        self.rewards = np.zeros((self.games, self.game_length, 1))
        self.root_values = np.zeros((self.games, self.game_length, 1))
        self.child_ucbs = np.zeros((self.games, self.game_length, self.action_shape))
        self.actions = np.zeros((self.games, self.game_length, 1))
        self.states = np.zeros((self.games, self.game_length, 4))



        self.mem_cntr = 0

    def update_memory(self, state, action, reward, r_v, child_ucbs, game_idx):
        self.states[game_idx, self.mem_cntr] = state
        self.actions[game_idx, self.mem_cntr] = action
        self.rewards[game_idx, self.mem_cntr] = reward
        self.root_values[game_idx, self.mem_cntr] = r_v 
        self.child_ucbs[game_idx, self.mem_cntr] = child_ucbs
        self.mem_cntr += 1
        if self.mem_cntr == self.game_length-1:
            self.mem_cntr = 0


    def initial_inference(self, inp):
        inp = torch.tensor(inp).to(self.device)

        hidden_state = self.h_network(inp)
        policy_logits, value = self.f_network(hidden_state)

        return NetworkOutput(value, None, hidden_state, policy_logits)

    def recurrent_inference(self, hidden_inp, action):
        hidden_inp = torch.tensor(hidden_inp).unsqueeze(0).to(self.device)
        action = torch.tensor(action).unsqueeze(0).to(self.device)

        pred_reward, hidden_state = self.g_network(hidden_inp, action)
        policy_logits, value = self.f_network(hidden_state)

        return NetworkOutput(value, pred_reward, hidden_state.squeeze(0), policy_logits)

    def sample_batch(self, discount=0.98):
        batch = np.random.choice(self.games, self.batch_size, replace=False)

        # making value target
        rewards = torch.tensor(self.rewards[batch]).to(self.device)[0]
        root_values = torch.tensor(self.root_values[batch]).to(self.device)[0]
        child_ucbs = torch.tensor(self.child_ucbs[batch]).to(self.device)[0]
        actions = torch.tensor(self.actions[batch]).to(self.device)[0]
        states = torch.tensor(self.states[batch]).to(self.device)[0]


        states_batch = torch.zeros((self.batch_size, self.rollout_length, 4))
        actions_batch = torch.zeros((self.batch_size, self.rollout_length))
        values_batch = torch.zeros((self.batch_size, self.rollout_length, 1))
        rewards_batch = torch.zeros((self.batch_size, self.rollout_length, 1))
        policy_batch = torch.zeros((self.batch_size, self.rollout_length, self.action_shape))

        for b in range(self.batch_size):
            start_idx = random.randint(0, self.game_length - self.rollout_length - self.td_steps - 1)
            for idx in range(start_idx, start_idx + self.rollout_length):
                unroll_till = idx + self.td_steps
                
                # its Uk*Y  + Uk+1*Y**K + ... + Un * Y**N-1 + Zn * Y ** n
                # where u is reward, k is the index, Z is the root value
                # Y is gamma the discount and N is the td steps
                # lowercase means index
                value = root_values[unroll_till] * discount**self.td_steps
                
                for k in range(idx, unroll_till):
                    value += rewards[k] * discount ** k
                
                # make policy logits 
                policy_logit_targets = torch.zeros((len(child_ucbs[idx])))
                for k in range(len(child_ucbs[idx])):
                    policy_logit_targets[k] = child_ucbs[idx, k]/sum(child_ucbs[idx])

                values_batch[b, idx] = torch.tensor(value)
                rewards_batch[b, idx] = torch.tensor(rewards[idx])
                policy_batch[b, idx] = policy_logit_targets

            states_batch[b] = states[start_idx:start_idx + self.rollout_length]
            actions_batch[b] = actions[b, start_idx:start_idx + self.rollout_length]

        return states_batch, actions_batch, values_batch, rewards_batch, policy_batch

    def train(self):
        batch = self.sample_batch()
        states, actions, values_batch, rewards_batch, policy_batch = batch

        for idx in range(self.batch_size):
            
            # first update the initial inference
            self.h_network.optim.zero_grad()
            self.g_network.optim.zero_grad()
            self.f_network.optim.zero_grad()
            print(states.shape)
            network_output =  self.initial_inference(states[idx])
            pred_value, hidden_state, pred_policy_logits = network_output.value, network_output.hidden_state, network_output.policy_logits
            target_value, target_policy_logits = values_batch[idx], policy_batch[idx]
            
            print(target_policy_logits.shape, pred_policy_logits.shape)

            (    
                nn.functional.mse_loss(target_value, pred_value)+
                nn.functional.cross_entropy(target_policy_logits, pred_policy_logits)
            ).backward()

            self.h_network.optim.step()
            self.g_network.optim.step()
            self.f_network.optim.step()

            for action in actions:
                self.g_network.optim.zero_grad()
                self.f_network.optim.zero_grad()

                action = action.unsqueeze(1)
                print(hidden_state.shape, action.shape)
                pred_reward, hidden_state = self.g_network(hidden_state, action)
                pred_policy_logits, pred_value = self.f_network(hidden_state)
                
                target_value, target_reward, target_policy_logits = values_batch[idx], rewards_batch[idx], policy_batch[idx]
                
                (
                    nn.functional.mse_loss(target_value, pred_value)+
                    nn.functional.mse_loss(target_reward, pred_reward)+
                    nn.functional.cross_entropy(target_policy_logits, pred_policy_logits)
                ).backward()

                self.g_network.optim.step()
                self.f_network.optim.step()


import gym
env = gym.make('CartPole-v1')

GAME_LENGTH = 10
HIDDEN_SHAPE = 32
ACTION_SHAPE = 2
NUM_GAMES = 1
STATE_SHAPE = 4

agent = Agent(HIDDEN_SHAPE, GAME_LENGTH, ACTION_SHAPE, STATE_SHAPE, NUM_GAMES)
scores = []


for i in range(100):
    # self-play
    for game in range(NUM_GAMES):

        done = False
        score = 0
        state = env.reset()

        for idx in tqdm(range(GAME_LENGTH)):
            if not done:
                r = Node(0)
                h = agent.initial_inference(state)
                expand_node(r, ACTION_SHAPE, NetworkOutput(0, 0, h.hidden_state, h.policy_logits))

                mcts(r, agent, ACTION_SHAPE)
                action, _ = select_node(r)
                state_, reward, done, _ = env.step(action)

                score += reward
            
                agent.update_memory(
                    state, action, reward, 
                    r.value().detach().cpu(), [ucb(r, i) for i in r.children.values()],
                    game
                )
                state = state_
            else:
                agent.update_memory(
                    state, action, -1, 
                    r.value().detach().cpu(), [ucb(r, i) for i in r.children.values()],
                    game
                )

        scores.append(score)
        print(f'score: {score}, avg_score: {np.average(scores)}')

    # train
    agent.train()
    agent.flush_memory()