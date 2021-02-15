import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import gym
import pybulletgym
import numpy as np
from continuous_cartpole import ContinuousCartPoleEnv
from matplotlib.animation import FuncAnimation
import warnings; warnings.filterwarnings('ignore')
from argparse import ArgumentParser

#parser = ArgumentParser(description='Arguments for the ting')
#parser.add_argument('env', type=str, default='ContinuousCartPoleEnv')
#parser.add_argument('input_shape', type=str, default='4')
#parser.add_argument('n_actions', type=str, default='1')
#parser.add_argument('standard_deviation', type=int, default=0.0)
#parser.add_argument('action_limit', type=int, default=1)
#parser.add_argument('play_steps', type=str, default=20)

ENTROPY = []
RETURNS = []

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, std=9.0):
        super().__init__()

        self.n_actions = n_actions

        self.actor = nn.Sequential(
            nn.Linear(input_shape, 1024), nn.ReLU(),
            nn.Linear(1024, 512),         nn.ReLU(),
            nn.Linear(512, n_actions) 
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.to(self.device)
        
        self.lg_std = nn.Parameter(torch.ones(1, n_actions)*std).to(self.device)
        
    def forward(self, state):
        state = torch.tensor([state]).float().to(self.device) if type(state) is not torch.Tensor else state
        mu = self.actor(state)
        lg_std = self.lg_std.exp().expand_as(mu)
        
        dist = torch.distributions.Normal(mu, lg_std)
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, input_shape):
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(input_shape, 1024), nn.ReLU(),
            nn.Linear(1024,512),         nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = torch.tensor([state]).float().to(self.device) if type(state) is not torch.Tensor else state

        return self.critic(state)


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

        self.value = []
        self.probs = []
        self.dones = []

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

        self.value = []
        self.probs = []
        self.dones = []

    def store_memory(self, s, a, r, v, p, d):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

        self.value.append(v)
        self.probs.append(p)
        self.dones.append(d)

    def get_batches(self, batch_size):
        t_l = len(self.dones)
        indices = np.arange(t_l, dtype=np.float32)
        np.random.shuffle(indices)
        start_indicies = np.arange(0, t_l, batch_size)
        batches = [indices[i:i+batch_size] for i in start_indicies]

        return batches

    def get_nps(self):
        return np.array(self.states), \
                np.array(self.actions), \
                np.array(self.rewards), \
                np.array(self.value), \
                np.array(self.probs), \
                np.array(self.dones) 

class PPOAgent:
    def __init__(self, env):
        # SETTINGS
        self.input_shape = 96*96*3 #env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]

        self.env = env

        self.epochs = 4
        self.timesteps = 512
        self.mini_batch_size =128
        self.gamma = 0.99
        self.tau = 0.95
        self.play_steps = self.timesteps
        self.std = 0.0
        self.action_limit = 1

        self.adv_norm = False
        self.gae = False
        
        self.high_score = -np.inf
        self.avg_scores = []
        self.scores = []

        self.actor = PolicyNetwork(self.input_shape, self.n_actions, std=self.std)
        self.critic = CriticNetwork(self.input_shape)
        
        self.device = self.actor.device
        
        self.memory = PPOMemory()

    def choose_action(self, state):
        global ENTROPY
        dist = self.actor.forward(state)
        value = self.critic.forward(state)
        action = dist.sample()
        prob = dist.log_prob(action)
        #print(dist.entropy().cpu().detach().numpy())
        #ENTROPY.append(dist.entropy().cpu().detach().numpy())

        action = action.clamp(self.action_limit, -self.action_limit)

        return action.detach().cpu().numpy()[0], value.item(), prob.detach().cpu().numpy()[0]

    def compute_gae(self, rewards, masks, values, next_val=None):
        returns = []
        gae = 0
        value_ = np.append(values, next_val)

        for i in reversed(range(len(rewards))):
            td_res = rewards[i] + self.gamma * value_[i+1] * masks[i] - value_[i]
            gae = td_res + self.gamma * self.tau * masks[i] * gae
            returns.insert(0, gae+value_[i])

        return torch.tensor(returns).to(self.device)
        
    def compute_adv(self, rewards, masks, values, next_val=None):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        value_ = np.append(values, next_val)

        for i in reversed(range(len(rewards))):
           td_res = rewards[i] + self.gamma * value_[i+1] * masks[i] - value_[i]
           gae = td_res + self.gamma * self.tau * masks[i] * gae
           advantages[i] = gae

        return torch.tensor(advantages).to(self.device)

    def learn(self):
        global RETURNS
        next_val = self.critic.forward(
             torch.tensor(state_).float().to(agent.device)
        ).detach().cpu().numpy().tolist()[0]

        for _ in range(self.epochs):
            states, actions, rewards, values, probs, dones = self.memory.get_nps()
            
            if self.gae:
                returns =  agent.compute_gae(rewards, dones, values, next_val=next_val)
            else:
                advantages = self.compute_adv(rewards, dones, values, next_val=next_val)

            probs   =  torch.tensor(probs  ).reshape(self.play_steps, self.n_actions).detach().to(self.device)
            states  =  torch.tensor(states ).float().to(self.device)
            actions =  torch.tensor(actions).reshape(self.play_steps, self.n_actions).detach().to(self.device)
            values  =  torch.tensor(values ).reshape(self.play_steps).detach().to(self.device)
            dones   =  torch.tensor(dones  ).to(self.device)

            if self.gae:
                advantages = returns - values
            else: 
                returns = advantages + values

            returns =  returns.reshape(self.play_steps).detach().to(self.device)
            
            RETURNS.append(returns.mean().cpu().detach().numpy())

            batches = self.memory.get_batches(self.mini_batch_size)

            for batch in batches:

                old_log_probs =      probs[batch]
                state         =     states[batch]
                action        =    actions[batch]
                return_       =    returns[batch]
                adv_          = advantages[batch].reshape(self.mini_batch_size, 1)
                
                if self.adv_norm:
                    adv_ = (adv_ - adv_.mean()) / ( \
                                     adv_.std() + 1e-4)

                epsilon = 0.2
                dist   = self.actor.forward(state)
                value_ = self.critic.forward(state)
                new_log_probs = dist.log_prob(action)

                entropy = dist.entropy().mean()
                #print(entropy)

                ratio = new_log_probs.exp() - old_log_probs.exp()
                surr1 = ratio * adv_
                surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * adv_

                a_loss = -torch.min(surr1, surr2).mean()
                c_loss = (return_ - value_)**2
                c_loss = c_loss.mean()

                total_loss = a_loss + 0.25*c_loss - 0.001*entropy
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

def animate(i):
    plt.cla()
    plt.plot(agent.avg_scores)

def setter(args, agent):
    agent.input_shape = args.input_shape
    agent.n_actions = args.n_actions
    agent.std = args.std
    agent.play_steps = args.play_steps
    agent.action_limit = args.action_limit

#args = parser.parse_args()

env = gym.make('CarRacing-v0')
agent = PPOAgent(env)
#setter(args, agent)
frame = 0
high_score = -np.inf
returns = None
#env.render()
ani = FuncAnimation(plt.gcf(), animate, interval=1000)

env.render()
for episode in range(50):
    state = env.reset()
    done = False
    score = 0
    while not done:
        state = state.reshape(-1)
        action, value, prob = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        score += reward

        agent.memory.store_memory(state, action, reward, value, prob, 1-done)
        env.render()
        state_ = state_.reshape(-1)
        state = state_
        frame += 1
        if frame % agent.play_steps == 0:
            print('learn')
            agent.learn()
            agent.memory.reset()

    agent.scores.append(score)
    agent.avg_scores.append(np.mean(agent.scores))
    high_score = max(high_score, score)
    avg = np.mean(agent.scores)

    print(f'episode: {episode}, high_score: {high_score}, score: {score}, avg: {avg}')

plt.show()
plt.cla()
plt.plot(RETURNS)
plt.show()



#plt.savefig('plot.png')
