import torch
import numpy as np
from nets import ActorNetwork, CriticNetwork
from cpprb import ReplayBuffer
from noise import OUActionNoise
import math

class DDPGAgent(object):
    def __init__(self, input_shape, n_actions, gamma=0.99, tau=0.001, batch_size=64, fc1_dims=400, fc2_dims=300, mem_size=1_000_000):
        self.gamma = gamma
        self.tau = tau
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(input_shape, n_actions, fc1_dims, fc2_dims, "actor")
        self.actor_ = ActorNetwork(input_shape, n_actions, fc1_dims, fc2_dims, "target_actor")

        self.critic = CriticNetwork(input_shape, n_actions, fc1_dims, fc2_dims, "critic")
        self.critic_ = CriticNetwork(input_shape, n_actions, fc1_dims, fc2_dims, "target_critic")

        self.noise = OUActionNoise(np.zeros(n_actions))

        self.memory = ReplayBuffer(mem_size, env_dict={
            "obs"     : {"shape": self.input_shape},
            "act"     : {"shape": self.n_actions  },
            "rew"     : {                         },
            "next_obs": {"shape": self.input_shape},
            "done"    : {                         }
        })

        self.update_network_params(tau=1)

    def load(self):
        self.actor.load_checkpoint()
        self.actor_.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_.load_checkpoint()

        self.memory.load()
    
    def save(self):
        self.actor.save_checkpoint()
        self.actor_.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_.save_checkpoint()

        self.memory.save()

    def choose_action(self, observation):
        self.actor.eval()
        observation = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
        self.actor.train()

        return mu.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.add(obs=state, act=action, rew=reward, next_obs=state_, done=done)

    def learn(self):
        if self.batch_size > self.memory.memCount:
            return None
        
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        states  = torch.tensor(states ).to(torch.float   ).to(self.actor.device)
        actions = torch.tensor(actions).to(torch.float   ).to(self.actor.device)
        rewards = torch.tensor(rewards).to(torch.float   ).to(self.actor.device)
        states_ = torch.tensor(states_).to(torch.float   ).to(self.actor.device)
        dones   = torch.tensor(dones  ).to(torch.bool    ).to(self.actor.device)
        
        #print(actions)
        actor = self.actor.forward(states)
        actor_ = self.actor_.forward(states_)

        critic = self.critic.forward(states, actions)
        critic_ = self.critic_.forward(states_, actor_)
        critic_[dones] = 0.0
        critic_ = critic_.view(-1)
        
        criticTd = rewards+self.gamma*critic_
        criticTd  = criticTd.view(self.batch_size, 1)
        loss = torch.nn.functional.mse_loss(criticTd, critic)
        loss.backward()
        self.critic.optimizer.step()

        actorLoss = -self.critic.forward(states, actor)
        actorLoss = torch.mean(actorLoss)
        actorLoss.backward()
        self.actor.optimizer.step()
        
        self.update_network_params()

    def update_network_params(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        actor_params_ = self.actor_.named_parameters()
        critic_params = self.critic.named_parameters()
        critic_params_ = self.critic_.named_parameters()

        target_actor = dict(actor_params)
        actor_ = dict(actor_params_)
        target_critic = dict(critic_params)
        critic_ = dict(critic_params_)

        for name in target_actor:
            target_actor[name] = tau*target_actor[name].clone() + (1-tau)*actor_[name].clone()

        for name in target_critic:
            target_critic[name] = tau*target_critic[name].clone() + (1-tau)*critic_[name].clone()

        self.actor_.load_state_dict(target_actor)
        self.critic_.load_state_dict(target_critic)
        

    
