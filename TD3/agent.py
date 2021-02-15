"""
This contains the agent code for TD3
4-11-2020, Kae.
MIT License
"""
import torch
from nets import *
from cpprb import ReplayBuffer
import numpy as np
import random


class Agent:
    def __init__(self, input_shape, env=None,
                 n_actions=2, d=2, gamma=0.99, tau=5*10e-3,
                 actor=None,
                 critic=None,
                 mem_size=1000000, batch_size=64,
                 low_action=None, high_action=None):

        if actor is None:
            actor = {'fc1': 256, 'fc2': 256, 'alpha': 3e-4}
        if critic is None:
            critic = {'fc1': 256, 'fc2': 256, 'alpha': 3e-4}

        self.noise_scale = 0.2
        self.noise_clamp = 0.5

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.time_step = 0
        self.store_time = 0
        self.cooldown = 100
        self.learn_step_cntr = 0

        self.high_action = env.action_space.high if env else high_action
        self.low_action = env.action_space.low if env else low_action

        self.gamma = gamma
        self.d = d
        self.tau = tau
        self.batch_size = batch_size

        self.learn_calls = 0
        self.critic_updates = 0
        self.actor_updates = 0

        self.actor = ActorNetwork(self.input_shape, self.n_actions, actor['fc1'], actor['fc2'], 'Actor', actor['alpha'])
        self.critic = CriticNetwork(self.input_shape, self.n_actions, critic['fc1'], critic['fc2'], 'Critic_1', critic['alpha'])
        self.critic2 = CriticNetwork(self.input_shape, self.n_actions, critic['fc1'], critic['fc2'], 'Critic_2', critic['alpha'])

        self.target_actor = ActorNetwork(self.input_shape, self.n_actions, actor['fc1'], actor['fc2'], 'Target_Actor', actor['alpha'])
        self.target_critic = CriticNetwork(self.input_shape, self.n_actions, critic['fc1'], critic['fc2'], 'Target_Critic_1',
                                    critic['alpha'])
        self.target_critic2 = CriticNetwork(self.input_shape, self.n_actions, critic['fc1'], critic['fc2'], 'Target_Critic_2',
                                     critic['alpha'])

        self.memory = ReplayBuffer(mem_size, env_dict={
            "obs":      {"shape": self.input_shape     },
            "act":      {"shape": self.n_actions       },
            "rew":      {                              },
            "next_obs": {"shape": self.input_shape     },
            "done":     {                              }
        })

        torch.autograd.set_detect_anomaly(True)

        self.update_actor(tau=1)
        self.update_critic(tau=1)

    def choose_action(self, state):
        self.actor.eval()
        self.time_step += 1
        if self.time_step < self.cooldown:
            action = np.random.uniform(low=self.low_action[0], high=self.high_action[0], size=self.n_actions)
            return action

        observation = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device).squeeze(0)
        #print(mu.shape)

        noise = torch.randn_like(mu) * self.noise_scale
        noise = torch.clamp(noise, -self.noise_clamp, self.noise_clamp)

        mu += noise
        mu = mu.clamp(self.low_action[0], self.high_action[0])
        self.actor.train()

        return mu.cpu().detach().numpy()

    def store_transition(self, state, action, reward, state_, done):
        self.store_time += 1
        self.memory.add(obs=state, act=action, rew=reward, next_obs=state_, done=done)

    def save_checkpoint(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.target_critic2.save_checkpoint()

    def load_checkpoint(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic2.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.target_critic2.load_checkpoint()

    def learn(self):
        self.learn_calls += 1
        if self.batch_size > self.store_time:
            return
        # else:
        #     print('we have enough samples')

        self.learn_step_cntr += 1

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()

        # sample memory
        sample = self.memory.sample(self.batch_size)
        states, actions, rewards, states_, dones = [sample[i] for i in sample]
        states = torch.tensor(states).to(torch.float).to(self.actor.device)
        actions = torch.tensor(actions).to(torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards).to(torch.float).to(self.actor.device)
        states_ = torch.tensor(states_).to(torch.float).to(self.actor.device)
        dones = torch.tensor(dones).to(torch.bool).to(self.actor.device)

        # updating the two critics
        with torch.no_grad():
            actions_ = self.target_actor(states_)
            noise = torch.randn_like(actions_) * self.noise_scale
            noise = torch.clamp(noise, -self.noise_clamp, self.noise_clamp)
            actions_ += noise
            actions_ = torch.clamp(actions_, self.low_action[0], self.high_action[0])

            critic_1 = self.target_critic(states_, actions_)
            critic_2 = self.target_critic2(states_, actions_)

            vals = torch.min(critic_1, critic_2)
            vals[dones] = 0.0

        critic = self.critic(states, actions)
        critic2 = self.critic2(states, actions)
        #
        # print(vals.shape, rewards.shape)
        #
        target = rewards + self.gamma*vals

        # print(target.shape)

        loss1 = F.mse_loss(target, critic).to(self.critic.device)
        loss2 = F.mse_loss(target, critic2).to(self.critic2.device)
        #print(loss1, loss2, critic_1.shape, critic2.shape)
        loss = loss1 + loss2
        loss.backward()

        self.critic.optimizer.step()
        self.critic2.optimizer.step()

        self.critic_updates += 1

        self.update_critic()
        # update the actor

        if self.learn_step_cntr < self.cooldown:
            return
        # else:
        #     print('actor cooldown over')

        if self.learn_step_cntr % self.d != 0:
            return

        targ = self.critic(states, self.actor(states))
        actor_loss = -torch.mean(targ)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.actor_updates += 1

        self.update_actor()

    def update_critic(self, tau=None):
        if tau is None:
            tau = self.tau

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update_actor(self, tau=None):
        if tau is None:
            tau = self.tau

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

