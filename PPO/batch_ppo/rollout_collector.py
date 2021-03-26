import math
import random

import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

DEBUG = False
def dbg_tensor_print(x):
    print(f"shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")


class RolloutCollector:
    def __init__(self, config, agent, envs):
        self.device = agent.device
        self.agent = agent
        self.env = envs

        self.size = config['rollout_length']
        self.num_workers = config['num_workers']
        self.batch_size = config['batch_size']
        self.state_shape = config['state_shape']#env.observation_space.shape[0]
        self.action_shape = config['action_shape']#env.action_space.shape[0]

        self.reset()
        self.scores = []
        self.config = config
        self.state = self.env.reset()

        self.gamma = config['gamma']
        self.tau = config['tau']

    def reset(self):
        batch_shape = [self.batch_size, self.size+1, self.num_workers]
        self.states = torch.zeros(batch_shape + [*self.state_shape], dtype=torch.float32).to(self.device)
        self.actions = torch.zeros(batch_shape + [*self.action_shape], dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros(batch_shape, dtype=torch.float16).to(self.device)

        self.values = torch.zeros(batch_shape , dtype=torch.float32).to(self.device)
        self.returns = None
        self.advantages = None
        self.probs = torch.zeros(batch_shape + [*self.action_shape], dtype=torch.float32).to(self.device)
        self.dones = torch.zeros(batch_shape, dtype=torch.int16).to(self.device)

    def collect_samples(self):
        score = 0
        with torch.no_grad():
            # for inference

            for batch in range(self.batch_size):
                # batch
                for step in range(self.size+1):
                    #step
                    state = torch.Tensor(self.state).float().to(self.device)
                    policy_dist = self.agent.actor(state)
                    action = policy_dist.sample()

                    action = self.agent.clamp_action(action)    #   depends on env
                    cpu_actions = action.cpu().detach().numpy()

                    values = self.agent.critic(state).detach()
                    prob = policy_dist.log_prob(action).detach()

                    if self.config['debug']:
                        print(f'actions: {action}, probs: {prob}, values: {value}')

                    state_, reward, done, info = self.env.step(cpu_actions)

                    score += np.mean(reward)
                    self.states[batch, step, :, :] = state
                    self.actions[batch, step, :, :] = action
                    self.rewards[batch, step, :] = torch.FloatTensor(reward).to(self.device)
                    self.values[batch, step, :] = values[:,0]
                    self.probs[batch, step, :, :] = prob
                    self.dones[batch, step, :] = torch.FloatTensor(1-done).to(self.device)
                    self.state = state_

                    if self.config['debug']:
                        time.sleep(0.5)

        score = np.mean(score)
        self.scores.append(score)

        advantages, returns = self.compute_gae()
        self.advantages = advantages
        self.returns = returns

        return score, np.mean(self.scores)

    def compute_gae(self, rewards=None, values=None, dones=None):
        advantages = torch.zeros_like(self.rewards, dtype=torch.float32)
        returns = torch.zeros_like(self.rewards, dtype=torch.float32)
        gae = 0
        for i in reversed(range(self.size)):
            td_res = self.rewards[:, i] + self.gamma * self.values[:,i+1] * self.dones[:,i] - self.values[:,i]
            gae = td_res + self.gamma * self.tau * self.dones[:,i] * gae
            advantages[:,i] = gae
            returns[:,i] = gae + self.values[:,i]

        return advantages, returns

    def batches(self):
        indices = torch.randperm(self.size)

        for i in range(self.size):
            index = indices[i]
            shape = self.batch_size * self.num_workers
            old_log_probs =      self.probs[:, index,:,:].detach().reshape(shape, *self.action_shape)
            state         =     self.states[:, index,:,:].detach().reshape(shape, *self.state_shape)
            action        =    self.actions[:, index,:,:].detach().reshape(shape, *self.action_shape)
            return_       =    self.returns[:, index].detach().reshape(shape, 1)
            adv_          = self.advantages[:, index].detach().reshape(shape, 1)
            yield old_log_probs, state, action, adv_, return_
