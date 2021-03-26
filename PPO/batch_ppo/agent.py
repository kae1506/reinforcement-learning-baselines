import math
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from models import Actor, Critic

'''
TODO:
-seperate actor and critic optimizers
    |-update them on seperate intervals like TD3 if possible
        |- does this interfere with actor epochs??

-add entropy maximization as exploration
-try letting network output the deviation
-add network load and save
-add graphing
-add lstm (might want that to be alternate version)
'''

class Agent:
    def __init__(self, env, config):
        # SETTINGS
        self.input_shape = config['state_shape'] #env.observation_space.shape[0]
        self.n_actions = config['action_shape']

        self.env = env
        self.config = config

        self.epochs = config['epochs']

        self.action_limit = config['action_limit']

        self.adv_norm = config['adv_norm']
        self.epsilon = config['epsilon']

        self.high_score = -np.inf
        self.scores = []

        self.value_loss = 0
        self.policy_loss = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config['lr']
        )

    def clamp_action(self, action):
        return action.clamp(-self.action_limit, self.action_limit)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def play_test_episodes(self, episode, episodes=1):
        env = self.env

        with torch.no_grad():
            for j in range(episodes):

                state = self.env.reset()
                done = False
                score = 0

                while not done:
                    action, _, _ = self.choose_action(state)
                    state_, reward, done, info = env.step(action[0])

                    score += reward
                    state = state_
                    env.render()

                    if self.config['debug']:
                        time.sleep(0.1)

                self.scores.append(score)
                avg_score = np.mean(self.scores[-100:])
                self.high_score = np.max([score, self.high_score])

                print(f'episode: {episode}, score: {score}, high_score: {self.high_score}, ' +
                    f' average: {avg_score}')

        return score, avg_score


    def choose_action(self, state):
        state = torch.tensor([state]).float().to(self.device)
        dist = self.actor.forward(state)
        value = self.critic.forward(state)
        action = dist.rsample()
        prob = dist.log_prob(action)

        action = action.clamp(-self.action_limit, self.action_limit)
        return action.detach().cpu().numpy(), value.detach().cpu().numpy(), prob.detach().cpu().numpy()


    def learn(self, collector):
        c_losses = []
        a_losses = []

        for epoch in range(self.epochs):
            for batch in collector.batches():
                old_log_probs, state, action, adv_, return_ = batch
                if self.config['debug']:
                    print(f'old_log_probs: {old_log_probs} \n',
                          f'state:         {state}         \n',
                          f'action:        {action}        \n',
                          f'return_:       {return_}       \n',
                          f'adv_:          {adv_}          \n',
                          f'epoch:         {epoch}')

                if self.config['debug']:
                    print(f'actions: {action} \n probs: {old_log_probs}')

                if self.adv_norm:
                    adv_ = (adv_ - adv_.mean()) / ( \
                                     adv_.std() + 1e-4)

                dist   = self.actor.forward(state)
                new_value = self.critic.forward(state)
                new_log_probs = dist.log_prob(action)

                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * adv_
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv_

                a_loss = - torch.min(surr1, surr2).mean()
                c_loss = (return_  - new_value).pow(2).mean()

                total_loss = 0.5 * c_loss + a_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                c_losses.append(torch.mean(c_loss.detach()).cpu().numpy())
                a_losses.append(torch.mean(a_loss.detach()).cpu().numpy())

        self.value_loss = np.mean(c_losses)
        self.actor_loss = np.mean(a_losses)
