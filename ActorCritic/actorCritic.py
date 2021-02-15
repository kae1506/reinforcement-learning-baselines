'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import plotLearning
import matplotlib.pyplot as plt
import gym

class Network(nn.Module):
    def __init__(self, inputShape, numActions, lr):
        super().__init__()
        self.fc1 = nn.Linear(*inputShape, 512)
        self.fc2 = nn.Linear(512, 128)

        self.actor = nn.Linear(128, numActions)
        self.critic = nn.Linear(128, 1)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, state):
        state = torch.tensor(state).float().to(self.device)
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))

        actor = self.actor(state)
        critic = self.critic(state)

        return actor, critic

class Agent:
    def __init__(self, numActions, inputShape):
        self.model = Network(inputShape, numActions, 0.00001)
        self.gamma = 0.99
        self.logProbs = None

    def choose_action(self, state):
        actor, critic = self.model(state)
        #print(actor.shape, critic.shape)
        policy = F.softmax(actor, dim=0)
        actionProbs = torch.distributions.Categorical(policy)
        action = actionProbs.sample()
        self.logProbs = actionProbs.log_prob(action)
        return action.item()

    def learn(self, state, reward, newState, done):
        self.model.optimizer.zero_grad()

        _, critic = self.model(state)
        _, critic_ = self.model(newState)

        reward = torch.tensor(reward).to(self.model.device)
        td = reward + self.gamma*critic_ + (1-int(done))- critic

        actorLoss = -self.logProbs * td
        criticLoss = td**2

        (actorLoss+criticLoss).backward()
        self.model.optimizer.step()

if __name__ == '__main__':
    agent = Agent(2, (4,))
    print(agent.model.device)
    env = gym.make('CartPole-v1').unwrapped
    scores = []
    avg_scores = []

    for i in range(500):
        state = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            env.render()
            agent.learn(state, reward, state_, done)
            state = state_
            score += reward

        avg_score = np.mean(scores[-100:])
        scores.append(score)
        avg_scores.append(avg_score)
        print(f"episode: {i}, score: {score}, average_score: {avg_score}")

    plotLearning([i for i in range(200)], avg_scores, scores, "AC.png")
    plt.show()
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, lr, inputDims, numActions, fc1Dims=1024, fc2Dims=512):
        super().__init__()
        self.inputDims = inputDims
        self.numActions = numActions
        self.fc1Dims = fc1Dims
        self.fc2Dims = fc2Dims

        #   primary network
        self.fc1 = nn.Linear(*inputDims, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)

        #   tail networks
        self.policy = nn.Linear(self.fc2Dims, self.numActions)
        self.critic = nn.Linear(self.fc2Dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #   self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, observation):
        state = torch.tensor(observation).float().to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy = self.policy(x)
        value = self.critic(x)
        return policy, value

class ActorCriticAgent():
    def __init__(self, lr, inputDims, numActions, gamma=0.99, layer1Size=1024, layer2Size=512):
        self.gamma = gamma
        self.actorCritic = ActorCriticNetwork(lr, inputDims, numActions, layer1Size, layer2Size)
        self.logProbs = None

    def chooseAction(self, observation):
        policy, _ = self.actorCritic.forward(observation)
        policy = F.softmax(policy, dim=0)
        actionProbs = torch.distributions.Categorical(policy)
        action = actionProbs.sample()
        self.logProbs = actionProbs.log_prob(action)
        return action.item()

    def learn(self, state, reward, nextState, done):
        self.actorCritic.optimizer.zero_grad()

        _, criticValue = self.actorCritic.forward(state)
        _, nextCriticValue = self.actorCritic.forward(nextState)

        reward = torch.tensor(reward).float().to(self.actorCritic.device)
        td = reward + self.gamma * nextCriticValue * (1 - int(done)) - criticValue

        actorLoss = -self.logProbs * td
        criticLoss = td**2

        (actorLoss + criticLoss).backward()
        self.actorCritic.optimizer.step()

if __name__ == '__main__':
    import gym
    import math
    from matplotlib import pyplot as plt
    
    agent = ActorCriticAgent(
        lr=0.00001, inputDims=(4,), gamma=0.99, numActions=2, layer1Size=1024, layer2Size=512)
    env = gym.make("CartPole-v0")

    scoreHistory = []
    numEpisodes = 200
    numTrainingEpisodes = 50
    highScore = -math.inf
    recordTimeSteps = math.inf
    for episode in range(numEpisodes):
        done = False
        observation = env.reset()
        score, frame = 0, 1
        while not done:
            if episode > numTrainingEpisodes:
                env.render()
            action = agent.chooseAction(observation)
            nextObservation, reward, done, info = env.step(action)
            agent.learn(observation, reward, nextObservation, done)
            observation = nextObservation
            score += reward
            frame += 1
        scoreHistory.append(score)

        recordTimeSteps = min(recordTimeSteps, frame)
        highScore = max(highScore, score)
        print(( "ep {}: high-score {:12.3f}, shortest-time {:d}, "
                "score {:12.3f}, avg_score {:12.3f}").format(
            episode, highScore, recordTimeSteps, score, np.mean(scoreHistory)))

    fig = plt.figure()
    meanWindow = 10
    meanedScoreHistory = np.convolve(scoreHistory, np.ones(meanWindow), 'valid') / meanWindow
    plt.plot(np.arange(0, numEpisodes-1, 1.0), meanedScoreHistory)    
    plt.ylabel("score")
    plt.xlabel("episode")
#'''
