import numpy as np
import gym
import math
import time
from agent import DDPGAgent
from utils import plotLearning
import matplotlib.pyplot as plt

n_games = 950
env = gym.make("LunarLanderContinuous-v2")
high_score = -math.inf
scores, avg_scores = [],[]
load = False
render = False

agent = DDPGAgent(env.observation_space.shape, env.action_space.shape[0])

if load:
    agent.load()

agent.save()

for i in range(n_games):
    done = False
    obs = env.reset()
    agent.noise.reset()
    score  = 0
    while not done:
        action = agent.choose_action(obs)
        #print(action) 
        obs_, reward, done, info = env.step(action)
        score += reward
        if render:
            env.render()
        agent.remember(obs, action, reward, obs_, done)
        agent.learn()
        obs = obs_

    high_score = max(high_score, score)
    avg_score = np.mean(scores[-100:])

    scores.append(score)
    avg_scores.append(avg_score)

    print(f"Episode: {i}, Score: {score}, Avg_score: {avg_score}, High_Score: {high_score}")

agent.save()

a = [i for i in range(n_games)]
plotLearning(a, avg_scores, "DDGP_03.png")
plt.show()
