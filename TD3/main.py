"""
This contains the main code for TD3
4-11-2020, Kae.
MIT License
"""


from matplotlib import pyplot as plt
from agent import Agent
import gym
import numpy as np
from continuous_cartpole import ContinuousCartPoleEnv

# TODO
# remove the extra critic??
# add dueling to critic
# add PER

stats_dict = {'Episode': [],
              'Average': [],
              'min': [],
              'max': [],
              'Moving Average': [],
              'model_accuracy': []
              }

def plot(stats_dict, filename='TD3_002.png'):

    plt.style.use('fivethirtyeight')
    # sns.set(style='darkgrid', palette='bright', font_scale=0.9)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(5, 4)
    ax = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[3:5, :2])
    ax5 = fig.add_subplot(gs[3:5, 2:4])

    def plot_stats():
        ax.cla()
        ax.scatter(stats_dict['Episode'], stats_dict['model_accuracy'], s=5, label='Scores')
        ax.plot(stats_dict['Episode'], stats_dict['Moving Average'], linewidth=0.9, label='Moving Average')
        ax.set_title('Training History')
        ax.legend()

        ax2.cla()
        ax2.plot(stats_dict['Episode'], stats_dict['max'], linewidth=0.5, label='Max Reward')
        ax2.plot(stats_dict['Episode'], stats_dict['min'], linewidth=0.5, label='Min Reward')
        ax2.plot(stats_dict['Episode'], stats_dict['Average'], linewidth=0.5, label='Avg Reward')
        ax2.tick_params(axis='x', labelbottom=False)
        ax2.set_title('Game averages')
        ax2.legend()

        ax5.cla()
        ax5.plot(stats_dict['Episode'], stats_dict['Average'], linewidth=0.5)
        ax5.set_title('Average')

        plt.savefig(filename)
        plt.show()

    plot_stats()


if __name__ == '__main__':
    env = ContinuousCartPoleEnv()
    #agent = Agent(env.observation_space.shape, env, n_actions=env.action_space.shape[0], tau=0.005)
    agent = Agent(input_shape=(4,), n_actions=1, high_action=[1.0], low_action=[-1.0])

    high_score, min_score = -np.inf, np.inf
    LOAD = False

    if LOAD:
        agent.load_checkpoint()

    render = True

    n_games = 500
    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            #print(action)
            obs_, reward, done, info = env.step(action)
            #print(obs)
            score += reward
            if render:
                env.render()
            agent.store_transition(obs, action, reward, obs_, done)
            agent.learn()

            obs = obs_

        high_score = max(score, high_score)
        if score > high_score:
            agent.save_checkpoint()
        min_score = min(score, min_score)

        avg_score = np.mean(stats_dict['model_accuracy'][-100:])
        total_avg = np.mean(stats_dict['model_accuracy'])

        stats_dict['Episode'].append(i)
        stats_dict['Average'].append(total_avg)
        stats_dict['max'].append(high_score)
        stats_dict['min'].append(min_score)
        stats_dict['Moving Average'].append(avg_score)
        stats_dict['model_accuracy'].append(score)

        #print(f"Episode: {i+1}, Score: {score}, Avg_score: {avg_score}, High Score: {high_score}")
        print(("ep {}: high-score {:12.3f}, score {:12.3f}, Avg_score {:12.3f}, Critic/Total {}/{}, Actor/Total {}/{}").format(
            i+1, high_score, score, avg_score, agent.critic_updates, agent.learn_calls, agent.actor_updates, agent.learn_calls))

    plot(stats_dict)
    agent.save_checkpoint()
