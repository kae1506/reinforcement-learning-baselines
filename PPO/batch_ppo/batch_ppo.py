import gym

from continuous_cartpole import ContinuousCartPoleEnv
from rollout_collector import RolloutCollector
from multiprocessing_env import SubprocVecEnv
from agent import Agent
from logger import Logger
from config import config

def make_env():
    def _thunk():
        env = ContinuousCartPoleEnv()
        return env
    return _thunk

if __name__ == "__main__":
    test_env = ContinuousCartPoleEnv()
    envs = SubprocVecEnv([make_env() for i in range(config['num_workers'])])

    train_run = 0
    agent = Agent(test_env, config)
    logger = Logger(['scores_test', 'scores_train', 'value_loss', 'policy_loss'])
    rollout_collector = RolloutCollector(config, agent, envs)

    for _ in range(100):
        train_run += 1
        score, avg_score = agent.play_test_episodes(train_run)

        rollout_collector.collect_samples()
        rollout_collector.compute_gae()
        agent.learn(rollout_collector)
        rollout_collector.reset()

        logger.log([score, avg_score, agent.value_loss, agent.policy_loss])

    logger.plot(sub_plots=2, stat_per_plot=2, filename='ppo_continuous_cartpole_100.png')
