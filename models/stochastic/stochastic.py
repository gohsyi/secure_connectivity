import numpy as np


class Stochastic(object):
    """
    A stochastic (random) model
    """

    def __init__(self, env):
        self.act_size = env.act_size
        self.n_actions = env.n_actions

    def step(self, obs):
        return np.random.choice(self.act_size, self.n_actions, replace=False), 0

    def train(self, obs, d_rewards, d_actions, d_values):
        pass

    def output(self, info):
        pass

    def save(self, save_path):
        pass

    def load(self, load_path):
        pass
