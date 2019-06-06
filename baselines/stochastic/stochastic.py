import numpy as np


class Stochastic(object):
    """
    A stochastic (random) model
    """

    def __init__(self, env):
        # self.act_size = env.act_size
        self.act_size = env.n
        self.n_actions = env.n_actions

    def step(self, obs):
        return np.random.choice(self.act_size-1, self.n_actions), 0

    def train(self, obs, d_rewards, d_actions, d_values):
        pass

    def output(self, info):
        pass
