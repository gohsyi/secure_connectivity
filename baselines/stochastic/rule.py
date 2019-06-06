import numpy as np


class Rule(object):
    """
    A rule-based model
    """

    def __init__(self, env):
        self.act_size = env.n-1
        self.n_actions = env.n_actions

    def step(self, obs):
        return np.random.choice(self.act_size, self.n_actions, replace=False), 0

    def train(self, obs, d_rewards, d_actions, d_values):
        pass

    def output(self, info):
        pass
