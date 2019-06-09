import numpy as np


class Rule(object):
    """
    A rule-based model
    """

    def __init__(self, env):
        self.n = env.n
        self.n_actions = env.n_actions

    def step(self, obs):
        neighbors = []
        others = []
        for i in range(1, self.n):
            if obs[i] == 1:
                neighbors.append(i)
            else:
                others.append(i)
        neighbors = np.array(neighbors)
        others = np.array(others)

        if self.n_actions > neighbors.shape[0]:
            noUse = others[np.random.choice(others.shape[0], self.n_actions-neighbors.shape[0], replace=False)]
            return np.append(neighbors, noUse), 0
        else:
            return neighbors[np.random.choice(neighbors.shape[0], self.n_actions, replace=False)], 0

    def train(self, obs, d_rewards, d_actions, d_values):
        pass

    def output(self, info):
        pass

    def save(self, save_path):
        pass

    def load(self, load_path):
        pass
