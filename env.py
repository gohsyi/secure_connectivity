"""
This is where we define the environment of both defender and its attacker

* state: adjacent matrix of a random undirected connected graph
* action: (defender's action, attacker's action)
* reward: (defender's reward, attacker's reward)
          if the graph is still connected after the attacking, defender gets +1, attacker gets -1
          else defender got -1, attacker gets +1

Tue May 21, 2019, Hongyi Guo
"""

from queue import Queue

import numpy as np


class Env(object):
    """
    We use this class to :
        __init__:
        - Create the environment model

        reset():
        - reset the environment

        step():
        - take a step in this environment
    """

    def __init__(self, n, m, k):
        self.num_envs = 1
        self.n = n
        self.m = m

        self.ob_size = n * n
        self.act_size = n * n
        self.n_actions = k

        self.max_edges = self.n * (self.n - 1) // 2
        self.m = min(self.m, self.max_edges)
        self.m = max(self.m, self.n - 1)

        # map: int index -> (row, col)
        self.map = np.array([(i, j) for i in range(self.n) for j in range(i)])

    def reset(self):
        return self.gen_connected_graph()

    def step(self, action):
        """
        take a step
        :param action: a tuple, (defender's action, attacker's action)
        :return: tuple, (new_observation, reward)
        """

        d_action, a_action = action
        for aa in a_action:
            if aa not in d_action:
                i, j = self.map[aa]
                self.adj_mat[i][j] = 0

        defender_rew = 1 if self.is_connected() else -1

        return self.gen_connected_graph(), defender_rew, True, None

    def gen_connected_graph(self):
        """
        generate a connected graph with self.n vertices and self.m edges
        :return: the adjacent matrix of the generated graph
        """

        while True:
            self.adj_mat = np.eye(self.n)
            edges = np.random.choice(self.max_edges, self.m)

            for i, j in self.map[edges]:
                self.adj_mat[i, j] = 1
                self.adj_mat[j, i] = 1

            if self.is_connected():
                break

        return np.reshape(self.adj_mat, -1)

    def is_connected(self):
        """
        check if the given adjacent matrix is connected or not
        :return: boolean, True if the graph is connected
        """

        # bfs
        Q = Queue()
        vis = [False for _ in range(self.n)]
        Q.put(0)
        vis[0] = True

        while not Q.empty():
            u = Q.get()
            for v, e in enumerate(self.adj_mat[u]):
                if e == 1 and not vis[v]:
                    vis[v] = True
                    Q.put(v)

        return vis.count(False) == 0  # if all vertices are visited, return True


def build_env(n_vertices, n_edges, n_actions):
    return Env(n=n_vertices, m=n_edges, k=n_actions)


# testing
if __name__ == '__main__':
    env = Env(10, 20, 10)
    for _ in range(10):
        print(env.gen_connected_graph())

    pass
