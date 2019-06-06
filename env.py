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


class ConnectedGraphEnv(object):
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
        """
        Initialize the environment

        Parameters
        ----------
        n: an integer
            The number of vertices
        m: an integer
            The number of edges
        k: an integer
            The number of edges to be defended/attacked
        """

        assert k < n, 'k cannot exceed n '
        assert m < n * (n-1), 'm cannot exceed n*(n-1)'

        self.num_envs = 1
        self.n = n
        self.m = m

        # source point
        self.s = 0

        self.ob_size = n * n
        self.act_size = n * (n-1)
        self.n_actions = k

        self.max_edges = self.n * (self.n - 1)

        # map: int index -> (row, col)
        # self.map = np.array([(i, j) for i in range(self.n) for j in range(i)])
        self.map = [(i, j) for i in range(n) for j in range(n)]
        for i in range(n):
            self.map.remove((i, i))
        self.map = np.array(self.map)

    def reset(self):
        # return self.gen_connected_graph()
        return self.gen_graph()

    def step(self, action):
        """
        take a step

        Parameters
        ----------
        action: a tuple
            (defender's action, attacker's action)

        Returns
        -------
        a tuple: (new_observation, reward, done, info)
        """

        return self.gen_connected_graph(), self.eval(action)[0], True, None

    def eval(self, action):
        """
        evaluate the reward of the actions

        Parameters
        ----------
        action: a tuple
            (defender_action, attacker_action), both are one-hot vectors

        Returns
        -------
        reward: a tuple
            (defender_reward, attacker_reward), attacker_reward equals -defender_reward
        """

        pre_connections = self.connections(self.adj_mat)
        adj_mat = self.adj_mat.copy()

        d_action, a_action = action

        assert d_action.shape[0] == self.act_size or d_action.shape[0] == self.n_actions
        assert a_action.shape[0] == self.act_size or a_action.shape[0] == self.n_actions

        if d_action.shape[0] == self.act_size:
            d_action = list(map(int, np.argwhere(d_action == 1)))
        else:
            d_action = d_action.tolist()

        if a_action.shape[0] == self.act_size:
            a_action = list(map(int, np.argwhere(a_action == 1)))
        else:
            a_action = a_action.tolist()

        for aa in a_action:
            if aa not in d_action:
                i, j = self.map[aa]
                adj_mat[i][j] = 0

        # defender_rew = 1 if self.is_connected() else -1
        defender_rew = (self.connections(adj_mat) - pre_connections) / (self.n-1)  # normalize -> [-1, 0]
        return defender_rew, -defender_rew

    def gen_connected_graph(self):
        """
        generate a connected graph with `self.n` vertices and `self.m` edges

        Returns
        -------
        the adjacent matrix of the generated graph
        """

        while True:
            self.adj_mat = np.eye(self.n)
            edges = np.random.choice(self.max_edges, self.m)
            for i, j in self.map[edges]:
                self.adj_mat[i, j] = 1
            if self.is_connected():
                break

        return np.reshape(self.adj_mat, -1)

    def gen_graph(self):
        """
        generate a graph with `self.n` vertices and `self.m` edges

        Returns
        -------
        the adjacent matrix of the generated graph
        """

        self.adj_mat = np.eye(self.n)
        edges = np.random.choice(self.max_edges, self.m)
        for i, j in self.map[edges]:
            self.adj_mat[i, j] = 1
        return np.reshape(self.adj_mat, -1)

    def is_connected(self):
        """
        check if the given adjacent matrix is connected or not
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

    def connections(self, adj_mat):
        """
        count the number of vertices that `s` is directly or indirectly connected to
        """

        # bfs
        Q = Queue()
        vis = [False for _ in range(self.n)]
        Q.put(self.s)
        vis[self.s] = True

        while not Q.empty():
            u = Q.get()
            for v, e in enumerate(adj_mat[u]):
                if e == 1 and not vis[v]:
                    vis[v] = True
                    Q.put(v)

        return vis.count(True)


def build_env(n_vertices, n_edges, n_actions):
    """
    create an Env entity
    """

    return ConnectedGraphEnv(n=n_vertices, m=n_edges, k=n_actions)
