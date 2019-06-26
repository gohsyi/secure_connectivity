# Adversarial Learning for Secure Connectivity

Course project of SJTU EE447: Mobile Internet, advised by Prof. Luoyi Fu and Prof. Xinbing Wang. We consider the connectivity determination of a network under the risk of adversarial models. The task is to design a defending strategy to predict and protect the edges that is most likely to be attacked by an attacker. In this project, we propose a framework based on reinforcement learning, in which we simultaneously train an attacker model and a defender model. The models generate actions against each other, which is similar to the training process of generative adversarial networks (GAN). To test the efficiency of our method, we also implement a rule-based method and a stochastic method. Experiments show that the RL-based model performs better than the other two methods.


## Usage

```
python main.py \
-d_model {defender model, a2c/rule/stochastic} \
-a_model {attacker modek, a2c/rule/stochastic} \
-d_load {defender model load path, option and only if a2c model is selected} \
-a_load {attacker model load path, option and only if a2c model is selected} \
-n_vertices {number of vertices in the graph} \
-n_edges {number of edges in the graph} \
-n_actions {how many edges to be protected/attacked}
```

Please refer to `common/argparser.py` for more available arguments.


## Introduction

The description of the problem is illustrated as below.

<img src="https://github.com/gohsyi/secure_connectivity/blob/master/figures/description.png" width="450" height="350" \>


## Framework & Models

We propose a framework based on reinforcement learning to build a strong defender. Inspired by GAN, we model an attacker and a defender as the same time. In the training process, a random graph is created from the environment as the observation. Then it is inputted into the attacker model and the defender model. The two models are trained simultaneously, using a2c algorithm. The attacker's action is to select some edges to delete. The defender's action is to select some edges that can't be deleted. At the end of a training step, the environment detect the connectivity of the graph. If `l` points are disconnected from the source point after the attack, the attacker will get `l/N-1` reward, and the defender will get `-l/N-1` reward as a punishment. 

To test the efficiency of our method, we implement another two methods for comparison. One is called rule-based method, where the attacker and the defender generate actions according to a certain strategy. The other is called stochastic method, where the two models select the edges randomly. We set a series of experiments to compare the ability of the above methods, and find that the RL-based method performs better than the others in both attacking and defending.

<img src="https://github.com/gohsyi/secure_connectivity/blob/master/figures/framework.png" width="600" height="600" \>


## Experiment

In the following experiments, the network has 5 vertices, 16 edges. Attacker/defender chooses 3 edges to defend/attack. And by _baseline model_, we mean model adopting rule-based defend/attack regime.

_**exp 1**_

First, we train a RL-based attacker model against a defender adopting stochastic defending regime. The results are as below, where the red line represents the rewards that our RL-based attacker gets, and the blue line represents the rewards the baseline attacker gets. It's easy to notice our RL-based attacker outperforms the baseline after less than 5,000 epochs.

<img src="https://github.com/gohsyi/secure_connectivity/blob/master/figures/exp1.png" width="420" height="320" \>

_**exp 2**_

Then, we train the same RL-based attacker. However, the defender this time is a rule-based model. Rule-based model is slightly stronger than the stochastic model. So we notice that the improvement of the RL-based attacker's performance becomes slower than in the last experiment. But near the end of the training, the reward is very close to that in the last experiment. In contrast, the rule-based attacker becomes useless facing the defender with the same rule.

<img src="https://github.com/gohsyi/secure_connectivity/blob/master/figures/exp2.png" width="420" height="320" \>

_**exp 3**_

Now, we train an RL defender model against the RL-based attacker model already trained in _**exp1**_ (freeze the parameter). Our RL-based model outperforms baseline and can defend almost every attack of the smart trained attacker.

<img src="https://github.com/gohsyi/secure_connectivity/blob/master/figures/exp3.png" width="420" height="320" \>

_**exp 4**_

In the last experiment, we mimic the training of GAN, training an RL-based attacker and an RL-based defender at the same time. We can see that both defender and attacker outperform the baselines in the end, indicating that the GAN-style training is effective.

<img src="https://github.com/gohsyi/secure_connectivity/blob/master/figures/exp4_1.png" width="420" height="320" \><img src="https://github.com/gohsyi/secure_connectivity/blob/master/figures/exp4_2.png" width="420" height="320" \>
