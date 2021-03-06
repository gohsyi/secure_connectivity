import numpy as np
import tensorflow as tf

from tensorflow import losses


from common.argparser import args
from common.util import get_logger

from models.common import tf_util
from models.common import set_global_seeds

from models.a2c.policy import build_policy
from models.a2c.utils import find_trainable_variables
from models.a2c.runner import Runner


class Model(object):
    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """

    def __init__(
            self,
            name,  # name of this model
            env,  # environment
            latents,  # network hidden layer sizes
            lr=1e-5,  # learning rate
            activation='relu',  # activation function
            optimizer='adam',  # optimization function
            vf_coef=0.1,  # vf_loss weight
            ent_coef=0.01,  # ent_loss weight
            max_grad_norm=0.5):  # how frequently the logs are printed out

        sess = tf_util.get_session()

        # output to both file and console
        logger = get_logger(name)
        output = logger.info
        output(args)

        activation = tf_util.get_activation(activation)
        optimizer = tf_util.get_optimizer(optimizer)

        # lr = tf.train.polynomial_decay(
        #     learning_rate=lr,
        #     global_step=tf.train.get_or_create_global_step(),
        #     decay_steps=total_epoches,
        #     end_learning_rate=lr/10,
        # )

        ob_size = env.ob_size
        act_size = env.act_size
        n_actions = env.n_actions

        # placeholders for use
        X = tf.placeholder(tf.float32, [None, ob_size], 'observation')
        A = tf.placeholder(tf.int32, [None, act_size], 'action')
        ADV = tf.placeholder(tf.float32, [None], 'advantage')
        R = tf.placeholder(tf.float32, [None], 'reward')

        with tf.variable_scope(name):
            policy = build_policy(
                observations=X,
                act_size=act_size,
                n_actions=n_actions,
                latents=latents,
                vf_latents=latents,
                activation=activation
            )

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = policy.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(policy.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(policy.vf), R)

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # gradients and optimizer
        params = find_trainable_variables(name)
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        # 3. Make op for one policy and value update step of A2C
        trainer = optimizer(learning_rate=lr)

        _train = trainer.apply_gradients(grads)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name))

        def step(obs):
            action, value = sess.run([policy.action, policy.vf], feed_dict={
                X: np.reshape(obs, (-1, ob_size))
            })
            return action, value

        def value(obs):
            return sess.run(policy.vf, feed_dict={
                X: np.reshape(obs, (-1, ob_size))
            })

        def train(obs, rewards, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values

            td_map = {X:obs, A:actions, ADV:advs, R:rewards}
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )

            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            saver.save(sess, save_path)
            print(f'Model saved to {save_path}')

        def load(load_path):
            saver.restore(sess, load_path)
            print(f'Model restored from {load_path}')

        self.train = train
        self.step = step
        self.value = value
        self.output = output
        self.save = save
        self.load = load

        tf.global_variables_initializer().run(session=sess)


def learn(env,
          defender,
          attacker,
          seed=None,
          nsteps=5,
          total_epoches=int(1e6),
          vf_coef=0.5,
          ent_coef=0.01,
          max_grad_norm=0.5,
          lr=1e-4,
          gamma=0.99,
          d_load_path=None,
          a_load_path=None,
          d_save_path=None,
          a_save_path=None):

    set_global_seeds(seed)

    from models.stochastic.stochastic import Stochastic
    from rule.rule2 import Rule

    # Instantiate the model objects (that creates defender_model and adversary_model)
    if defender == 'a2c':
        d_model = Model(
            name='defender',
            env=env,
            lr=lr,
            latents=args.latents,
            activation=args.activation,
            optimizer=args.optimizer,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm)
    elif defender == 'stochastic':
        d_model = Stochastic(env)
    elif defender == 'rule':
        d_model = Rule(env)
    else:
        raise NotImplementedError

    if attacker == 'a2c':
        a_model = Model(
            name='attacker',
            env=env,
            lr=lr,
            latents=args.latents,
            activation=args.activation,
            optimizer=args.optimizer,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm)
    elif attacker == 'stochastic':
        a_model = Stochastic(env)
    elif attacker == 'rule':
        a_model = Rule(env)
    else:
        raise NotImplementedError

    if d_load_path:
        d_model.load(d_load_path)
    if a_load_path:
        a_model.load(a_load_path)

    # Instantiate the runner object
    runner = Runner(
        env=env,
        d_model=d_model,
        a_model=a_model,
        bl_d_model=Rule(env),
        bl_a_model=Rule(env),
        nsteps=nsteps,
        gamma=gamma
    )

    for ep in range(total_epoches):
        # Get mini batch of experiences
        obs, rewards, actions, values, epinfos = runner.run()
        d_rewards, a_rewards = rewards
        d_actions, a_actions = actions
        d_values, a_values = values
        bl_d_rewards, bl_a_rewards = epinfos

        # train defender model if the model is not loaded
        if defender == 'a2c' and not d_load_path:
            train_results = d_model.train(obs, d_rewards, d_actions, d_values)
            pg_loss, vf_loss, ent_loss = train_results
            d_model.output(f'\n\tep:{ep}\n' +
                           f'\tpg_loss:%.3f\tvf_loss:%.3f\tent_loss:%.3f\n' % (pg_loss, vf_loss, ent_loss) +
                           f'\tavg_rew:%.2f\tavg_val:%.2f' % (float(np.mean(d_rewards)), float(np.mean(d_values))) +
                           f'\tbl_avg_rew:%.2f\t' % np.mean(bl_d_rewards))

        # train attacker model if the model is not loaded
        if attacker == 'a2c' and not a_load_path:
            train_results = a_model.train(obs, a_rewards, a_actions, a_values)
            pg_loss, vf_loss, ent_loss = train_results
            a_model.output(f'\n\tep:{ep}\n' +
                           f'\tpg_loss:%.3f\tvf_loss:%.3f\tent_loss:%.3f\n' % (pg_loss, vf_loss, ent_loss) +
                           f'\tavg_rew:%.2f\tavg_val:%.2f' % (float(np.mean(a_rewards)), float(np.mean(a_values))) +
                           f'\tbl_avg_rew:%.2f\t' % np.mean(bl_a_rewards))

    d_model.save(d_save_path)
    a_model.save(a_save_path)

    return d_model, a_model
