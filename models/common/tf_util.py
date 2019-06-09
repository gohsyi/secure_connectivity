import os
import tensorflow as tf


def get_activation(ac_fn):
    """
    get activation function
    :param ac_fn: name of activation function,
                  eg: 'relu', 'sigmoid', 'elu', 'tanh'
    :return: corresponding activation function
    """

    if ac_fn == 'tanh':
        return tf.nn.tanh
    elif ac_fn == 'relu':
        return tf.nn.relu
    elif ac_fn == 'sigmoid':
        return tf.nn.sigmoid
    elif ac_fn == 'elu':
        return tf.nn.elu
    else:
        raise ValueError


def get_optimizer(opt_fn):
    """
    get optimizer function
    :param opt_fn: name of optimizer method
                   eg: 'sgd', 'adam', 'adagrad'
    :return:
    """
    if opt_fn == 'gd':
        return tf.train.GradientDescentOptimizer
    elif opt_fn == 'adam':
        return tf.train.AdamOptimizer
    elif opt_fn == 'adagrad':
        return tf.train.AdagradOptimizer
    elif opt_fn == 'rms':
        return tf.train.RMSPropOptimizer
    elif opt_fn == 'momentum':
        return tf.train.MomentumOptimizer
    else:
        raise ValueError


def get_session():
    """
    get default session of tensorflow,
    allowing soft placement for the better use of GPU

    :return: default tf session
    """

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)


def sample_k(logits, k):
    """
    sample the largest k logits
    :param logits: logits before softmax
    :param k: number of samples
    :return:
        tensor, indices of the largest k logits, represented with onehot
        the shape is (batch_size x act_size)
    """

    act_size = logits.shape[-1]
    noise = tf.random_uniform(tf.shape(logits))
    _, indices = tf.nn.top_k(logits - tf.log(-tf.log(noise)), k=k)

    return tf.reduce_sum(tf.one_hot(indices, act_size, dtype=tf.int32), 1)
