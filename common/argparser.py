import argparse


parser = argparse.ArgumentParser()

# gpu device
parser.add_argument('-gpu', type=str, default='-1')

# global random seed
parser.add_argument('-seed', type=int, default=0)

# algorithm setting
parser.add_argument('-d_model', type=str, default='a2c')
parser.add_argument('-a_model', type=str, default='a2c')

# experiment setting
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-lr_decay', action='store_true', default=False)
parser.add_argument('-batchsize', type=int, default=64)
parser.add_argument('-latents', type=str, default='64')
parser.add_argument('-total_epoches', type=int, default=int(2e4))
parser.add_argument('-vf_coef', type=float, default=0.1)
parser.add_argument('-ent_coef', type=float, default=0.01)
parser.add_argument('-gamma', type=float, default=0.99)
parser.add_argument('-log_interval', type=int, default=10)
parser.add_argument('-max_grad_norm', type=float, default=0.5)
parser.add_argument('-activation', type=str, default='tanh',
                    help='relu/sigmoid/elu/tanh')
parser.add_argument('-optimizer', type=str, default='adam',
                    help='adam/adagrad/gd/rms/momentum')

# environment setting
parser.add_argument('-n_vertices', type=int, default=5)
parser.add_argument('-n_edges', type=int, default=16)
parser.add_argument('-n_actions', type=int, default=2)

parser.add_argument('-note', type=str, default='test')
parser.add_argument('-d_load', type=str, default=None)
parser.add_argument('-a_load', type=str, default=None)

args = parser.parse_args()

abstract = '{}_{}_{}_{}_lr{}{}hid{}_bs{}_ep{}_grad{}_vf{}_ent{}_seed{}'.format(
    args.d_model,
    args.a_model,
    args.activation,
    args.optimizer,
    args.lr,
    '_decay_' if args.lr_decay else '_',
    args.latents,
    args.batchsize,
    args.total_epoches,
    args.max_grad_norm,
    args.vf_coef,
    args.ent_coef,
    args.seed,
)

args.latents = list(map(int, args.latents.split(',')))

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
