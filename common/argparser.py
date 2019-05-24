import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-seed', type=int, default=0)

# algorithm setting
parser.add_argument('-model', type=str, default='a2c')

# experiment setting
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-hidsizes', type=str, default='256')
parser.add_argument('-total_epoches', type=int, default=int(80e6))
parser.add_argument('-vf_coef', type=float, default=0.1)
parser.add_argument('-ent_coef', type=float, default=0.01)
parser.add_argument('-max_grad_norm', type=float, default=0.5)
parser.add_argument('-activation', type=str, default='relu',
                    help='relu/sigmoid/elu/tanh')
parser.add_argument('-optimizer', type=str, default='adam',
                    help='adam/adagrad/gd/rms/momentum')

# environment setting
parser.add_argument('-n_vertices', type=int, default=10)
parser.add_argument('-n_edges', type=int, default=45)
parser.add_argument('-n_actions', type=int, default=10)

args = parser.parse_args()
