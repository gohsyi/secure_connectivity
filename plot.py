import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")


parser = argparse.ArgumentParser()
parser.add_argument('-smooth', type=float, default=0)
args = parser.parse_args()


for root, dirs, files in os.walk('logs'):
    for f in files:
        if f[0] != '.' and os.path.splitext(f)[-1] == '.log':  # process .log
            p = os.path.join(root, f)
            print('processing %s' % p)
            pg_loss, vf_loss, ent_loss, rew, val = [], [], [], [], []

            for line in open(p):
                line = line.split()
                for x in line:
                    x = x.split(':')
                    if x[0] == 'ep' and x[1] == '0':
                        pg_loss, vf_loss, ent_loss, rew, val = [], [], [], [], []
                    if x[0] == 'pg_loss':
                        pg_loss.append(float(x[1]))
                    if x[0] == 'vf_loss':
                        vf_loss.append(float(x[1]))
                    if x[0] == 'ent_loss':
                        ent_loss.append(float(x[1]))
                    if x[0] == 'avg_rew':
                        rew.append(float(x[1]))
                    if x[0] == 'avg_val':
                        val.append(float(x[1]))

            if len(pg_loss) > 0:
                plt.plot(pg_loss)
                plt.title('policy gradient loss')
                plt.savefig('.'.join(p.split('.')[:-1]) + '_pg_loss.jpg')
                plt.cla()

            if len(vf_loss) > 0:
                plt.plot(vf_loss)
                plt.title('value function loss')
                plt.savefig('.'.join(p.split('.')[:-1]) + '_vf_loss.jpg')
                plt.cla()

            if len(ent_loss) > 0:
                plt.plot(ent_loss)
                plt.title('entropy loss')
                plt.savefig('.'.join(p.split('.')[:-1]) + '_ent_loss.jpg')
                plt.cla()

            if len(rew) > 0 and len(val) > 0:
                plt.plot(rew, label='reward')
                plt.plot(val, label='value')
                plt.title('reward and value')
                plt.savefig('.'.join(p.split('.')[:-1]) + '_rew_val.jpg')
                plt.cla()
