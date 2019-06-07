import os
import argparse

from common.plot import SmoothPlot


parser = argparse.ArgumentParser()
parser.add_argument('-smooth_rate', type=float, default=0.9)
parser.add_argument('-line_width', type=float, default=0.5)
args = parser.parse_args()

plt = SmoothPlot(args.smooth_rate, args.line_width)


for root, dirs, files in os.walk('logs'):
    for f in files:
        if f[0] != '.' and os.path.splitext(f)[-1] == '.log':  # process .log
            p = os.path.join(root, f)
            print('processing %s' % p)
            pg_loss, vf_loss, ent_loss, rew, val, bl_rew = [], [], [], [], [], []

            for line in open(p):
                line = line.split()
                for x in line:
                    x = x.split(':')
                    if x[0] == 'ep' and x[1] == '0':
                        pg_loss, vf_loss, ent_loss, rew, val, bl_rew = [], [], [], [], [], []
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
                    if x[0] == 'bl_avg_rew':
                        bl_rew.append(float(x[1]))

            if len(pg_loss) > 0:
                plt.plot(
                    pg_loss,
                    title='policy gradient loss',
                    save_path='.'.join(p.split('.')[:-1]) + '_pg_loss.jpg',
                )

            if len(vf_loss) > 0:
                plt.plot(
                    vf_loss,
                    title='value function loss',
                    save_path='.'.join(p.split('.')[:-1]) + '_vf_loss.jpg',
                )

            if len(ent_loss) > 0:
                plt.plot(
                    ent_loss,
                    title='entropy loss',
                    save_path='.'.join(p.split('.')[:-1]) + '_ent_loss.jpg',
                )

            if len(rew) > 0:
                plt.plot(rew,
                    title='reward',
                    save_path='.'.join(p.split('.')[:-1]) + '_rew.jpg',
                )

            if len(val) > 0:
                plt.plot(
                    val,
                    title='value',
                    save_path='.'.join(p.split('.')[:-1]) + '_val.jpg',
                )

            if len(rew) > 0 and len(val) > 0:
                plt.plot(
                    [val, rew],
                    label=['value', 'reward'],
                    title='performance',
                    save_path='.'.join(p.split('.')[:-1]) + '_rew_val.jpg',
                )

            if len(bl_rew) > 0:
                plt.plot(
                    [rew, bl_rew],
                    label=['reward', 'baseline'],
                    title='reward',
                    save_path='.'.join(p.split('.')[:-1]) + '_rew_bl.jpg',
                )
