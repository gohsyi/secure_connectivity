from common.argparser import args

from env import build_env

from baselines.a2c.a2c import learn


if __name__ == '__main__':
    env = build_env(n_vertices=args.n_vertices,
                    n_edges=args.n_edges,
                    n_actions=args.n_actions)

    d_model, a_model = learn(
        env=env,
        seed=None,
        nsteps=args.batch_size,
        total_epoches=int(80e6),
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=7e-4,
        gamma=0.99,
        log_interval=100,
        load_paths=None
    )
