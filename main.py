from common.argparser import args

from env import build_env

from baselines.a2c.a2c import learn


if __name__ == '__main__':
    env = build_env(
        n_vertices=args.n_vertices,
        n_edges=args.n_edges,
        n_actions=args.n_actions
    )

    d_model, a_model = learn(
        env=env,
        defender=args.d_model,
        attacker=args.a_model,
        seed=args.seed,
        nsteps=args.batchsize,
        total_epoches=args.total_epoches,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        gamma=args.gamma,
        log_interval=args.log_interval,
        load_paths=args.load_paths,
    )


