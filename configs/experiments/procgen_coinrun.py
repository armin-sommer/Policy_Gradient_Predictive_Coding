"""Procgen coinrun config. Full PPO (Pipeline B, behind the seam) on a
NatureCNN actor-critic policy.

Status: pieces are in place (algorithms/procgen/ppo.py, policies/nature_cnn.py,
envs/procgen.py, rollout/procgen_rollout.py) but the *driver wiring* for
Procgen is not yet in scripts/run_experiment.py. Procgen rollouts are
host-Python (not JIT-compatible), so the existing jit-vmap runner cannot
drive them. A `run_hosted` variant is required:

    env_handle = make_procgen_env(env_name="coinrun", num_envs=N, ...)
    policy = make_policy("nature_cnn", env_handle)
    rollout_fn = partial(procgen_rollout.collect_rollout,
                         env_handle=env_handle, policy=policy, T=T)
    init_opt_state_fn, step_fn = full.ppo.make_step(
        rollout_fn, policy, **HP)
    step_jit = jax.jit(step_fn)
    params = policy.init(jax.random.PRNGKey(SEED))
    opt_state = init_opt_state_fn(params)
    for it in range(ITERS):
        key = jax.random.fold_in(base_key, it)
        params, opt_state, metrics = step_jit(params, opt_state, key)

Wiring this up + CUDA verification is deferred.
"""
from algorithms.procgen import ppo, reinforce, trpo


NAME    = "procgen_coinrun"
ENV     = "procgen_coinrun"        # signals make_procgen_env, not make_env
POLICY  = "nature_cnn"
SEEDS   = 1
ITERS   = 1500
T       = 256
N       = 64
HIDDEN  = 512                      # NatureCNN's fixed head size
SEED    = 42
REF     = "PPO"

UPDATES = {
    "PPO":       (ppo,       dict(lr=5e-4, clip=0.2, epochs=3,
                                  num_minibatches=8, vf_coef=0.5,
                                  ent_coef=0.01, gamma=0.999,
                                  gae_lambda=0.95, max_grad_norm=0.5)),
    # Uncomment when scaling experiments:
    # "REINFORCE": (reinforce, dict(lr=3e-4, gamma=0.99, ent_coef=1e-4)),
    # "TRPO":      (trpo,      dict(target_kl=0.01, value_lr=3e-4)),
}
