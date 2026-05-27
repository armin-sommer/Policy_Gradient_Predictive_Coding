"""Procgen rollout (host-Python).

Procgen-mirror is a stateful C++ env; we can't JIT through it. This
collector runs as ordinary Python: it calls the env's step in a loop,
calls policy.apply (JIT'd separately) on the host to sample actions,
and assembles a Rollout(T, N, ...) at the end whose shape matches
scan_rollout's output. Algorithms consume the Rollout and don't know
the difference.

Because this function is NOT jit-compatible, the existing run_seed in
src/runner.py (which jit-vmaps the whole training loop) cannot drive it.
A hosted runner — Python loop over iterations, JIT only the step_fn
update — is required. That wiring is left as a TODO for when CUDA is
available; see configs/experiments/procgen_coinrun.py for the intended
shape.
"""
import distrax
import jax
import jax.numpy as jnp
import numpy as np

from rollout._protocol import Rollout


def collect_rollout(params, key, *, env_handle, policy, T: int):
    """env_handle: ProcgenEnvHandle. policy: Policy with value_apply.

    Returns a Rollout where (T, N, ...) matches scan_rollout's contract.
    `dones` flips at the end of every Procgen episode (auto-reset is
    handled inside procgen-mirror).
    """
    vec_env = env_handle.vec_env
    N = env_handle.num_envs

    # Pre-allocate host-side buffers.
    obs_buf      = np.zeros((T, N) + env_handle.obs_shape, dtype=np.uint8)
    actions_buf  = np.zeros((T, N), dtype=np.int32)
    rewards_buf  = np.zeros((T, N), dtype=np.float32)
    dones_buf    = np.zeros((T, N), dtype=np.bool_)
    logp_buf     = np.zeros((T, N), dtype=np.float32)

    # Get current obs (procgen-mirror gym3 API).
    reward, obs, first = vec_env.observe()

    for t in range(T):
        obs_jax = jnp.asarray(obs)
        logits = policy.apply(params, obs_jax)
        # Per-step subkey for action sampling.
        key, sub = jax.random.split(key)
        dist = distrax.Categorical(logits=logits)
        actions = dist.sample(seed=sub)
        logp = dist.log_prob(actions)

        actions_np = np.asarray(actions)
        vec_env.act(actions_np)
        next_reward, next_obs, next_first = vec_env.observe()

        obs_buf[t]     = obs
        actions_buf[t] = actions_np
        rewards_buf[t] = next_reward
        dones_buf[t]   = next_first
        logp_buf[t]    = np.asarray(logp)

        obs = next_obs

    return Rollout(
        obs=jnp.asarray(obs_buf),
        actions=jnp.asarray(actions_buf),
        rewards=jnp.asarray(rewards_buf),
        dones=jnp.asarray(dones_buf),
        logp_old=jnp.asarray(logp_buf),
        last_obs=jnp.asarray(obs),
    )
