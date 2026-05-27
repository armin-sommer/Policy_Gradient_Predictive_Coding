"""Generalized Advantage Estimation, ported from src/backprop_algorithms/common.py.

Inputs follow our Rollout convention (T, N) with `dones` flagging episode
boundaries. We don't track truncation separately; all dones are treated as
terminations. For envs with time-limit truncation that should not bootstrap
(rare for current scope), extend Rollout with a truncation flag and revisit.
"""
import jax
import jax.numpy as jnp


def compute_gae(rewards: jnp.ndarray,
                values: jnp.ndarray,
                dones: jnp.ndarray,
                bootstrap_value: jnp.ndarray,
                gamma: float = 0.99,
                gae_lambda: float = 0.95):
    """rewards, values, dones: (T, N). bootstrap_value: (N,).

    Returns (vs, advantages), each shape (T, N). vs are the GAE-targets for
    the value head; advantages are the GAE-estimated advantages.
    """
    termination = dones.astype(rewards.dtype)
    values_t_plus_1 = jnp.concatenate(
        [values[1:], bootstrap_value[None]], axis=0)
    deltas = rewards + gamma * (1.0 - termination) * values_t_plus_1 - values

    def body(acc, target_t):
        delta, term = target_t
        acc = delta + gamma * (1.0 - term) * gae_lambda * acc
        return acc, acc

    _, gae = jax.lax.scan(body, jnp.zeros_like(bootstrap_value),
                          (deltas, termination), reverse=True)
    vs = gae + values
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(gae)
