"""Rollout type + the discounted-return utility.

Every rollout collector (scan_rollout, future procgen_rollout) returns a
Rollout with these five fields and shape conventions (T, N, ...). Algorithms
consume Rollouts; they never see the collector. discounted_returns is the
cross-collector utility for converting per-step rewards to Monte-Carlo
discounted returns.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Rollout(NamedTuple):
    obs:      jnp.ndarray   # (T, N, obs_dim)
    actions:  jnp.ndarray   # (T, N)
    rewards:  jnp.ndarray   # (T, N)
    dones:    jnp.ndarray   # (T, N)
    logp_old: jnp.ndarray   # (T, N) log-prob under params used to collect
    last_obs: jnp.ndarray   # (N, obs_dim) obs *after* the final step, for bootstrap


def discounted_returns(rewards: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """G_t = sum_{k>=t} gamma^(k-t) r_k, per env.

    rewards: (T, N) -> returns: (T, N). T=1 collapses to G_0 = r_0, so
    bandit numerics are unchanged. Within-rollout episode boundaries must
    be handled by the env's absorbing-terminal convention (rewards=0 after
    done).
    """
    def body(G_next, r_t):
        G_t = r_t + gamma * G_next
        return G_t, G_t

    _, G = jax.lax.scan(body, jnp.zeros(rewards.shape[-1]), rewards,
                        reverse=True)
    return G
