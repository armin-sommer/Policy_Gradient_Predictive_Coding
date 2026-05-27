"""Shared types and parameter-vector utilities.

Env, Policy, and Log are passed around between envs/, updates/, and runner.
flatten/unflatten convert a Flax params dict to/from a flat 1-D array; used
by TRPO/NPG's Fisher solve and by the runner's logging.
"""
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp


class Env(NamedTuple):
    OBS_DIM: int
    N_ACTIONS: int
    # sample_batch(params, key, *, logits_fn, batch_size) -> (x, a, r)
    sample_batch: Callable[..., Any]
    # J(params, *, logits_fn) -> scalar expected return.
    J: Callable[..., Any]


class Policy(NamedTuple):
    init: Callable[[jax.Array], dict]                       # rng_key -> params
    logits_fn: Callable[[dict], jnp.ndarray]                # params -> logits at fixed_obs
    apply: Callable[[dict, jnp.ndarray], jnp.ndarray]       # params, x -> logits (batched)
    fixed_obs: jnp.ndarray                                  # shape (obs_dim,)
    obs_dim: int
    hidden: int
    n_actions: int


class Log(NamedTuple):
    theta:  jnp.ndarray   # (T, D) flattened parameter vector per step
    pa1:    jnp.ndarray   # (T,)   probability of action 1 (logged for compatibility)
    dtheta: jnp.ndarray   # (T, D) unit update direction per step
    kl:     jnp.ndarray   # (T,)   realized KL(pi_t || pi_{t-1})
    dJ:     jnp.ndarray   # (T,)   J(theta_t) - J(theta_{t-1})


def flatten(params) -> jnp.ndarray:
    leaves, _ = jax.tree_util.tree_flatten(params)
    return jnp.concatenate([l.ravel() for l in leaves])


def unflatten(template, flat: jnp.ndarray):
    leaves, treedef = jax.tree_util.tree_flatten(template)
    out, idx = [], 0
    for l in leaves:
        sz = l.size
        out.append(flat[idx:idx + sz].reshape(l.shape))
        idx += sz
    return jax.tree_util.tree_unflatten(treedef, out)


def kl_categorical(p_old: jnp.ndarray, p_new: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(p_old * (jnp.log(p_old + 1e-12) - jnp.log(p_new + 1e-12)))
