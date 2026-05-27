"""Param-vector + KL utilities shared by the Fisher solve, the runner, and
any algorithm that needs to step in flat parameter space.
"""
import jax
import jax.numpy as jnp


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
