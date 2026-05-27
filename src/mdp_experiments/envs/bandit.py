"""Mei et al. 2020 one-state two-action MDP.

Single constant observation x = [1.0], two actions with deterministic
rewards r(a=0)=0, r(a=1)=1. J(theta) = pi_theta(a=1).

The whole world is two arrays. Kept as module-level constants so other
modules can import them when they need the same fixed observation (e.g.,
the PCPG step calls into pc_algorithms.pcpg with the same x_batched).
"""
import distrax
import jax
import jax.numpy as jnp

from mdp_experiments._base import Env


OBS_DIM = 1
N_ACTIONS = 2
OBS = jnp.array([1.0])
REWARDS = jnp.array([0.0, 1.0])


def sample_batch(params, key, *, logits_fn, batch_size):
    logits = logits_fn(params)
    dist = distrax.Categorical(logits=logits)
    actions = dist.sample(seed=key, sample_shape=(batch_size,))
    rewards = REWARDS[actions]
    x = jnp.broadcast_to(OBS, (batch_size, OBS_DIM))
    return x, actions, rewards


def J(params, *, logits_fn):
    p = jax.nn.softmax(logits_fn(params))
    return jnp.dot(p, REWARDS)


ENV = Env(OBS_DIM=OBS_DIM, N_ACTIONS=N_ACTIONS,
          sample_batch=sample_batch, J=J)
