"""Mei et al. 2020 one-state two-action MDP, as a reset/step env.

Single constant observation x = [1.0], two actions with deterministic
rewards r(a=0)=0, r(a=1)=1. Episode terminates after one step. Closed-form
J(theta) = pi_theta(a=1) is exposed so NPG (oracle) and TRPO's line search
keep working.
"""
import jax
import jax.numpy as jnp

from envs._protocol import Env


OBS_DIM = 1
N_ACTIONS = 2
OBS = jnp.array([1.0])
REWARDS = jnp.array([0.0, 1.0])


def reset(key):
    del key
    return jnp.int32(0), OBS


def step(state, action, key):
    del key
    reward = REWARDS[action]
    next_obs = OBS
    done = jnp.bool_(True)
    return state, next_obs, reward, done


def J(params, *, apply):
    logits = apply(params, OBS[None])[0]
    p = jax.nn.softmax(logits)
    return jnp.dot(p, REWARDS)


ENV = Env(obs_dim=OBS_DIM, n_actions=N_ACTIONS,
          reset=reset, step=step, J=J)
