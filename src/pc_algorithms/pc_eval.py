"""Vectorized evaluation for jpc policy networks."""

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from pc_algorithms.gaussian_policy import (
    deterministic_gaussian_action,
    sample_gaussian_action,
)


def evaluate_discrete_policy(
    eval_env,
    policy_forward,
    model,
    flatten_obs,
    eval_key,
    max_steps,
):
    state = eval_env.reset()
    n = eval_env.num_envs
    ep_ret = np.zeros(n, dtype=np.float64)
    ep_len = np.zeros(n, dtype=np.int64)
    done_mask = np.zeros(n, dtype=bool)
    for _ in range(int(max_steps)):
        obs = flatten_obs(state.obs, update=False)
        logits = policy_forward(model, jnp.asarray(obs))
        eval_key, k = jr.split(eval_key)
        actions = jr.categorical(k, logits)
        state = eval_env.step(np.asarray(actions))
        live = ~done_mask
        ep_ret[live] += np.asarray(state.reward)[live]
        ep_len[live] += 1
        done_mask |= np.asarray(state.done).astype(bool)
        if done_mask.all():
            break
    return ep_ret.tolist(), ep_len.tolist(), eval_key


def evaluate_gaussian_policy(
    eval_env,
    policy_forward,
    model,
    action_dim,
    flatten_obs,
    eval_key,
    max_steps,
    exp_std=True,
    deterministic=True,
):
    state = eval_env.reset()
    n = eval_env.num_envs
    ep_ret = np.zeros(n, dtype=np.float64)
    ep_len = np.zeros(n, dtype=np.int64)
    done_mask = np.zeros(n, dtype=bool)
    for _ in range(int(max_steps)):
        obs = flatten_obs(state.obs, update=False)
        params = policy_forward(model, jnp.asarray(obs))
        if deterministic:
            actions = deterministic_gaussian_action(params, action_dim, exp_std=exp_std)
        else:
            eval_key, k = jr.split(eval_key)
            actions, _ = sample_gaussian_action(k, params, action_dim, exp_std=exp_std)
        state = eval_env.step(np.asarray(actions))
        live = ~done_mask
        ep_ret[live] += np.asarray(state.reward)[live]
        ep_len[live] += 1
        done_mask |= np.asarray(state.done).astype(bool)
        if done_mask.all():
            break
    return ep_ret.tolist(), ep_len.tolist(), eval_key
