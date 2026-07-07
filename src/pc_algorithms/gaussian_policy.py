"""Gaussian policy helpers for continuous-control PCPG.

The PC target rule mirrors the discrete softmax case:
  discrete:  target = logits + A * (one_hot(a) - pi)
  Gaussian:  target_mu = mu + A * (z - mu) / sigma^2
             target_log_std = log_std + A * ((z - mu) / sigma)^2 - 1)

where z is the pre-tanh Gaussian sample and A is the advantage.
Actions sent to Brax are tanh(z), matching the backprop SOTA stack.
"""

import jax
import jax.numpy as jnp
import jax.random as jr


def split_gaussian_params(params, action_dim, exp_std=True, min_std=0.001):
    loc, log_scale = jnp.split(params, 2, axis=-1)
    if exp_std:
        scale = jnp.exp(log_scale)
    else:
        scale = jax.nn.softplus(log_scale) + min_std
    return loc, scale, log_scale


def sample_gaussian_action(key, params, action_dim, exp_std=True, min_std=0.001):
    loc, scale, _ = split_gaussian_params(params, action_dim, exp_std, min_std)
    key, k = jr.split(key)
    pre_tanh = loc + scale * jr.normal(k, loc.shape)
    return jnp.tanh(pre_tanh), pre_tanh


def gaussian_pc_targets(
    params,
    pre_tanh,
    advantages,
    action_dim,
    target_scale,
    exp_std=True,
    min_std=0.001,
):
    loc, scale, log_scale = split_gaussian_params(
        params, action_dim, exp_std, min_std)
    z_score = (pre_tanh - loc) / scale
    adv = advantages[:, None]
    loc_target = loc + target_scale * adv * z_score / scale
    log_scale_target = log_scale + target_scale * adv * (jnp.square(z_score) - 1.0)
    return jnp.concatenate([loc_target, log_scale_target], axis=-1)


def discrete_pc_targets(logits, actions, advantages, action_size, target_scale):
    pi = jax.nn.softmax(logits)
    onehot = jax.nn.one_hot(actions, action_size)
    return logits + target_scale * advantages[:, None] * (onehot - pi)


def deterministic_gaussian_action(params, action_dim, exp_std=True, min_std=0.001):
    loc, _, _ = split_gaussian_params(params, action_dim, exp_std, min_std)
    return jnp.tanh(loc)
