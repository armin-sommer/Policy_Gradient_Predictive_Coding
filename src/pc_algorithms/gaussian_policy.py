"""Gaussian policy helpers for continuous-control PCPG.

The Gaussian PC targets are the *natural-gradient* step in distribution space:
the raw score is premultiplied by the inverse Fisher, which is exact and
diagonal for a diagonal Gaussian in (mu, log_std) coordinates:
  F = diag(1/sigma^2, 2)   =>
  target_mu      = mu + A * (z - mu)                     [sigma^2 * score]
  target_log_std = log_std + (A / 2) * (z_score^2 - 1)   [score / 2]

where z is the pre-tanh Gaussian sample, z_score = (z - mu) / sigma, and A is
the advantage. The tanh squashing leaves the Fisher unchanged (invertible
transform of the sample), so this is exact for the tanh-Gaussian too.

The Euclidean score A*(z-mu)/sigma^2 amplifies mu-targets by 1/sigma^2 (x55 at
the sigma clamp floor), which diagnostics showed was the collapse trigger:
rare advantage outliers produced targets 50-100 sigma away and one-update
policy jumps of ~18 sigma. The natural target bounds offsets at |A|*sigma; a
+-MU_OFFSET_CLIP_SIGMA*sigma clip backstops the remaining advantage tail.

Actions sent to Brax are tanh(z), matching the backprop SOTA stack.
log_std is clamped to [LOG_STD_MIN, LOG_STD_MAX] when sampling and in targets
(sigma in ~[0.14, 1.65]).

The discrete rule mirrors the softmax case: target = logits + A*(one_hot - pi).
"""

import jax
import jax.numpy as jnp
import jax.random as jr

LOG_STD_MIN = -2.0
LOG_STD_MAX = 0.5
MU_OFFSET_CLIP_SIGMA = 3.0


def split_gaussian_params(params, action_dim, exp_std=True, min_std=0.001):
    loc, log_scale = jnp.split(params, 2, axis=-1)
    if exp_std:
        log_scale = jnp.clip(log_scale, LOG_STD_MIN, LOG_STD_MAX)
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
    # natural gradient: sigma^2 * (A * (z - mu) / sigma^2) = A * (z - mu)
    loc_offset = jnp.clip(
        target_scale * adv * z_score,
        -MU_OFFSET_CLIP_SIGMA, MU_OFFSET_CLIP_SIGMA) * scale
    loc_target = loc + loc_offset
    # natural gradient: (1/2) * (A * (z_score^2 - 1))
    log_scale_target = jnp.clip(
        log_scale + 0.5 * target_scale * adv * (jnp.square(z_score) - 1.0),
        LOG_STD_MIN, LOG_STD_MAX)
    return jnp.concatenate([loc_target, log_scale_target], axis=-1)


def discrete_pc_targets(logits, actions, advantages, action_size, target_scale):
    pi = jax.nn.softmax(logits)
    onehot = jax.nn.one_hot(actions, action_size)
    return logits + target_scale * advantages[:, None] * (onehot - pi)


def deterministic_gaussian_action(params, action_dim, exp_std=True, min_std=0.001):
    loc, _, _ = split_gaussian_params(params, action_dim, exp_std, min_std)
    return jnp.tanh(loc)
