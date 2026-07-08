"""Gaussian policy helpers for continuous-control PCPG.

The Gaussian PC targets use the Euclidean score with a trust-region clip on
the mu offset (in sigma units):
  target_mu      = mu + clip(A * z_score / sigma, +-MU_OFFSET_CLIP_SIGMA) * sigma
  target_log_std = log_std + A * (z_score^2 - 1)

where z is the pre-tanh Gaussian sample, z_score = (z - mu) / sigma, and A is
the advantage.

Why this form: diagnostics showed the raw Euclidean score A*(z-mu)/sigma^2
learns well (typical mu-targets 1-5 sigma) but its 1/sigma^2 amplification
(x55 at the sigma clamp floor) turns rare advantage outliers into 50-100 sigma
targets and ~18-sigma one-update policy jumps -> irreversible collapse. The
exact natural-gradient target (sigma^2 * score, Fisher-preconditioned) fixes
the tail but shrinks typical updates by sigma^2 (~5-50x) and stalls learning
(~150 return vs 985). The clip cuts only the destructive tail while leaving
typical updates identical to the rule that learns.

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
MU_OFFSET_CLIP_SIGMA = 6.0


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
    # Euclidean score A*z_score/sigma, trust-region clipped in sigma units
    loc_offset = jnp.clip(
        target_scale * adv * z_score / scale,
        -MU_OFFSET_CLIP_SIGMA, MU_OFFSET_CLIP_SIGMA) * scale
    loc_target = loc + loc_offset
    log_scale_target = jnp.clip(
        log_scale + target_scale * adv * (jnp.square(z_score) - 1.0),
        LOG_STD_MIN, LOG_STD_MAX)
    return jnp.concatenate([loc_target, log_scale_target], axis=-1)


def discrete_pc_targets(logits, actions, advantages, action_size, target_scale):
    pi = jax.nn.softmax(logits)
    onehot = jax.nn.one_hot(actions, action_size)
    return logits + target_scale * advantages[:, None] * (onehot - pi)


def deterministic_gaussian_action(params, action_dim, exp_std=True, min_std=0.001):
    loc, _, _ = split_gaussian_params(params, action_dim, exp_std, min_std)
    return jnp.tanh(loc)
