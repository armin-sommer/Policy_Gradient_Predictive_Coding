"""Gaussian policy helpers for continuous-control PCPG.

The PC target rule mirrors the discrete softmax case:
  discrete:  target = logits + A * (one_hot(a) - pi)
  Gaussian:  target_mu = mu + A * (z - mu) / sigma^2
             target_log_std = log_std + A * (((z - mu) / sigma)^2 - 1)

where z is the pre-tanh Gaussian sample and A is the advantage.
Actions sent to Brax are tanh(z), matching the backprop SOTA stack.

log_std is clamped to [LOG_STD_MIN, LOG_STD_MAX] both when sampling and in the
targets: the mu-target amplifies by 1/sigma^2, so an unbounded shrinking sigma
makes targets explode, while a growing sigma washes out the policy. Clamping
bounds both failure modes (sigma in ~[0.14, 1.65]).
"""

import jax
import jax.numpy as jnp
import jax.random as jr

LOG_STD_MIN = -2.0
LOG_STD_MAX = 0.5


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
    target_clip=None,
    target_clip_rel=False,
):
    """PC targets for the squashed-Gaussian policy.

    `target_clip` bounds the *mean-target offset* `ts*A*(z-mu)/sigma^2` per
    coordinate -- an output-space trust region that caps the 1/sigma^2 blow-up
    directly, and (unlike a weight-gradient clip) is optimizer-agnostic.
      - target_clip_rel=False: absolute cap, |loc_target - mu| <= target_clip.
      - target_clip_rel=True:  relative cap, |loc_target - mu| <= target_clip*sigma.
    """
    loc, scale, log_scale = split_gaussian_params(
        params, action_dim, exp_std, min_std)
    z_score = (pre_tanh - loc) / scale
    adv = advantages[:, None]
    mu_offset = target_scale * adv * z_score / scale
    if target_clip is not None:
        cap = target_clip * scale if target_clip_rel else target_clip
        mu_offset = jnp.clip(mu_offset, -cap, cap)
    loc_target = loc + mu_offset
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
