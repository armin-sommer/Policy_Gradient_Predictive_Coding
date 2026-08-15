"""Additive precision-energy inference for Gaussian policy outputs.

This is the direct policy analogue of the activity-space construction in
Innocenti et al. (2023).  The free variable is a policy-output activity
``z = (mu, log_sigma)`` initialised at the old policy output.  Its energy is a
linear policy-gradient drive plus ordinary precision-weighted squared errors.
Settling therefore supplies the inverse output-Fisher action without building a
Fisher matrix or inserting a precomputed natural target.

The settled output is subsequently clamped as the target for the existing JPC
hidden-activity inference and local weight update.  Keeping these two inference
stages separate avoids adding the network's identity output-prediction curvature
to the policy-output natural-gradient energy.
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from pc_algorithms.gaussian_policy import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    split_gaussian_params,
)


def effective_policy_outputs(params, action_dim, exp_std=True):
    """Outputs actually used by the policy, including log-std clipping."""
    loc, _, log_std = split_gaussian_params(
        params, action_dim, exp_std=exp_std)
    return jnp.concatenate([loc, log_std], axis=-1)


def gaussian_output_score(old_params, pre_tanh, advantages, action_dim,
                          target_scale=1.0, exp_std=True):
    """Return ``d(A log pi)/d(mu, log_sigma)`` at the old policy."""
    if not exp_std:
        raise ValueError("precision-energy policy inference requires exp_std=True")
    loc, scale, _ = split_gaussian_params(
        old_params, action_dim, exp_std=True)
    z_score = (pre_tanh - loc) / scale
    adv = advantages[:, None]
    score_mu = target_scale * adv * z_score / scale
    score_log_std = target_scale * adv * (jnp.square(z_score) - 1.0)
    return score_mu, score_log_std


def quadratic_policy_radius(old_params, outputs, action_dim, exp_std=True):
    """Mean local Gaussian KL metric, ``0.5 * delta.T F_out delta``."""
    if not exp_std:
        raise ValueError("precision-energy policy inference requires exp_std=True")
    old_loc, old_scale, old_log_std = split_gaussian_params(
        old_params, action_dim, exp_std=True)
    loc, log_std = jnp.split(outputs, 2, axis=-1)
    delta_mu = loc - old_loc
    delta_log_std = log_std - old_log_std
    per_sample = jnp.sum(
        0.5 * jnp.square(delta_mu / old_scale)
        + jnp.square(delta_log_std), axis=-1)
    return jnp.mean(per_sample)


def natural_output_displacement(old_params, pre_tanh, advantages, action_dim,
                                target_scale=1.0, exp_std=True):
    """Analytic unit-precision minimizer, used only to set the TR precision."""
    _, scale, _ = split_gaussian_params(
        old_params, action_dim, exp_std=exp_std)
    score_mu, score_log_std = gaussian_output_score(
        old_params, pre_tanh, advantages, action_dim,
        target_scale=target_scale, exp_std=exp_std)
    return jnp.concatenate(
        [jnp.square(scale) * score_mu, 0.5 * score_log_std], axis=-1)


def trust_region_precision(old_params, pre_tanh, advantages, action_dim,
                           max_radius, target_scale=1.0, exp_std=True,
                           min_precision=1e-6):
    """KKT precision whose box-constrained minimizer lies on ``max_radius``."""
    if max_radius <= 0:
        raise ValueError("max_radius must be positive")
    unit_delta = natural_output_displacement(
        old_params, pre_tanh, advantages, action_dim,
        target_scale=target_scale, exp_std=exp_std)
    old_outputs = effective_policy_outputs(
        old_params, action_dim, exp_std=exp_std)
    old_loc, old_log_std = jnp.split(old_outputs, 2, axis=-1)
    unit_mu, unit_log_std = jnp.split(unit_delta, 2, axis=-1)

    def radius_at(beta):
        target = jnp.concatenate([
            old_loc + unit_mu / beta,
            jnp.clip(old_log_std + unit_log_std / beta,
                     LOG_STD_MIN, LOG_STD_MAX),
        ], axis=-1)
        return quadratic_policy_radius(
            old_outputs, target, action_dim, exp_std=exp_std)

    dtype = old_params.dtype
    low = jnp.asarray(min_precision, dtype=dtype)
    high = jnp.asarray(1.0, dtype=dtype)

    # First bracket a feasible precision, then bisect.  Accounting for the
    # production log-std box here makes the requested radius exact even when an
    # old output is already on a bound and its natural direction points outward.
    def expand(_, bounds):
        lo, hi = bounds
        hi = jnp.where(radius_at(hi) > max_radius, 2.0 * hi, hi)
        return lo, hi

    low, high = jax.lax.fori_loop(0, 40, expand, (low, high))

    def bisect(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        too_large = radius_at(mid) > max_radius
        return jnp.where(too_large, mid, lo), jnp.where(too_large, hi, mid)

    low, high = jax.lax.fori_loop(0, 50, bisect, (low, high))
    return jnp.where(radius_at(low) <= max_radius, low, high)


def precision_policy_energy(outputs, old_params, pre_tanh, advantages, *,
                            action_dim, beta, target_scale=1.0,
                            exp_std=True):
    """Linear policy drive plus additive Gaussian precision errors."""
    old_params = jax.lax.stop_gradient(old_params)
    old_loc, old_scale, old_log_std = split_gaussian_params(
        old_params, action_dim, exp_std=exp_std)
    loc, log_std = jnp.split(outputs, 2, axis=-1)
    score_mu, score_log_std = gaussian_output_score(
        old_params, pre_tanh, advantages, action_dim,
        target_scale=target_scale, exp_std=exp_std)

    delta_mu = loc - old_loc
    delta_log_std = log_std - old_log_std
    drive = -score_mu * delta_mu - score_log_std * delta_log_std
    precision_error = beta * (
        0.5 * jnp.square(delta_mu / old_scale)
        + jnp.square(delta_log_std))
    return jnp.mean(jnp.sum(drive + precision_error, axis=-1))


@eqx.filter_jit
def settle_precision_outputs(old_params, pre_tanh, advantages, *, action_dim,
                             beta, max_t1=20.0, target_scale=1.0,
                             exp_std=True, dt=None, rtol=1e-5, atol=1e-6,
                             max_steps=1_000_000):
    """Settle free output activities and report fixed-point diagnostics."""
    old_params = jax.lax.stop_gradient(effective_policy_outputs(
        old_params, action_dim, exp_std=exp_std))

    def energy(outputs):
        return precision_policy_energy(
            outputs, old_params, pre_tanh, advantages,
            action_dim=action_dim, beta=beta, target_scale=target_scale,
            exp_std=exp_std)

    grad_output = jax.grad(energy)
    rate = float(old_params.shape[0])

    def vector_field(t, outputs, args):
        return -rate * grad_output(outputs)

    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(vector_field),
        solver=diffrax.Heun(),
        t0=0.0,
        t1=float(max_t1),
        dt0=dt,
        y0=old_params,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=max_steps,
    )
    outputs = solution.ys[0]
    loc, log_std = jnp.split(outputs, 2, axis=-1)
    outputs = jnp.concatenate(
        [loc, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)], axis=-1)

    residual = rate * grad_output(outputs)
    residual_mu, residual_log_std = jnp.split(residual, 2, axis=-1)
    _, log_std = jnp.split(outputs, 2, axis=-1)
    blocked_low = ((log_std <= LOG_STD_MIN + 1e-6)
                   & (residual_log_std > 0.0))
    blocked_high = ((log_std >= LOG_STD_MAX - 1e-6)
                    & (residual_log_std < 0.0))
    projected_log_std = jnp.where(
        blocked_low | blocked_high, 0.0, residual_log_std)
    projected_residual = jnp.concatenate(
        [residual_mu, projected_log_std], axis=-1)
    residual_rms = jnp.sqrt(jnp.mean(jnp.square(projected_residual)))
    radius = quadratic_policy_radius(
        old_params, outputs, action_dim, exp_std=exp_std)
    return {
        "targets": outputs,
        "loss": energy(outputs),
        "radius": radius,
        "residual_rms": residual_rms,
        "beta": beta,
    }
