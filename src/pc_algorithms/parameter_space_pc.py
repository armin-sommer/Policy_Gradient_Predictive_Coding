"""Parameter-space predictive-coding policy updates.

The free PC variable is a shared parameter displacement ``delta_theta``.  Local
linearised output errors define the quadratic energy

    E(delta) = 0.5 delta.T (F_theta + damping I) delta - g.T delta.

``F_theta`` is never materialised.  A JVP propagates a parameter perturbation to
policy-output perturbations, the analytic Gaussian output precision acts on the
local errors, and a VJP returns their shared parameter-space effect.  Conjugate
gradient minimises this PC energy substantially faster than first-order gradient
flow while preserving the same fixed point.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from pc_algorithms import gaussian_policy as gpol
from pc_algorithms.gaussian_policy import (
    split_gaussian_params,
)
from pc_algorithms.error_neuron_graph import (
    joint_error_neuron_direction,
    local_categorical_parameter_pc_direction,
    local_parameter_pc_direction,
)


def _forward(model, obs):
    activity = obs
    for layer in model:
        activity = jax.vmap(layer)(activity)
    return activity


def _effective_gaussian_outputs(network_outputs, action_dim, policy_log_std):
    if policy_log_std is None:
        return network_outputs
    mean = network_outputs[..., :action_dim]
    return jnp.concatenate(
        [mean, jnp.broadcast_to(policy_log_std, mean.shape)], axis=-1)


def _unpack_solve_parameters(solve, flat_parameters):
    parameters = solve["unravel"](flat_parameters)
    if solve.get("state_independent_std", False):
        return parameters
    return parameters, None


def _log_prob(params, pre_tanh, action_dim):
    loc, scale, log_scale = split_gaussian_params(
        params, action_dim, exp_std=True)
    return jnp.sum(
        -0.5 * jnp.square((pre_tanh - loc) / scale) - log_scale,
        axis=-1)


def _gaussian_kl(old_params, new_params, action_dim):
    old_loc, old_scale, _ = split_gaussian_params(
        old_params, action_dim, exp_std=True)
    new_loc, new_scale, _ = split_gaussian_params(
        new_params, action_dim, exp_std=True)
    return jnp.mean(jnp.sum(
        jnp.log(new_scale / old_scale)
        + (jnp.square(old_scale) + jnp.square(old_loc - new_loc))
        / (2.0 * jnp.square(new_scale))
        - 0.5,
        axis=-1))


def _categorical_kl(old_logits, new_logits):
    old_log_probs = jax.nn.log_softmax(old_logits, axis=-1)
    new_log_probs = jax.nn.log_softmax(new_logits, axis=-1)
    return jnp.mean(jnp.sum(
        jnp.exp(old_log_probs) * (old_log_probs - new_log_probs), axis=-1))


def _settle_exact_kl_scale(solve, obs, action_dim, *, max_kl,
                           inference_steps, primal_rate, dual_rate,
                           augmented_penalty, inference_tol):
    """Jointly settle scalar step and dual activities on the exact policy KL.

    The activities are dimensionless: ``x`` multiplies the local Fisher-boundary
    scale and ``b`` is the normalized nonnegative KL dual. This normalization
    keeps the inference rates meaningful as gradient and Fisher scales change.
    """
    direction = solve["direction"]
    local_scale = jnp.sqrt(
        max_kl / (solve["quadratic_radius_unscaled"] + 1e-30))
    objective_slope = jnp.dot(solve["gradient"], direction)

    def exact_kl_at_x(x):
        arrays, policy_log_std = _unpack_solve_parameters(
            solve,
            solve["flat_parameters"] + x * local_scale * direction)
        network_outputs = _forward(
            eqx.combine(arrays, solve["static"]), obs)
        outputs = _effective_gaussian_outputs(
            network_outputs, action_dim, policy_log_std)
        return _gaussian_kl(solve["old_outputs"], outputs, action_dim)

    def normalized_kl(x):
        return exact_kl_at_x(x) / max_kl

    value_and_slope = jax.value_and_grad(normalized_kl)

    def dynamics(x, dual):
        normalized_value, normalized_slope = value_and_slope(x)
        constraint = normalized_value - 1.0
        primal_velocity = (
            1.0
            - (dual + augmented_penalty * constraint) * normalized_slope)
        residual = jnp.maximum(
            jnp.abs(constraint), jnp.abs(primal_velocity))
        return normalized_value, normalized_slope, primal_velocity, residual

    x0 = jnp.asarray(1.0, dtype=direction.dtype)
    normalized_value0, normalized_slope0 = value_and_slope(x0)
    dual0 = jnp.where(
        normalized_slope0 > 1e-12,
        1.0 / normalized_slope0,
        1.0,
    )
    _, _, primal_velocity0, residual0 = dynamics(x0, dual0)

    def cond_fn(carry):
        step, _, _, _, residual = carry
        return (step < inference_steps) & (residual > inference_tol)

    def body_fn(carry):
        step, x, dual, _, _ = carry
        normalized_value, normalized_slope, _, _ = dynamics(x, dual)
        constraint = normalized_value - 1.0
        # Alternating projected primal-dual descent/ascent adds numerical
        # damping without changing the saddle-point equations.
        next_dual = jnp.maximum(
            0.0,
            dual + dual_rate * jnp.clip(constraint, -2.0, 2.0),
        )
        primal_velocity = (
            1.0
            - (next_dual + augmented_penalty * constraint)
            * normalized_slope)
        next_x = jnp.clip(
            x + primal_rate * jnp.clip(primal_velocity, -2.0, 2.0),
            0.0,
            4.0,
        )
        next_value, _, next_velocity, next_residual = dynamics(
            next_x, next_dual)
        return step + 1, next_x, next_dual, next_value, next_residual

    initial = (
        jnp.asarray(0, dtype=jnp.int32),
        x0,
        dual0,
        normalized_value0,
        residual0,
    )
    steps, x, normalized_dual, normalized_value, residual = (
        jax.lax.while_loop(cond_fn, body_fn, initial))
    actual_kl, actual_kl_slope_x = value_and_slope(x)
    actual_kl = actual_kl * max_kl
    actual_kl_slope = actual_kl_slope_x * max_kl / (local_scale + 1e-30)
    applied_scale = x * local_scale
    physical_dual = (
        normalized_dual * objective_slope * local_scale / (max_kl + 1e-30))
    scalar_kkt = jnp.abs(
        objective_slope - physical_dual * actual_kl_slope
    ) / (jnp.abs(objective_slope) + 1e-30)
    return {
        "applied_scale": applied_scale,
        "precision_beta": physical_dual,
        "normalized_dual": normalized_dual,
        "actual_kl": actual_kl,
        "constraint": normalized_value - 1.0,
        "inference_residual": residual,
        "scalar_kkt": scalar_kkt,
        "inference_steps": steps,
        "local_scale": local_scale,
    }


def parameter_pc_direction(model, obs, pre_tanh, advantages, action_dim, *,
                           damping=0.1, cg_iters=50, cg_tol=1e-8,
                           policy_log_std=None):
    """Infer ``(F_theta + damping I)^-1 g`` without forming the Fisher."""
    arrays, static = eqx.partition(model, eqx.is_array)
    state_independent_std = policy_log_std is not None
    parameter_state = (
        (arrays, policy_log_std) if state_independent_std else arrays)
    flat0, unravel = ravel_pytree(parameter_state)
    batch_size = obs.shape[0]

    def outputs(parameters):
        if state_independent_std:
            a, log_std = parameters
        else:
            a, log_std = parameters, None
        network_outputs = _forward(eqx.combine(a, static), obs)
        return _effective_gaussian_outputs(
            network_outputs, action_dim, log_std)

    old_outputs, output_vjp = jax.vjp(outputs, parameter_state)
    old_log_prob = _log_prob(old_outputs, pre_tanh, action_dim)

    def objective(a):
        out = outputs(a)
        return jnp.mean(advantages * _log_prob(out, pre_tanh, action_dim))

    objective_value, gradient_tree = jax.value_and_grad(objective)(
        parameter_state)
    gradient, _ = ravel_pytree(gradient_tree)

    _, scale, _ = split_gaussian_params(
        old_outputs, action_dim, exp_std=True)
    raw_log_std = old_outputs[..., action_dim:]
    in_bounds = (
        jnp.ones_like(raw_log_std) if state_independent_std else
        ((raw_log_std > gpol.LOG_STD_MIN)
         & (raw_log_std < gpol.LOG_STD_MAX)).astype(old_outputs.dtype))
    output_fisher_diag = jnp.concatenate(
        [1.0 / jnp.square(scale), 2.0 * in_bounds], axis=-1)

    def fisher_vector_product(vector):
        tangent = jax.jvp(
            outputs, (parameter_state,), (unravel(vector),))[1]
        weighted = output_fisher_diag * tangent / batch_size
        return ravel_pytree(output_vjp(weighted)[0])[0]

    def damped_fisher(vector):
        return fisher_vector_product(vector) + damping * vector

    direction, _ = jax.scipy.sparse.linalg.cg(
        damped_fisher, gradient, tol=cg_tol, maxiter=cg_iters)
    fisher_direction = fisher_vector_product(direction)
    kkt = damped_fisher(direction) - gradient
    relative_kkt = (jnp.linalg.norm(kkt)
                    / (jnp.linalg.norm(gradient) + 1e-30))
    fisher_equation_cosine = (
        jnp.dot(damped_fisher(direction), gradient)
        / (jnp.linalg.norm(damped_fisher(direction))
           * jnp.linalg.norm(gradient) + 1e-30))
    quadratic_radius_unscaled = 0.5 * jnp.dot(direction, fisher_direction)
    energy = (0.5 * jnp.dot(direction, damped_fisher(direction))
              - jnp.dot(gradient, direction))

    return {
        "arrays": arrays,
        "static": static,
        "flat_parameters": flat0,
        "unravel": unravel,
        "direction": direction,
        "gradient": gradient,
        "gradient_norm": jnp.linalg.norm(gradient),
        "direction_norm": jnp.linalg.norm(direction),
        "fisher_direction": fisher_direction,
        "quadratic_radius_unscaled": quadratic_radius_unscaled,
        "relative_kkt": relative_kkt,
        "fisher_equation_cosine": fisher_equation_cosine,
        "energy": energy,
        "old_outputs": old_outputs,
        "old_log_prob": old_log_prob,
        "objective": objective_value,
        "state_independent_std": state_independent_std,
    }


def make_parameter_pc_policy_step(model, obs, pre_tanh, advantages, action_dim,
                                  *, mode="parameter_npg", damping=0.1,
                                  step_size=0.05, max_radius=0.01,
                                  max_kl=0.01, cg_iters=50, cg_tol=1e-8,
                                  joint_steps=64,
                                  joint_parameter_rate=0.01,
                                  joint_activity_rate=0.01,
                                  joint_error_rate=0.01,
                                  joint_penalty=1.0,
                                  joint_precision_rate=0.01,
                                  joint_precision_min=0.1,
                                  joint_precision_max=10.0,
                                  exact_kl_steps=1000,
                                  exact_kl_primal_rate=0.01,
                                  exact_kl_dual_rate=0.02,
                                  exact_kl_penalty=1.0,
                                  exact_kl_tol=1e-4,
                                  policy_log_std=None):
    """Apply a settled NPG-PC or quadratic trust-region PC displacement.

    NPG uses the external scale ``step_size``. Trust-region PC directly scales
    the settled (optionally damped) direction to the local Fisher radius; it
    deliberately performs no nonlinear-KL line search or rejection step.
    """
    if mode not in (
            "parameter_npg", "parameter_tr", "parameter_local_tr",
            "parameter_local_joint_tr", "parameter_local_precision_tr",
            "parameter_exact_tr"):
        raise ValueError(
            "mode must be parameter_npg, parameter_tr, parameter_local_tr, "
            "parameter_local_joint_tr, parameter_local_precision_tr, or "
            "parameter_exact_tr")

    if mode in ("parameter_local_joint_tr", "parameter_local_precision_tr"):
        if policy_log_std is not None:
            raise ValueError(
                "joint parameter-PC modes do not yet support state-independent std")
        solve = joint_error_neuron_direction(
            model, obs, pre_tanh, advantages, action_dim,
            damping=damping,
            inference_steps=joint_steps,
            parameter_rate=joint_parameter_rate,
            activity_rate=joint_activity_rate,
            error_rate=joint_error_rate,
            augmented_penalty=joint_penalty,
            precision_rate=(
                joint_precision_rate
                if mode == "parameter_local_precision_tr" else 0.0),
            precision_min=joint_precision_min,
            precision_max=joint_precision_max,
        )
    else:
        direction_solver = (
            local_parameter_pc_direction
            if mode == "parameter_local_tr"
            else parameter_pc_direction)
        solve = direction_solver(
            model, obs, pre_tanh, advantages, action_dim,
            damping=damping, cg_iters=cg_iters, cg_tol=cg_tol,
            policy_log_std=policy_log_std)
    direction = solve["direction"]

    exact_kl_result = None
    if mode in (
            "parameter_tr", "parameter_local_tr", "parameter_local_joint_tr",
            "parameter_local_precision_tr", "parameter_exact_tr"):
        if mode == "parameter_exact_tr":
            exact_kl_result = _settle_exact_kl_scale(
                solve, obs, action_dim, max_kl=max_kl,
                inference_steps=exact_kl_steps,
                primal_rate=exact_kl_primal_rate,
                dual_rate=exact_kl_dual_rate,
                augmented_penalty=exact_kl_penalty,
                inference_tol=exact_kl_tol,
            )
            precision_beta = exact_kl_result["precision_beta"]
            applied_scale = exact_kl_result["applied_scale"]
        else:
            precision_beta = jnp.sqrt(
                solve["quadratic_radius_unscaled"] / (max_radius + 1e-30))
            applied_scale = jnp.where(
                solve["quadratic_radius_unscaled"] > 0.0,
                1.0 / (precision_beta + 1e-30),
                0.0)
    else:
        precision_beta = jnp.asarray(1.0, dtype=direction.dtype)
        applied_scale = jnp.asarray(step_size, dtype=direction.dtype)

    old_outputs = solve["old_outputs"]
    old_log_prob = solve["old_log_prob"]
    candidate_arrays, candidate_log_std = _unpack_solve_parameters(
        solve,
        solve["flat_parameters"] + applied_scale * direction)
    candidate = eqx.combine(candidate_arrays, solve["static"])
    candidate_outputs = _effective_gaussian_outputs(
        _forward(candidate, obs), action_dim, candidate_log_std)
    candidate_log_prob = _log_prob(candidate_outputs, pre_tanh, action_dim)
    ratio = jnp.exp(candidate_log_prob - old_log_prob)
    old_objective = jnp.mean(advantages)
    candidate_objective = jnp.mean(advantages * ratio)
    candidate_kl = _gaussian_kl(old_outputs, candidate_outputs, action_dim)
    quadratic_radius = (jnp.square(applied_scale)
                        * solve["quadratic_radius_unscaled"])
    return {
        "model": candidate,
        "policy_log_std": candidate_log_std,
        "loss": -candidate_objective,
        "gradient_norm": solve["gradient_norm"],
        "direction_norm": solve["direction_norm"],
        "relative_kkt": solve["relative_kkt"],
        "fisher_equation_cosine": solve["fisher_equation_cosine"],
        "energy": solve["energy"],
        "quadratic_radius": quadratic_radius,
        "actual_kl": candidate_kl,
        "step_scale": applied_scale,
        "precision_beta": precision_beta,
        "objective_before": old_objective,
        "objective_after": candidate_objective,
        "direction": direction,
        "gradient": solve["gradient"],
        "fisher_direction": solve["fisher_direction"],
        "joint_constraint_rms": solve.get(
            "joint_constraint_rms", jnp.asarray(0.0, dtype=direction.dtype)),
        "joint_precision_mean": solve.get(
            "joint_precision_mean", jnp.asarray(1.0, dtype=direction.dtype)),
        "joint_inference_steps": solve.get(
            "joint_inference_steps", jnp.asarray(0, dtype=jnp.int32)),
        "exact_kl_constraint": (
            exact_kl_result["constraint"] if exact_kl_result is not None
            else jnp.asarray(0.0, dtype=direction.dtype)),
        "exact_kl_inference_residual": (
            exact_kl_result["inference_residual"] if exact_kl_result is not None
            else jnp.asarray(0.0, dtype=direction.dtype)),
        "exact_kl_scalar_kkt": (
            exact_kl_result["scalar_kkt"] if exact_kl_result is not None
            else jnp.asarray(0.0, dtype=direction.dtype)),
        "exact_kl_inference_steps": (
            exact_kl_result["inference_steps"] if exact_kl_result is not None
            else jnp.asarray(0, dtype=jnp.int32)),
        "exact_kl_local_scale": (
            exact_kl_result["local_scale"] if exact_kl_result is not None
            else applied_scale),
    }


def make_categorical_parameter_pc_policy_step(
        model, obs, actions, advantages, *, max_radius=0.01,
        cg_iters=50, cg_tol=1e-8):
    """Apply categorical layerwise direct-TR parameter predictive coding."""
    solve = local_categorical_parameter_pc_direction(
        model, obs, actions, advantages,
        damping=0.0, cg_iters=cg_iters, cg_tol=cg_tol)
    direction = solve["direction"]
    precision_beta = jnp.sqrt(
        solve["quadratic_radius_unscaled"] / (max_radius + 1e-30))
    applied_scale = jnp.where(
        solve["quadratic_radius_unscaled"] > 0.0,
        1.0 / (precision_beta + 1e-30),
        0.0,
    )
    candidate_arrays = solve["unravel"](
        solve["flat_parameters"] + applied_scale * direction)
    candidate = eqx.combine(candidate_arrays, solve["static"])
    candidate_logits = _forward(candidate, obs)
    candidate_log_probs = jax.nn.log_softmax(candidate_logits, axis=-1)
    candidate_log_prob = jnp.take_along_axis(
        candidate_log_probs, actions[:, None], axis=-1).squeeze(-1)
    ratio = jnp.exp(candidate_log_prob - solve["old_log_prob"])
    candidate_objective = jnp.mean(advantages * ratio)
    return {
        "model": candidate,
        "loss": -candidate_objective,
        "gradient_norm": solve["gradient_norm"],
        "direction_norm": solve["direction_norm"],
        "relative_kkt": solve["relative_kkt"],
        "fisher_equation_cosine": solve["fisher_equation_cosine"],
        "energy": solve["energy"],
        "quadratic_radius": (
            jnp.square(applied_scale)
            * solve["quadratic_radius_unscaled"]),
        "actual_kl": _categorical_kl(
            solve["old_outputs"], candidate_logits),
        "step_scale": applied_scale,
        "precision_beta": precision_beta,
        "objective_before": jnp.mean(advantages),
        "objective_after": candidate_objective,
        "direction": direction,
        "gradient": solve["gradient"],
        "fisher_direction": solve["fisher_direction"],
    }
