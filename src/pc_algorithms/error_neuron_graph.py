"""Layer-local error-neuron view of parameter-space policy PC.

This module does not solve the global quadratic. It exposes the local primal
constraints and dual error neurons that are exactly equivalent to the implicit
JVP/Fisher/VJP solve in ``parameter_space_pc.py``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from pc_algorithms import gaussian_policy as gpol
from pc_algorithms.gaussian_policy import (
    split_gaussian_params,
)


def _activate(block, values):
    return jax.vmap(block.layers[0])(values)


def _forward(model, obs):
    values = obs
    for block in model:
        values = jax.vmap(block)(values)
    return values


def _log_prob(params, pre_tanh, action_dim):
    loc, scale, log_scale = split_gaussian_params(
        params, action_dim, exp_std=True)
    return jnp.sum(
        -0.5 * jnp.square((pre_tanh - loc) / scale) - log_scale,
        axis=-1)


def local_parameter_pc_direction(model, obs, pre_tanh, advantages, action_dim,
                                 *, damping=0.1, cg_iters=50, cg_tol=1e-8,
                                 policy_log_std=None):
    """Solve the parameter-PC energy with explicit layer-local error neurons.

    No global reverse-mode transform is used. A parameter tangent is propagated
    through the local linearised activity constraints, output precision acts on
    the final perturbation, and neighboring transposed weights propagate the
    resulting error-neuron activities back one layer at a time.
    """
    arrays, static = eqx.partition(model, eqx.is_array)
    state_independent_std = policy_log_std is not None
    parameter_state = (
        (arrays, policy_log_std) if state_independent_std else arrays)
    flat0, unravel = ravel_pytree(parameter_state)
    batch_size = obs.shape[0]

    bases = []
    activated_inputs = []
    activation_derivatives = []
    base_prev = obs
    for block in model:
        activated = _activate(block, base_prev)
        derivative = jax.jvp(
            lambda x: _activate(block, x),
            (base_prev,), (jnp.ones_like(base_prev),))[1]
        linear = block.layers[1]
        base = activated @ linear.weight.T
        if linear.bias is not None:
            base = base + linear.bias
        activated_inputs.append(activated)
        activation_derivatives.append(derivative)
        bases.append(base)
        base_prev = base

    network_outputs = bases[-1]
    if state_independent_std:
        mean = network_outputs[..., :action_dim]
        old_outputs = jnp.concatenate(
            [mean, jnp.broadcast_to(policy_log_std, mean.shape)], axis=-1)
    else:
        old_outputs = network_outputs
    old_log_prob = _log_prob(old_outputs, pre_tanh, action_dim)
    loc, scale, _ = split_gaussian_params(
        old_outputs, action_dim, exp_std=True)
    raw_log_std = old_outputs[..., action_dim:]
    in_bounds = (
        jnp.ones_like(raw_log_std) if state_independent_std else
        ((raw_log_std > gpol.LOG_STD_MIN)
         & (raw_log_std < gpol.LOG_STD_MAX)).astype(old_outputs.dtype))
    output_fisher_diag = jnp.concatenate(
        [1.0 / jnp.square(scale), 2.0 * in_bounds], axis=-1)
    score_drive = jnp.concatenate(
        [(pre_tanh - loc) / jnp.square(scale),
         in_bounds * (jnp.square((pre_tanh - loc) / scale) - 1.0)], axis=-1)
    output_gradient_drive = advantages[:, None] * score_drive

    def tangent_output(vector):
        tangent_parameters = unravel(vector)
        if state_independent_std:
            tangent_arrays, tangent_log_std = tangent_parameters
        else:
            tangent_arrays, tangent_log_std = tangent_parameters, None
        tangent_model = eqx.combine(tangent_arrays, static)
        delta_prev = jnp.zeros_like(obs)
        for layer_index, (block, tangent_block) in enumerate(
                zip(model, tangent_model)):
            linear = block.layers[1]
            tangent_linear = tangent_block.layers[1]
            delta_activated = (
                activation_derivatives[layer_index] * delta_prev)
            delta = delta_activated @ linear.weight.T
            delta = delta + (
                activated_inputs[layer_index] @ tangent_linear.weight.T)
            if tangent_linear.bias is not None:
                delta = delta + tangent_linear.bias
            delta_prev = delta
        if state_independent_std:
            delta_mean = delta_prev[..., :action_dim]
            return jnp.concatenate([
                delta_mean,
                jnp.broadcast_to(tangent_log_std, delta_mean.shape),
            ], axis=-1)
        return delta_prev

    def local_parameter_pullback(output_error):
        if state_independent_std:
            error = output_error[..., :action_dim]
            log_std_signal = jnp.mean(
                output_error[..., action_dim:], axis=0)
        else:
            error = output_error
            log_std_signal = None
        weight_signals = [None] * len(model)
        bias_signals = [None] * len(model)
        for layer_index in range(len(model) - 1, -1, -1):
            linear = model[layer_index].layers[1]
            activated = activated_inputs[layer_index]
            weight_signals[layer_index] = (
                jnp.einsum('bo,bi->oi', error, activated) / batch_size)
            if linear.bias is not None:
                bias_signals[layer_index] = jnp.mean(error, axis=0)
            if layer_index > 0:
                error = (error @ linear.weight) * (
                    activation_derivatives[layer_index])

        # Equinox flattens each JPC block as linear weight followed by bias.
        leaves = []
        for weight_signal, bias_signal in zip(weight_signals, bias_signals):
            leaves.append(jnp.ravel(weight_signal))
            if bias_signal is not None:
                leaves.append(jnp.ravel(bias_signal))
        if log_std_signal is not None:
            leaves.append(jnp.ravel(log_std_signal))
        return jnp.concatenate(leaves)

    gradient = local_parameter_pullback(output_gradient_drive)

    def fisher_vector_product(vector):
        return local_parameter_pullback(
            output_fisher_diag * tangent_output(vector))

    def damped_fisher(vector):
        return fisher_vector_product(vector) + damping * vector

    direction, _ = jax.scipy.sparse.linalg.cg(
        damped_fisher, gradient, tol=cg_tol, maxiter=cg_iters)
    fisher_direction = fisher_vector_product(direction)
    kkt = fisher_direction + damping * direction - gradient
    relative_kkt = (
        jnp.linalg.norm(kkt) / (jnp.linalg.norm(gradient) + 1e-30))
    fisher_equation_cosine = (
        jnp.dot(fisher_direction + damping * direction, gradient)
        / (jnp.linalg.norm(fisher_direction + damping * direction)
           * jnp.linalg.norm(gradient) + 1e-30))
    quadratic_radius_unscaled = 0.5 * jnp.dot(
        direction, fisher_direction)
    energy = (0.5 * jnp.dot(direction, fisher_direction + damping * direction)
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
        "objective": jnp.mean(advantages * old_log_prob),
        "state_independent_std": state_independent_std,
    }


def local_categorical_parameter_pc_direction(
        model, obs, actions, advantages, *, damping=0.0, cg_iters=50,
        cg_tol=1e-8):
    """Solve categorical parameter PC with explicit layer-local errors.

    The categorical output precision is the softmax Fisher
    ``diag(p) - p p.T``. It acts locally on logit perturbations before the
    resulting error population is propagated through neighboring layers.
    """
    arrays, static = eqx.partition(model, eqx.is_array)
    flat0, unravel = ravel_pytree(arrays)
    batch_size = obs.shape[0]

    bases = []
    activated_inputs = []
    activation_derivatives = []
    base_prev = obs
    for block in model:
        activated = _activate(block, base_prev)
        derivative = jax.jvp(
            lambda x: _activate(block, x),
            (base_prev,), (jnp.ones_like(base_prev),))[1]
        linear = block.layers[1]
        base = activated @ linear.weight.T
        if linear.bias is not None:
            base = base + linear.bias
        activated_inputs.append(activated)
        activation_derivatives.append(derivative)
        bases.append(base)
        base_prev = base

    old_logits = bases[-1]
    old_log_probs = jax.nn.log_softmax(old_logits, axis=-1)
    old_probs = jnp.exp(old_log_probs)
    old_log_prob = jnp.take_along_axis(
        old_log_probs, actions[:, None], axis=-1).squeeze(-1)
    score_drive = jax.nn.one_hot(
        actions, old_logits.shape[-1], dtype=old_logits.dtype) - old_probs
    output_gradient_drive = advantages[:, None] * score_drive

    def tangent_output(vector):
        tangent_model = eqx.combine(unravel(vector), static)
        delta_prev = jnp.zeros_like(obs)
        for layer_index, (block, tangent_block) in enumerate(
                zip(model, tangent_model)):
            linear = block.layers[1]
            tangent_linear = tangent_block.layers[1]
            delta = (
                activation_derivatives[layer_index] * delta_prev
            ) @ linear.weight.T
            delta = delta + (
                activated_inputs[layer_index] @ tangent_linear.weight.T)
            if tangent_linear.bias is not None:
                delta = delta + tangent_linear.bias
            delta_prev = delta
        return delta_prev

    def local_parameter_pullback(output_error):
        error = output_error
        weight_signals = [None] * len(model)
        bias_signals = [None] * len(model)
        for layer_index in range(len(model) - 1, -1, -1):
            linear = model[layer_index].layers[1]
            activated = activated_inputs[layer_index]
            weight_signals[layer_index] = (
                jnp.einsum('bo,bi->oi', error, activated) / batch_size)
            if linear.bias is not None:
                bias_signals[layer_index] = jnp.mean(error, axis=0)
            if layer_index > 0:
                error = (error @ linear.weight) * (
                    activation_derivatives[layer_index])
        leaves = []
        for weight_signal, bias_signal in zip(weight_signals, bias_signals):
            leaves.append(jnp.ravel(weight_signal))
            if bias_signal is not None:
                leaves.append(jnp.ravel(bias_signal))
        return jnp.concatenate(leaves)

    def output_fisher(vector):
        centered = vector - jnp.sum(
            old_probs * vector, axis=-1, keepdims=True)
        return old_probs * centered

    gradient = local_parameter_pullback(output_gradient_drive)

    def fisher_vector_product(vector):
        return local_parameter_pullback(output_fisher(tangent_output(vector)))

    def damped_fisher(vector):
        return fisher_vector_product(vector) + damping * vector

    direction, _ = jax.scipy.sparse.linalg.cg(
        damped_fisher, gradient, tol=cg_tol, maxiter=cg_iters)
    fisher_direction = fisher_vector_product(direction)
    kkt = fisher_direction + damping * direction - gradient
    relative_kkt = (
        jnp.linalg.norm(kkt) / (jnp.linalg.norm(gradient) + 1e-30))
    fisher_equation_cosine = (
        jnp.dot(fisher_direction + damping * direction, gradient)
        / (jnp.linalg.norm(fisher_direction + damping * direction)
           * jnp.linalg.norm(gradient) + 1e-30))
    quadratic_radius_unscaled = 0.5 * jnp.dot(
        direction, fisher_direction)
    energy = (0.5 * jnp.dot(
        direction, fisher_direction + damping * direction)
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
        "old_outputs": old_logits,
        "old_log_prob": old_log_prob,
        "objective": jnp.mean(advantages * old_log_prob),
    }


def joint_error_neuron_direction(
        model, obs, pre_tanh, advantages, action_dim, *, damping=0.0,
        inference_steps=64, parameter_rate=0.01, activity_rate=0.01,
        error_rate=0.01, augmented_penalty=1.0, precision_rate=0.0,
        precision_min=0.1, precision_max=10.0):
    """Infer a finite-time parameter correction with local primal-dual dynamics.

    Unlike :func:`local_parameter_pc_direction`, this does not eliminate the
    layer activities and then run CG. Parameter perturbations, layer activity
    perturbations, and constraint-error neurons are all free variables. A
    nonzero ``precision_rate`` additionally infers one positive constraint
    precision per layer from its current prediction-error variance.

    Finite settling is intentional: it suppresses poorly conditioned Fisher
    eigendirections instead of amplifying them all the way to the pseudoinverse.
    The final direct-TR rescaling is performed by ``parameter_space_pc.py``.
    """
    arrays, static = eqx.partition(model, eqx.is_array)
    flat0, unravel = ravel_pytree(arrays)
    batch_size = obs.shape[0]

    bases = []
    activated_inputs = []
    activation_derivatives = []
    base_prev = obs
    for block in model:
        activated = _activate(block, base_prev)
        derivative = jax.jvp(
            lambda x: _activate(block, x),
            (base_prev,), (jnp.ones_like(base_prev),))[1]
        linear = block.layers[1]
        base = activated @ linear.weight.T
        if linear.bias is not None:
            base = base + linear.bias
        activated_inputs.append(activated)
        activation_derivatives.append(derivative)
        bases.append(base)
        base_prev = base

    old_outputs = bases[-1]
    old_log_prob = _log_prob(old_outputs, pre_tanh, action_dim)
    loc, scale, _ = split_gaussian_params(
        old_outputs, action_dim, exp_std=True)
    raw_log_std = old_outputs[..., action_dim:]
    in_bounds = ((raw_log_std > gpol.LOG_STD_MIN)
                 & (raw_log_std < gpol.LOG_STD_MAX)).astype(old_outputs.dtype)
    output_fisher_diag = jnp.concatenate(
        [1.0 / jnp.square(scale), 2.0 * in_bounds], axis=-1)
    score_drive = jnp.concatenate(
        [(pre_tanh - loc) / jnp.square(scale),
         in_bounds * (jnp.square((pre_tanh - loc) / scale) - 1.0)], axis=-1)
    output_gradient_drive = advantages[:, None] * score_drive

    def constraints(direction, activities):
        tangent_model = eqx.combine(unravel(direction), static)
        result = []
        delta_prev = jnp.zeros_like(obs)
        for layer_index, (block, tangent_block) in enumerate(
                zip(model, tangent_model)):
            linear = block.layers[1]
            tangent_linear = tangent_block.layers[1]
            predicted = (
                activation_derivatives[layer_index] * delta_prev
            ) @ linear.weight.T
            predicted = predicted + (
                activated_inputs[layer_index] @ tangent_linear.weight.T)
            if tangent_linear.bias is not None:
                predicted = predicted + tangent_linear.bias
            result.append(activities[layer_index] - predicted)
            delta_prev = activities[layer_index]
        return tuple(result)

    def flatten_parameter_signals(layer_errors):
        leaves = []
        for error, block, activated in zip(
                layer_errors, model, activated_inputs):
            weight_signal = jnp.einsum(
                'bo,bi->oi', error, activated) / batch_size
            leaves.append(jnp.ravel(weight_signal))
            if block.layers[1].bias is not None:
                leaves.append(jnp.ravel(jnp.mean(error, axis=0)))
        return jnp.concatenate(leaves)

    def dynamics(direction, activities, errors, log_precisions):
        residuals = constraints(direction, activities)
        precisions = jnp.exp(log_precisions)
        effective_errors = tuple(
            error + augmented_penalty * precision * residual
            for error, precision, residual in zip(
                errors, precisions, residuals))

        parameter_velocity = (
            flatten_parameter_signals(effective_errors)
            - damping * direction)
        activity_velocities = []
        for layer_index, effective_error in enumerate(effective_errors):
            velocity = -effective_error
            if layer_index == len(model) - 1:
                velocity = velocity + output_gradient_drive
                velocity = velocity - (
                    output_fisher_diag * activities[layer_index])
            else:
                child = model[layer_index + 1]
                transmitted = (
                    effective_errors[layer_index + 1]
                    @ child.layers[1].weight)
                velocity = velocity + (
                    activation_derivatives[layer_index + 1] * transmitted)
            activity_velocities.append(velocity)

        residual_variances = jnp.stack([
            jnp.mean(jnp.square(residual)) for residual in residuals
        ])
        # Gradient descent on 0.5*pi*c^2 - 0.5*log(pi), parameterized by log pi.
        precision_velocity = 0.5 * (1.0 - precisions * residual_variances)
        return (parameter_velocity, tuple(activity_velocities), residuals,
                precision_velocity)

    direction0 = jnp.zeros_like(flat0)
    activities0 = tuple(jnp.zeros_like(base) for base in bases)
    errors0 = tuple(jnp.zeros_like(base) for base in bases)
    log_precisions0 = jnp.zeros((len(model),), dtype=flat0.dtype)
    min_log_precision = jnp.log(jnp.asarray(precision_min, dtype=flat0.dtype))
    max_log_precision = jnp.log(jnp.asarray(precision_max, dtype=flat0.dtype))

    def inference_step(_, carry):
        direction, activities, errors, log_precisions = carry
        (parameter_velocity, activity_velocities, residuals,
         precision_velocity) = dynamics(
             direction, activities, errors, log_precisions)
        next_direction = direction + parameter_rate * parameter_velocity
        next_activities = tuple(
            activity + activity_rate * velocity
            for activity, velocity in zip(activities, activity_velocities))
        next_errors = tuple(
            error + error_rate * residual
            for error, residual in zip(errors, residuals))
        next_log_precisions = jnp.clip(
            log_precisions + precision_rate * precision_velocity,
            min_log_precision,
            max_log_precision,
        )
        return (next_direction, next_activities, next_errors,
                next_log_precisions)

    direction, activities, errors, log_precisions = jax.lax.fori_loop(
        0, inference_steps, inference_step,
        (direction0, activities0, errors0, log_precisions0))

    def tangent_output(vector):
        tangent_model = eqx.combine(unravel(vector), static)
        delta_prev = jnp.zeros_like(obs)
        for layer_index, (block, tangent_block) in enumerate(
                zip(model, tangent_model)):
            linear = block.layers[1]
            tangent_linear = tangent_block.layers[1]
            delta_prev = (
                activation_derivatives[layer_index] * delta_prev
            ) @ linear.weight.T + (
                activated_inputs[layer_index] @ tangent_linear.weight.T)
            if tangent_linear.bias is not None:
                delta_prev = delta_prev + tangent_linear.bias
        return delta_prev

    def output_pullback(output_error):
        layer_errors = [None] * len(model)
        layer_errors[-1] = output_error
        for layer_index in range(len(model) - 2, -1, -1):
            child = model[layer_index + 1]
            layer_errors[layer_index] = (
                layer_errors[layer_index + 1] @ child.layers[1].weight
                * activation_derivatives[layer_index + 1])
        return flatten_parameter_signals(layer_errors)

    gradient = output_pullback(output_gradient_drive)
    fisher_direction = output_pullback(
        output_fisher_diag * tangent_output(direction))
    kkt = fisher_direction + damping * direction - gradient
    relative_kkt = (
        jnp.linalg.norm(kkt) / (jnp.linalg.norm(gradient) + 1e-30))
    fisher_equation_cosine = (
        jnp.dot(fisher_direction + damping * direction, gradient)
        / (jnp.linalg.norm(fisher_direction + damping * direction)
           * jnp.linalg.norm(gradient) + 1e-30))
    quadratic_radius_unscaled = 0.5 * jnp.dot(
        direction, fisher_direction)
    energy = (0.5 * jnp.dot(
        direction, fisher_direction + damping * direction)
        - jnp.dot(gradient, direction))
    final_constraints = constraints(direction, activities)

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
        "objective": jnp.mean(advantages * old_log_prob),
        "joint_constraint_rms": jnp.sqrt(jnp.mean(jnp.stack([
            jnp.mean(jnp.square(residual))
            for residual in final_constraints
        ]))),
        "joint_precision_mean": jnp.mean(jnp.exp(log_precisions)),
        "joint_inference_steps": jnp.asarray(inference_steps, dtype=jnp.int32),
    }


def explicit_error_neuron_diagnostics(model, tangent_arrays, obs, pre_tanh,
                                      advantages, action_dim, *, damping):
    """Evaluate local constraint and KKT residuals for a parameter tangent.

    JPC blocks apply an activation followed by a linear map. For block ``l``,
    the linearised activity constraint is

        dx_l = W_l Dphi_l dx_{l-1} + dW_l phi_l(x_{l-1}) + db_l.

    The returned dual activities are obtained by propagating the Gaussian
    output error backward through these local constraints.
    """
    arrays, static = eqx.partition(model, eqx.is_array)
    tangent_model = eqx.combine(tangent_arrays, static)
    batch_size = obs.shape[0]

    bases = []
    deltas = []
    activated_inputs = []
    constraint_residuals = []
    base_prev = obs
    delta_prev = jnp.zeros_like(obs)

    for block, tangent_block in zip(model, tangent_model):
        activated = _activate(block, base_prev)
        delta_activated = jax.jvp(
            lambda x: _activate(block, x),
            (base_prev,), (delta_prev,))[1]
        linear = block.layers[1]
        tangent_linear = tangent_block.layers[1]
        base = activated @ linear.weight.T
        if linear.bias is not None:
            base = base + linear.bias
        predicted_delta = delta_activated @ linear.weight.T
        predicted_delta = predicted_delta + activated @ tangent_linear.weight.T
        if tangent_linear.bias is not None:
            predicted_delta = predicted_delta + tangent_linear.bias

        # The primal activity is constructed by the local constraint. Keeping
        # the residual explicit makes deviations measurable in iterative solvers.
        delta = predicted_delta
        constraint_residuals.append(delta - predicted_delta)
        activated_inputs.append(activated)
        bases.append(base)
        deltas.append(delta)
        base_prev, delta_prev = base, delta

    outputs = bases[-1]
    delta_outputs = deltas[-1]
    loc, scale, _ = split_gaussian_params(outputs, action_dim, exp_std=True)
    raw_log_std = outputs[..., action_dim:]
    in_bounds = ((raw_log_std > gpol.LOG_STD_MIN)
                 & (raw_log_std < gpol.LOG_STD_MAX)).astype(outputs.dtype)
    fisher_diag = jnp.concatenate(
        [1.0 / jnp.square(scale), 2.0 * in_bounds], axis=-1)
    score_drive = jnp.concatenate(
        [(pre_tanh - loc) / jnp.square(scale),
         in_bounds * (jnp.square((pre_tanh - loc) / scale) - 1.0)], axis=-1)
    output_error = advantages[:, None] * score_drive - fisher_diag * delta_outputs

    # q_l is the dual/error-neuron population attached to constraint l.
    errors = [None] * len(model)
    errors[-1] = output_error
    for layer_index in range(len(model) - 2, -1, -1):
        child = model[layer_index + 1]
        child_linear = child.layers[1]
        transmitted = errors[layer_index + 1] @ child_linear.weight
        _, pullback = jax.vjp(
            lambda x: _activate(child, x), bases[layer_index])
        errors[layer_index] = pullback(transmitted)[0]

    parameter_residual_leaves = []
    parameter_signal_leaves = []
    for tangent_block, activated, error in zip(
            tangent_model, activated_inputs, errors):
        tangent_linear = tangent_block.layers[1]
        weight_signal = jnp.einsum('bo,bi->oi', error, activated) / batch_size
        parameter_residual_leaves.append(
            damping * tangent_linear.weight - weight_signal)
        parameter_signal_leaves.append(weight_signal)
        if tangent_linear.bias is not None:
            bias_signal = jnp.mean(error, axis=0)
            parameter_residual_leaves.append(
                damping * tangent_linear.bias - bias_signal)
            parameter_signal_leaves.append(bias_signal)

    residual_flat = jnp.concatenate(
        [jnp.ravel(x) for x in parameter_residual_leaves])
    signal_flat = jnp.concatenate(
        [jnp.ravel(x) for x in parameter_signal_leaves])
    constraint_flat = jnp.concatenate(
        [jnp.ravel(x) for x in constraint_residuals])

    global_delta = jax.jvp(
        lambda a: _forward(eqx.combine(a, static), obs),
        (arrays,), (tangent_arrays,))[1]
    tangent_flat, _ = ravel_pytree(tangent_arrays)
    return {
        "constraint_rms": jnp.sqrt(jnp.mean(jnp.square(constraint_flat))),
        "output_jvp_max_error": jnp.max(jnp.abs(delta_outputs - global_delta)),
        "local_parameter_kkt_relative": (
            jnp.linalg.norm(residual_flat)
            / (jnp.linalg.norm(signal_flat) + damping
               * jnp.linalg.norm(tangent_flat) + 1e-30)),
        "output_error_rms": jnp.sqrt(jnp.mean(jnp.square(output_error))),
        "errors": errors,
        "activity_perturbations": deltas,
    }
