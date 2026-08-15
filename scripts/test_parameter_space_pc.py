"""Synthetic checks for parameter-space NPG-PC and trust-region PC."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jpc

from pc_algorithms.gaussian_policy import sample_gaussian_action
from pc_algorithms.parameter_space_pc import (
    make_parameter_pc_policy_step,
    parameter_pc_direction,
)
from pc_algorithms.error_neuron_graph import (
    explicit_error_neuron_diagnostics,
    local_parameter_pc_direction,
)


def cosine(a, b):
    return float(jnp.dot(a, b) /
                 (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))


def main():
    batch_size = 128
    obs_dim = 5
    action_dim = 2
    key_model, key_obs, key_action, key_adv = jr.split(jr.PRNGKey(11), 4)
    model = jpc.make_mlp(
        key_model, input_dim=obs_dim, width=8, depth=2,
        output_dim=2 * action_dim, act_fn="tanh", use_bias=True)
    obs = jr.normal(key_obs, (batch_size, obs_dim))
    outputs = jpc.init_activities_with_ffwd(model=model, input=obs)[-1]
    _, pre_tanh = sample_gaussian_action(
        key_action, outputs, action_dim, exp_std=True)
    advantages = jr.normal(key_adv, (batch_size,))
    advantages = ((advantages - advantages.mean())
                  / (advantages.std() + 1e-8))

    print("1. parameter-PC energy reaches its KKT fixed point")
    solve = parameter_pc_direction(
        model, obs, pre_tanh, advantages, action_dim,
        damping=0.1, cg_iters=200, cg_tol=1e-10)
    print(f"   relative KKT residual: {float(solve['relative_kkt']):.3e}")
    assert float(solve["relative_kkt"]) < 1e-3

    print("\n2. inferred direction satisfies the damped Fisher equation")
    lhs = solve["fisher_direction"] + 0.1 * solve["direction"]
    alignment = cosine(lhs, solve["gradient"])
    print(f"   cos((F+lambda I)d, g): {alignment:.9f}")
    assert alignment > 0.99999

    print("\n3. the same solution satisfies the layer-local error-neuron equations")
    local = explicit_error_neuron_diagnostics(
        model, solve["unravel"](solve["direction"]), obs, pre_tanh,
        advantages, action_dim, damping=0.1)
    print(f"   activity constraint RMS: {float(local['constraint_rms']):.3e}")
    print(f"   local/global JVP error:  {float(local['output_jvp_max_error']):.3e}")
    print(f"   local parameter KKT:     {float(local['local_parameter_kkt_relative']):.3e}")
    assert float(local["output_jvp_max_error"]) < 1e-5
    assert float(local["local_parameter_kkt_relative"]) < 1e-3

    print("\n3b. explicit local error neurons reproduce the global solve")
    local_solve = local_parameter_pc_direction(
        model, obs, pre_tanh, advantages, action_dim,
        damping=0.1, cg_iters=200, cg_tol=1e-10)
    gradient_alignment = cosine(local_solve["gradient"], solve["gradient"])
    direction_alignment = cosine(local_solve["direction"], solve["direction"])
    print(f"   gradient cosine:  {gradient_alignment:.9f}")
    print(f"   direction cosine: {direction_alignment:.9f}")
    print(f"   local KKT:        {float(local_solve['relative_kkt']):.3e}")
    assert gradient_alignment > 0.99999
    assert direction_alignment > 0.99999
    assert float(local_solve["relative_kkt"]) < 1e-3

    print("\n4. NPG-PC directly applies eta times the settled direction")
    npg = make_parameter_pc_policy_step(
        model, obs, pre_tanh, advantages, action_dim,
        mode="parameter_npg", damping=0.1, step_size=0.02,
        cg_iters=200, cg_tol=1e-10)
    arrays = [x for x in jax.tree_util.tree_leaves(npg["model"])
              if eqx.is_array(x)]
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in arrays)
    print(f"   eta: {float(npg['step_scale']):.3f}, "
          f"actual KL: {float(npg['actual_kl']):.3e}")
    assert abs(float(npg["step_scale"]) - 0.02) < 1e-7

    print("\n5. TR-PC infers beta and directly applies the quadratic boundary")
    tr = make_parameter_pc_policy_step(
        model, obs, pre_tanh, advantages, action_dim,
        mode="parameter_tr", damping=0.0, max_radius=0.01,
        cg_iters=200, cg_tol=1e-10)
    print(f"   inferred beta: {float(tr['precision_beta']):.6f}")
    print(f"   quadratic radius: {float(tr['quadratic_radius']):.6f}")
    print(f"   actual KL:    {float(tr['actual_kl']):.6f}")
    print(f"   direct scale: {float(tr['step_scale']):.6f}")
    assert abs(float(tr["quadratic_radius"]) - 0.01) < 1e-5

    print("\n5b. layer-local TR-PC reaches the same quadratic boundary")
    local_tr = make_parameter_pc_policy_step(
        model, obs, pre_tanh, advantages, action_dim,
        mode="parameter_local_tr", damping=0.0, max_radius=0.01,
        cg_iters=200, cg_tol=1e-10)
    print(f"   quadratic radius: {float(local_tr['quadratic_radius']):.6f}")
    print(f"   actual KL:        {float(local_tr['actual_kl']):.6f}")
    assert abs(float(local_tr["quadratic_radius"]) - 0.01) < 1e-5

    print("\n5c. global log_std is a shared parameter activity in both solves")
    mean_model = jpc.make_mlp(
        key_model, input_dim=obs_dim, width=8, depth=2,
        output_dim=action_dim, act_fn="tanh", use_bias=True)
    mean = jpc.init_activities_with_ffwd(
        model=mean_model, input=obs)[-1]
    global_log_std = jnp.array([-0.2, 0.1], dtype=mean.dtype)
    global_outputs = jnp.concatenate([
        mean, jnp.broadcast_to(global_log_std, mean.shape)], axis=-1)
    _, global_pre_tanh = sample_gaussian_action(
        key_action, global_outputs, action_dim, exp_std=True)
    global_solve = parameter_pc_direction(
        mean_model, obs, global_pre_tanh, advantages, action_dim,
        damping=0.1, cg_iters=200, cg_tol=1e-10,
        policy_log_std=global_log_std)
    global_local = local_parameter_pc_direction(
        mean_model, obs, global_pre_tanh, advantages, action_dim,
        damping=0.1, cg_iters=200, cg_tol=1e-10,
        policy_log_std=global_log_std)
    global_alignment = cosine(
        global_solve["direction"], global_local["direction"])
    print(f"   global/local direction cosine: {global_alignment:.9f}")
    print(f"   local KKT: {float(global_local['relative_kkt']):.3e}")
    assert global_alignment > 0.99999
    assert float(global_local["relative_kkt"]) < 1e-3
    global_tr = make_parameter_pc_policy_step(
        mean_model, obs, global_pre_tanh, advantages, action_dim,
        mode="parameter_local_tr", damping=0.0, max_radius=0.01,
        cg_iters=200, cg_tol=1e-10, policy_log_std=global_log_std)
    assert global_tr["policy_log_std"].shape == global_log_std.shape
    assert abs(float(global_tr["quadratic_radius"]) - 0.01) < 1e-5

    print("\n5d. finite joint and precision dynamics produce valid TR steps")
    for mode in (
            "parameter_local_joint_tr", "parameter_local_precision_tr"):
        joint_tr = make_parameter_pc_policy_step(
            model, obs, pre_tanh, advantages, action_dim,
            mode=mode, damping=0.0, max_radius=0.01,
            joint_steps=64, joint_parameter_rate=0.01,
            joint_activity_rate=0.01, joint_error_rate=0.01,
            joint_penalty=1.0, joint_precision_rate=0.01)
        print(
            f"   {mode}: radius={float(joint_tr['quadratic_radius']):.6f}, "
            f"cos={float(joint_tr['fisher_equation_cosine']):.6f}, "
            f"constraint={float(joint_tr['joint_constraint_rms']):.3e}, "
            f"precision={float(joint_tr['joint_precision_mean']):.3f}")
        assert bool(jnp.isfinite(joint_tr["actual_kl"]))
        assert float(joint_tr["quadratic_radius"]) > 0.0
        assert abs(float(joint_tr["quadratic_radius"]) - 0.01) < 1e-5

    print("\n6. exact-KL TR-PC jointly settles step and dual activities")
    exact_tr = make_parameter_pc_policy_step(
        model, obs, pre_tanh, advantages, action_dim,
        mode="parameter_exact_tr", damping=0.0, max_kl=0.01,
        cg_iters=200, cg_tol=1e-10,
        exact_kl_steps=1000, exact_kl_primal_rate=0.01,
        exact_kl_dual_rate=0.02, exact_kl_penalty=1.0,
        exact_kl_tol=1e-4)
    print(f"   exact KL:       {float(exact_tr['actual_kl']):.6f}")
    print(f"   constraint:     {float(exact_tr['exact_kl_constraint']):.3e}")
    print(f"   scalar KKT:     {float(exact_tr['exact_kl_scalar_kkt']):.3e}")
    print(f"   inference steps:{int(exact_tr['exact_kl_inference_steps']):5d}")
    print(f"   local/exact scale: "
          f"{float(exact_tr['exact_kl_local_scale']):.6f} / "
          f"{float(exact_tr['step_scale']):.6f}")
    assert abs(float(exact_tr["actual_kl"]) - 0.01) < 2e-4
    assert abs(float(exact_tr["exact_kl_constraint"])) < 2e-2

    print("\nall parameter-space PC checks passed")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    main()
