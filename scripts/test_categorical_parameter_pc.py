"""Deterministic checks for categorical layerwise direct-TR parameter PC."""

import jax
import jax.numpy as jnp
import jax.random as jr
import jpc

from pc_algorithms.error_neuron_graph import (
    local_categorical_parameter_pc_direction,
)
from pc_algorithms.parameter_space_pc import (
    make_categorical_parameter_pc_policy_step,
)


def main():
    key_model, key_obs, key_actions = jr.split(jr.PRNGKey(7), 3)
    model = jpc.make_mlp(
        key_model, input_dim=3, width=8, depth=2, output_dim=2,
        act_fn="tanh", use_bias=True)
    obs = jr.normal(key_obs, (128, 3))
    actions = jr.randint(key_actions, (128,), 0, 2)
    rewards = jnp.where(actions == 0, 1.0, 0.9)
    advantages = rewards - rewards.mean()

    solve = local_categorical_parameter_pc_direction(
        model, obs, actions, advantages,
        damping=0.1, cg_iters=200, cg_tol=1e-10)
    print(f"categorical KKT: {float(solve['relative_kkt']):.3e}")
    print(f"categorical Fisher cosine: "
          f"{float(solve['fisher_equation_cosine']):.9f}")
    assert float(solve["relative_kkt"]) < 1e-3
    assert float(solve["fisher_equation_cosine"]) > 0.99999

    result = make_categorical_parameter_pc_policy_step(
        model, obs, actions, advantages,
        max_radius=0.01, cg_iters=200, cg_tol=1e-10)
    print(f"quadratic radius: {float(result['quadratic_radius']):.8f}")
    print(f"actual categorical KL: {float(result['actual_kl']):.8f}")
    assert bool(jnp.isfinite(result["actual_kl"]))
    assert abs(float(result["quadratic_radius"]) - 0.01) < 1e-5


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    main()
