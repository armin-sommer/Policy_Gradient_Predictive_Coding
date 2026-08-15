"""Deterministic checks for additive precision-energy policy inference.

Run with:
    PYTHONPATH=src python scripts/test_precision_energy.py
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from pc_algorithms.precision_energy import (
    effective_policy_outputs,
    natural_output_displacement,
    quadratic_policy_radius,
    settle_precision_outputs,
    trust_region_precision,
)


def main():
    batch_size = 32
    action_dim = 3
    key_loc, key_log_std, key_action, key_adv = jr.split(jr.PRNGKey(7), 4)
    loc = 0.2 * jr.normal(key_loc, (batch_size, action_dim))
    log_std = 0.1 * jr.normal(key_log_std, (batch_size, action_dim))
    old_params = jnp.concatenate([loc, log_std], axis=-1)
    pre_tanh = loc + jnp.exp(log_std) * jr.normal(
        key_action, (batch_size, action_dim))
    # Keep the unconstrained optimum inside the production log-std bounds; hard
    # clipping is intentionally a different constrained problem.
    advantages = 0.1 * jr.normal(key_adv, (batch_size,))

    print("1. fixed-precision inference equals output NPG")
    beta = jnp.asarray(1.7)
    result = settle_precision_outputs(
        old_params, pre_tanh, advantages,
        action_dim=action_dim, beta=beta, max_t1=20.0)
    expected = effective_policy_outputs(old_params, action_dim) + natural_output_displacement(
        old_params, pre_tanh, advantages, action_dim) / beta
    max_error = float(jnp.max(jnp.abs(result["targets"] - expected)))
    print(f"   max equilibrium error: {max_error:.3e}")
    print(f"   residual RMS:          {float(result['residual_rms']):.3e}")
    assert max_error < 2e-4

    print("\n2. adaptive precision reaches the quadratic trust boundary")
    max_radius = 0.015
    tr_beta = trust_region_precision(
        old_params, pre_tanh, advantages, action_dim, max_radius)
    tr_result = settle_precision_outputs(
        old_params, pre_tanh, advantages,
        action_dim=action_dim, beta=tr_beta, max_t1=20.0)
    radius = float(quadratic_policy_radius(
        old_params, tr_result["targets"], action_dim))
    print(f"   beta:                  {float(tr_beta):.6f}")
    print(f"   requested radius:      {max_radius:.6f}")
    print(f"   settled radius:        {radius:.6f}")
    print(f"   residual RMS:          {float(tr_result['residual_rms']):.3e}")
    assert abs(radius - max_radius) < 2e-5

    print("\nall precision-energy checks passed")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    main()
