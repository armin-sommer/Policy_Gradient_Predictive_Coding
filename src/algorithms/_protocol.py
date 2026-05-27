"""Algorithm contract — the seam between algorithms and everything else.

Every algorithm exposes:

    make_step(rollout_fn, policy, *, env_J=None, **hp)
        -> (init_opt_state_fn, step_fn)

    init_opt_state_fn(initial_params) -> opt_state
    step_fn(params, opt_state, key) -> (new_params, new_opt_state, metrics)

where:
    rollout_fn(params, key) -> Rollout        produced by the driver.
    env_J(params, *, apply) -> scalar         optional closed-form return.
    metrics                  dict             must include "mean_ep_return".

Algorithms see only rollout_fn (not env) so the same algorithm runs on a
bandit, a chain MDP, or a Procgen env without modification. env_J is the
single, optional special-case channel for NPG (oracle) and TRPO (line
search); other algorithms ignore it.

init_opt_state_fn is called once by the runner with the initial params so
optax can shape its state (e.g. Adam moments). Toy algorithms here return
lambda _: () as a no-op placeholder; full algorithms return optimizer.init.
"""
from typing import NamedTuple

import jax.numpy as jnp


class Log(NamedTuple):
    theta:           jnp.ndarray   # (T, D) flattened param vector per iteration
    mean_ep_return:  jnp.ndarray   # (T,)   mean episodic return per iteration
    dtheta:          jnp.ndarray   # (T, D) unit update direction
    kl:              jnp.ndarray   # (T,)   realized KL(pi_t || pi_{t-1})
    dJ:              jnp.ndarray   # (T,)   J(theta_t) - J(theta_{t-1}); 0 if env.J is None
