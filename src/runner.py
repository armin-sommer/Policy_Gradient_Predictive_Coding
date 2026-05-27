"""Seed-vmapped, iteration-scanned training loop. Identical for every algorithm.

run_seed(step_pkg, policy, J_fn, seed, num_iters) -> Log
run_all(updates_dict, policy, J_fn, num_seeds, num_iters) -> {name: Log}

step_pkg = (init_opt_state, step_fn) produced by an algorithm's make_step.
step_fn signature: (params, opt_state, key) -> (params, opt_state, metrics).
metrics must include "mean_ep_return"; the runner pulls it into Log.

J_fn is the env's closed-form expected return (or None). When None, dJ is
held at zero — sample-based envs (Procgen) still log everything else.
"""
import jax
import jax.numpy as jnp

from algorithms._protocol import Log
from algorithms._utils import flatten, kl_categorical


def run_seed(step_pkg, policy, J_fn, seed: int, num_iters: int) -> Log:
    init_opt_state_fn, step_fn = step_pkg
    key = jax.random.PRNGKey(seed)
    init_key, *step_keys = jax.random.split(key, num_iters + 1)
    params0 = policy.init(init_key)
    init_opt_state = init_opt_state_fn(params0)
    logits_fn = policy.logits_fn
    apply = policy.apply

    J = J_fn if J_fn is not None else (lambda p, *, apply: jnp.float32(0.0))

    def body(carry, k):
        params, opt_state, J_prev, theta_prev = carry
        p_old = jax.nn.softmax(logits_fn(params))
        new_params, new_opt_state, metrics = step_fn(params, opt_state, k)
        p_new = jax.nn.softmax(logits_fn(new_params))
        theta_new = flatten(new_params)
        d = theta_new - theta_prev
        d_unit = d / (jnp.linalg.norm(d) + 1e-12)
        J_new = J(new_params, apply=apply)
        log = (theta_new,
               metrics["mean_ep_return"],
               d_unit,
               kl_categorical(p_old, p_new),
               J_new - J_prev)
        return (new_params, new_opt_state, J_new, theta_new), log

    init_carry = (params0, init_opt_state,
                  J(params0, apply=apply),
                  flatten(params0))
    _, logs = jax.lax.scan(body, init_carry, jnp.stack(step_keys))
    return Log(*logs)


def run_all(updates: dict, policy, J_fn,
            num_seeds: int, num_iters: int,
            verbose: bool = True) -> dict:
    seeds = jnp.arange(num_seeds)
    results = {}
    for name, step_pkg in updates.items():
        if verbose:
            print(f"running {name}...")
        runner = jax.jit(jax.vmap(
            lambda s: run_seed(step_pkg, policy, J_fn, s, num_iters)))
        results[name] = runner(seeds)
    return results
