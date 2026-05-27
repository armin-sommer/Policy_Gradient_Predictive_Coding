"""Seed-vmapped, step-scanned training loop. Identical for every update rule.

run_seed(step_fn, policy, env, seed, num_steps) -> Log
run_all(updates_dict, policy, env, num_seeds, num_steps) -> {name: Log}
"""
import jax
import jax.numpy as jnp

from mdp_experiments._base import Log, flatten, kl_categorical


def run_seed(step_fn, policy, env, seed: int, num_steps: int) -> Log:
    key = jax.random.PRNGKey(seed)
    init_key, *step_keys = jax.random.split(key, num_steps + 1)
    params0 = policy.init(init_key)
    logits_fn = policy.logits_fn

    def body(carry, k):
        params, J_prev, theta_prev = carry
        p_old = jax.nn.softmax(logits_fn(params))
        new_params = step_fn(params, k)
        p_new = jax.nn.softmax(logits_fn(new_params))
        theta_new = flatten(new_params)
        d = theta_new - theta_prev
        d_unit = d / (jnp.linalg.norm(d) + 1e-12)
        J_new = env.J(new_params, logits_fn=logits_fn)
        log = (theta_new,
               p_new[1] if policy.n_actions >= 2 else p_new[0],
               d_unit,
               kl_categorical(p_old, p_new),
               J_new - J_prev)
        return (new_params, J_new, theta_new), log

    init_carry = (params0,
                  env.J(params0, logits_fn=logits_fn),
                  flatten(params0))
    _, logs = jax.lax.scan(body, init_carry, jnp.stack(step_keys))
    return Log(*logs)


def run_all(updates: dict, policy, env, num_seeds: int, num_steps: int,
            verbose: bool = True) -> dict:
    """Run each (name -> step_fn) over num_seeds seeds, num_steps steps each.

    Each algorithm is jit-vmapped independently for clean compile boundaries.
    """
    seeds = jnp.arange(num_seeds)
    results = {}
    for name, step_fn in updates.items():
        if verbose:
            print(f"running {name}...")
        runner = jax.jit(jax.vmap(
            lambda s: run_seed(step_fn, policy, env, s, num_steps)))
        results[name] = runner(seeds)
    return results
