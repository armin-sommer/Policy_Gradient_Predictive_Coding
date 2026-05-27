"""Closed-form Natural Policy Gradient: F^{-1} grad J, full expectation.

No sampling noise. This is the oracle direction the other algorithms are
compared against. Requires env_J (closed-form expected return). Bandit-only
because the Fisher solve is built at a single fixed observation.
"""
import jax

from algorithms._utils import flatten, unflatten
from algorithms.mdp_experiments._fisher import fisher_inv_g


def make_step(rollout_fn, policy, *, lr: float, env_J=None, gamma: float = 0.99):
    del rollout_fn, gamma  # NPG is closed-form and consumes no rollout
    if env_J is None:
        raise ValueError("NPG (oracle) requires env_J to be defined")
    apply = policy.apply
    logits_fn = policy.logits_fn
    n_actions = policy.n_actions

    def step(params, opt_state, key):
        del key
        g = jax.grad(lambda p: -env_J(p, apply=apply))(params)
        nat_flat = fisher_inv_g(params, g,
                                logits_fn=logits_fn, n_actions=n_actions)
        new_params = unflatten(params, flatten(params) - lr * nat_flat)
        mean_ep_return = env_J(new_params, apply=apply)
        return new_params, opt_state, {"mean_ep_return": mean_ep_return}

    return (lambda _: ()), step
