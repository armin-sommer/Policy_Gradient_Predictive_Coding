"""Closed-form Natural Policy Gradient: F^{-1} grad J, full expectation.

No sampling noise. This is the oracle direction the other algorithms are
compared against in the four-panel figure.
"""
import jax

from mdp_experiments._base import flatten, unflatten
from mdp_experiments.updates._fisher import fisher_inv_g


def make_step(env, policy, *, lr: float):
    logits_fn = policy.logits_fn
    n_actions = policy.n_actions
    J = env.J

    def step(params, key):
        # The key argument is unused (no sampling).
        del key
        g = jax.grad(lambda p: -J(p, logits_fn=logits_fn))(params)
        nat_flat = fisher_inv_g(params, g,
                                logits_fn=logits_fn, n_actions=n_actions)
        return unflatten(params, flatten(params) - lr * nat_flat)

    return step
