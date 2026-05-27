"""Closed-form Fisher solve for the small categorical policies used here.

For a discrete policy at a single observation, the Fisher matrix is
  F = sum_a pi(a) grad_logp(a) grad_logp(a)^T
where the gradient is taken with respect to the full flat parameter vector.
Damped solve: (F + lambda I) x = g.  Cheap because n_actions and dim are small.
"""
import jax
import jax.numpy as jnp
import distrax

from algorithms._utils import flatten, unflatten


def fisher_inv_g(params, g, *, logits_fn, n_actions: int,
                 damping: float = 1e-3) -> jnp.ndarray:
    g_flat = flatten(g)
    theta_flat = flatten(params)
    n = g_flat.size

    def logp(theta_flat_, a):
        unflat = unflatten(params, theta_flat_)
        return distrax.Categorical(logits=logits_fn(unflat)).log_prob(a)

    pi = jax.nn.softmax(logits_fn(params))

    def per_a(a):
        ga = jax.grad(logp, argnums=0)(theta_flat, a)
        return pi[a] * jnp.outer(ga, ga)

    F = jnp.sum(jnp.stack([per_a(a) for a in range(n_actions)]), axis=0)
    F = F + damping * jnp.eye(n)
    return jnp.linalg.solve(F, g_flat)
