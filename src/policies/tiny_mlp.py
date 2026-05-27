"""Tiny Flax MLP policy factory used by all toy/MDP experiments.

Architecture: obs_dim -> tanh(hidden) -> n_actions (logits). The output is
a discrete logits vector; sample with distrax.Categorical at the call site.
Layer names 'h' and 'out' match the keys expected by src/pc_algorithms/pcpg.py.
"""
import jax.numpy as jnp
from flax import linen as nn

from policies._protocol import Policy


class TinyPolicy(nn.Module):
    hidden: int
    n_actions: int

    @nn.compact
    def __call__(self, x):
        h = nn.tanh(nn.Dense(self.hidden, name="h")(x))
        return nn.Dense(self.n_actions, name="out")(h)


def make_tiny_policy(obs_dim: int, hidden: int, n_actions: int,
                     fixed_obs: jnp.ndarray | None = None) -> Policy:
    module = TinyPolicy(hidden=hidden, n_actions=n_actions)
    dummy = jnp.zeros((obs_dim,))

    def init(key):
        return module.init(key, dummy)

    def apply(params, x):
        return module.apply(params, x)

    if fixed_obs is None:
        fixed_obs = dummy

    def logits_fn(params):
        return module.apply(params, fixed_obs)

    return Policy(init=init, logits_fn=logits_fn, apply=apply,
                  fixed_obs=fixed_obs,
                  obs_dim=obs_dim, hidden=hidden, n_actions=n_actions)
