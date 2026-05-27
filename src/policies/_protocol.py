"""Policy protocol.

A Policy bundles network init/apply for the actor (logits) plus a
diagnostic logits_fn that evaluates the actor at a single canonical
observation (used by NPG's Fisher solve and by the runner's pi_t /
pi_{t-1} KL diagnostic).

Actor-critic policies (impala_cnn etc.) additionally provide value_apply.
Toy policies leave value_apply=None; procgen-tier algorithms
(algorithms/procgen/) require it and raise otherwise.

Convention for actor-critic policies: params is a dict with "actor" and
"critic" sub-keys. apply / logits_fn read params["actor"]; value_apply
reads params["critic"]. Toy policies use the entire params dict as actor
weights.
"""
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp


class Policy(NamedTuple):
    init: Callable[[jax.Array], dict]                       # rng_key -> params
    logits_fn: Callable[[dict], jnp.ndarray]                # params -> logits at fixed_obs
    apply: Callable[[dict, jnp.ndarray], jnp.ndarray]       # params, x -> logits (batched)
    fixed_obs: jnp.ndarray                                  # shape (obs_dim,)
    obs_dim: int
    hidden: int
    n_actions: int
    value_apply: Optional[Callable[[dict, jnp.ndarray], jnp.ndarray]] = None
