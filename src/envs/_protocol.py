"""Env protocol.

Every env (toy bandit, chain MDP, eventual Procgen) exposes the same five
fields. Multi-state MDPs and stateless 1-step envs alike implement reset
and step. The optional closed-form J(params, *, apply) is only present
for toy MDPs where expected return is analytic; NPG and TRPO diagnostics
use it, sample-based algorithms ignore it.
"""
from typing import Any, Callable, NamedTuple, Optional


class Env(NamedTuple):
    obs_dim: int
    n_actions: int
    # reset(key) -> (state, obs)
    reset: Callable[..., Any]
    # step(state, action, key) -> (state, next_obs, reward, done)
    step: Callable[..., Any]
    # Optional closed-form expected return J(params, *, apply) -> scalar.
    J: Optional[Callable[..., Any]] = None
