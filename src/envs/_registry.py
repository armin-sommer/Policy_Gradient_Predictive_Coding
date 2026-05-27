"""Env registry: string name -> Env.

Experiments name their env by string (ENV="bandit"|"chain") and the driver
resolves it here. Add new envs by importing the module and inserting it.
"""
from envs import bandit, chain


_REGISTRY = {
    "bandit": bandit,
    "chain": chain,
}


def make_env(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown env '{name}'. registered: {list(_REGISTRY)}")
    mod = _REGISTRY[name]
    return mod.ENV, getattr(mod, "OBS", None)
