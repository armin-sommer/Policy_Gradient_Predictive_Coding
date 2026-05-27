"""Procgen wrapper.

Procgen is not pure-JAX-compatible (the env is a C++ object with internal
state). It therefore does NOT fit the same Env(reset, step, J=None)
protocol that bandit and chain use — those rely on vmap+scan of pure
functions. Procgen training requires a host-Python rollout (see
rollout/procgen_rollout.py) and a hosted runner path.

make_procgen_env returns a stateful wrapper object that exposes the data
the rollout collector and policy factory need:
    .vec_env       the underlying ProcgenGym3Env (or compatible)
    .obs_shape     (H, W, C)
    .n_actions     int
    .num_envs      int
    .J             None (no closed-form return)

The driver chooses between `make_env` (pure JAX, registry) and
`make_procgen_env` (stateful) based on the experiment config.

This module guards the procgen-mirror import so the file is safe to load
even when procgen isn't installed (e.g. on a CPU-only dev box).
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class ProcgenEnvHandle:
    vec_env: Any
    obs_shape: tuple
    n_actions: int
    num_envs: int
    J: Any = None  # No closed-form expected return.


def make_procgen_env(env_name: str = "coinrun",
                     num_envs: int = 64,
                     num_train_levels: int = 200,
                     distribution_mode: str = "easy") -> ProcgenEnvHandle:
    try:
        from procgen import ProcgenGym3Env
    except ImportError as e:
        raise ImportError(
            "procgen-mirror is not installed. Install with: "
            "pip install -e '.[gpu]'  (or pip install procgen-mirror)"
        ) from e

    vec_env = ProcgenGym3Env(
        num=num_envs,
        env_name=env_name,
        start_level=0,
        num_levels=num_train_levels,
        distribution_mode=distribution_mode,
    )
    # Procgen's discrete action space is fixed at 15 across all games.
    n_actions = 15
    obs_shape = (64, 64, 3)
    return ProcgenEnvHandle(vec_env=vec_env, obs_shape=obs_shape,
                            n_actions=n_actions, num_envs=num_envs)
