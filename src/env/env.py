from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import jax.numpy as jnp
try:
    from procgen import ProcgenGym3Env
except ImportError:
    ProcgenGym3Env = None

from utils.utils import EnvConfig


@dataclass
class State:
    """Mimics the PolicyGradientsJax State interface over Procgen."""
    obs: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: List[Dict[str, Any]] = field(default_factory=list)


class ActionSpace:
    """Minimal discrete action space descriptor."""
    def __init__(self, n: int):
        self.n = n


class ProcgenVecEnv:
    """Wraps ProcgenGym3Env to match the PolicyGradientsJax VecEnv API.

    Provides: seed(), reset(), step(), close(), action_space, normalize_obs().
    """

    def __init__(self, cfg: EnvConfig):
        if ProcgenGym3Env is None:
            raise ImportError("procgen-mirror is required but not installed")
        self._cfg = cfg
        self._env = ProcgenGym3Env(
            num=cfg.num_envs,
            env_name=cfg.env_name,
            start_level=0,
            num_levels=cfg.num_train_levels,
            distribution_mode=cfg.distribution_mode,
        )
        self.action_space = ActionSpace(n=15)
        self.num_envs = cfg.num_envs
        self._prev_first = None

    def seed(self, seed: int):
        pass  # Procgen seeds via start_level/num_levels at construction

    def reset(self) -> State:
        reward, obs, first = self._env.observe()
        self._prev_first = np.asarray(first)
        return State(
            obs=obs["rgb"],
            reward=np.zeros(self.num_envs, dtype=np.float32),
            done=np.zeros(self.num_envs, dtype=np.float32),
        )

    def step(self, actions) -> State:
        if isinstance(actions, jnp.ndarray):
            actions = np.asarray(actions)
        self._env.act(actions.astype(np.int32))
        reward, obs, first = self._env.observe()
        first = np.asarray(first)
        done = first.astype(np.float32)
        # Procgen auto-resets, so "first" on current step means the *previous*
        # episode ended. There is no separate truncation signal.
        info = [{"truncation": 0.0} for _ in range(self.num_envs)]
        return State(
            obs=obs["rgb"],
            reward=np.asarray(reward, dtype=np.float32),
            done=done,
            info=info,
        )

    def normalize_obs(self, obs):
        return obs / 255.0

    def close(self):
        pass  # ProcgenGym3Env has no explicit close


class ProcgenEvalEnv(ProcgenVecEnv):
    """Evaluation wrapper that tracks episode returns and lengths."""

    def __init__(self, cfg: EnvConfig):
        eval_cfg = EnvConfig(
            env_name=cfg.env_name,
            num_envs=cfg.num_envs,
            num_train_levels=cfg.num_train_levels,
            num_test_levels=cfg.num_test_levels,
            distribution_mode=cfg.distribution_mode,
        )
        super().__init__(eval_cfg)
        self._ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returns: List[float] = []
        self.ep_lengths: List[int] = []

    def reset(self) -> State:
        self._ep_returns[:] = 0.0
        self._ep_lengths[:] = 0
        return super().reset()

    def step(self, actions) -> State:
        state = super().step(actions)
        self._ep_returns += state.reward
        self._ep_lengths += 1
        for i in range(self.num_envs):
            if state.done[i]:
                self.returns.append(float(self._ep_returns[i]))
                self.ep_lengths.append(int(self._ep_lengths[i]))
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0
        return state

    def evaluate(self):
        ret = list(self.returns)
        lens = list(self.ep_lengths)
        self.returns.clear()
        self.ep_lengths.clear()
        return ret, lens


# --- Convenience functions (original API, still usable) ---

def make_envs(cfg: EnvConfig):
    """Build train/test Procgen envs with disjoint level seed ranges."""
    if ProcgenGym3Env is None:
        raise ImportError("procgen-mirror is required but not installed")
    train_env = ProcgenGym3Env(
        num=cfg.num_envs,
        env_name=cfg.env_name,
        start_level=0,
        num_levels=cfg.num_train_levels,
        distribution_mode=cfg.distribution_mode,
    )
    test_env = ProcgenGym3Env(
        num=cfg.num_envs,
        env_name=cfg.env_name,
        start_level=cfg.num_train_levels,
        num_levels=0,
        distribution_mode=cfg.distribution_mode,
    )
    return train_env, test_env


def observe(env):
    """Pull current observation, reward, and 'is first step' flag."""
    reward, obs, first = env.observe()
    return obs["rgb"], np.asarray(reward, dtype=np.float32), np.asarray(first)


def step(env, actions):
    """Apply a batch of actions; Procgen auto-resets internally."""
    if isinstance(actions, jnp.ndarray):
        actions = np.asarray(actions)
    env.act(actions.astype(np.int32))
    return observe(env)
