"""Multi-armed bandit environment with the PolicyGradientsJax VecEnv API.

Single state, K arms (default 2). Every step is a full episode (done=1).
The observation is a constant 1-dim feature so the policy reduces to a
state-independent softmax over logits.
"""

from typing import List

import numpy as np
import jax.numpy as jnp

from env.env import State, ActionSpace
from utils.utils import EnvConfig


class BanditVecEnv:
    """Vectorized K-armed bandit matching the ProcgenVecEnv interface."""

    def __init__(self, cfg: EnvConfig):
        self._cfg = cfg
        self.arm_means = np.asarray(cfg.arm_means, dtype=np.float32)
        self.deterministic_rewards = bool(cfg.deterministic_rewards)
        self.action_space = ActionSpace(n=len(self.arm_means))
        self.num_envs = cfg.num_envs
        self._rng = np.random.default_rng(0)

    def seed(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def _obs(self) -> np.ndarray:
        return np.ones((self.num_envs, 1), dtype=np.float32)

    def reset(self) -> State:
        return State(
            obs=self._obs(),
            reward=np.zeros(self.num_envs, dtype=np.float32),
            done=np.zeros(self.num_envs, dtype=np.float32),
        )

    def step(self, actions) -> State:
        if isinstance(actions, jnp.ndarray):
            actions = np.asarray(actions)
        actions = actions.astype(np.int32)
        means = self.arm_means[actions]
        if self.deterministic_rewards:
            reward = means
        else:
            reward = self._rng.binomial(1, np.clip(means, 0.0, 1.0)).astype(np.float32)
        # one-step episodes: every pull terminates
        done = np.ones(self.num_envs, dtype=np.float32)
        info: List[dict] = [{"truncation": 0.0} for _ in range(self.num_envs)]
        return State(
            obs=self._obs(),
            reward=reward.astype(np.float32),
            done=done,
            info=info,
        )

    def normalize_obs(self, obs):
        return obs

    def close(self):
        pass


class BanditEvalEnv(BanditVecEnv):
    """Evaluation wrapper tracking per-episode returns (1 step = 1 episode)."""

    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)
        self.returns: List[float] = []
        self.ep_lengths: List[int] = []

    def reset(self) -> State:
        return super().reset()

    def step(self, actions) -> State:
        state = super().step(actions)
        for i in range(self.num_envs):
            self.returns.append(float(state.reward[i]))
            self.ep_lengths.append(1)
        return state

    def evaluate(self):
        ret = list(self.returns)
        lens = list(self.ep_lengths)
        self.returns.clear()
        self.ep_lengths.clear()
        return ret, lens
