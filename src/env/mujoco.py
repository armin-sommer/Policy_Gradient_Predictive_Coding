"""MuJoCo continuous-control VecEnv via Brax (mjx backend).

Wraps a Brax environment to match the ProcgenVecEnv/BanditVecEnv interface so the
backprop algorithms run unchanged. Brax handles vmap, episode termination, and
auto-reset internally (envs.create(..., batch_size=N, auto_reset=True)); this
wrapper jits reset/step and crosses the JAX->NumPy boundary each step, the same
eager pattern Procgen uses.
"""

from typing import List

import numpy as np
import jax
import jax.numpy as jnp

try:
    from brax import envs as brax_envs
except ImportError:
    brax_envs = None

from env.env import State, ActionSpace
from utils.utils import EnvConfig


class MujocoVecEnv:
    """Vectorized Brax MuJoCo env matching the ProcgenVecEnv interface."""

    def __init__(self, cfg: EnvConfig):
        if brax_envs is None:
            raise ImportError("brax is required but not installed")
        self._cfg = cfg
        self._env = brax_envs.create(
            cfg.env_name,
            episode_length=cfg.episode_length,
            action_repeat=1,
            auto_reset=True,
            batch_size=cfg.num_envs,
            backend="mjx",
        )
        self._reset_fn = jax.jit(self._env.reset)
        self._step_fn = jax.jit(self._env.step)
        # Continuous action space: n is the action dimension, plus a marker the
        # algorithms use to pick a Gaussian (vs softmax) policy.
        self.action_space = ActionSpace(n=int(self._env.action_size))
        self.action_space.continuous = True
        self.observation_size = int(self._env.observation_size)
        self.num_envs = cfg.num_envs
        self._key = jax.random.PRNGKey(0)
        self._state = None

    def seed(self, seed: int):
        self._key = jax.random.PRNGKey(int(seed))

    def reset(self) -> State:
        self._key, k = jax.random.split(self._key)
        self._state = self._reset_fn(k)  # VmapWrapper splits k by batch_size
        return State(
            obs=np.asarray(self._state.obs, dtype=np.float32),
            reward=np.zeros(self.num_envs, dtype=np.float32),
            done=np.zeros(self.num_envs, dtype=np.float32),
        )

    def step(self, actions) -> State:
        actions = jnp.asarray(actions, dtype=jnp.float32)
        self._state = self._step_fn(self._state, actions)
        trunc = np.asarray(self._state.info["truncation"], dtype=np.float32)
        info: List[dict] = [{"truncation": float(trunc[i])} for i in range(self.num_envs)]
        return State(
            obs=np.asarray(self._state.obs, dtype=np.float32),
            reward=np.asarray(self._state.reward, dtype=np.float32),
            done=np.asarray(self._state.done, dtype=np.float32),
            info=info,
        )

    def normalize_obs(self, obs):
        return obs

    def close(self):
        pass


class MujocoEvalEnv(MujocoVecEnv):
    """Evaluation wrapper that tracks per-episode returns and lengths."""

    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)
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
