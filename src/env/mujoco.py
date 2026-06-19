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
        # running obs mean/std (Welford) — TRPO's natural gradient needs normalized
        # inputs or its Fisher is ill-conditioned and the policy never moves.
        self._obs_mean = np.zeros(self.observation_size, dtype=np.float64)
        self._obs_var = np.ones(self.observation_size, dtype=np.float64)
        self._obs_count = 1e-4
        # running discounted return + its std, for SOTA reward normalization
        self._ret = np.zeros(self.num_envs, dtype=np.float64)
        self._ret_var = 1.0
        self._ret_count = 1e-4

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

    def normalize_obs(self, obs, update=True):
        """Normalize with a running mean/std (parallel Welford update).

        `normalize_obs` is stateful: each call with update=True advances the
        running stats. The rollout must therefore normalize each raw observation
        exactly once (and reuse the result), or the stored obs won't match the
        obs the policy acted on -> on-policy importance ratio != 1, which breaks
        TRPO's natural gradient. Pass update=False to apply the current stats
        without advancing them (next_observation, eval).
        """
        obs = np.asarray(obs, dtype=np.float32)
        if update:
            batch_mean = obs.mean(axis=0)
            batch_var = obs.var(axis=0)
            n = obs.shape[0]
            delta = batch_mean - self._obs_mean
            tot = self._obs_count + n
            self._obs_mean += delta * n / tot
            m2 = self._obs_var * self._obs_count + batch_var * n + delta ** 2 * self._obs_count * n / tot
            self._obs_var = m2 / tot
            self._obs_count = tot
        return np.clip((obs - self._obs_mean) / np.sqrt(self._obs_var + 1e-8),
                       -10.0, 10.0).astype(np.float32)

    def normalize_reward(self, reward, done, gamma):
        """SOTA reward scaling: divide reward by the running std of the
        discounted return (no mean subtraction), clip +/-10. Stateful and
        TRAINING ONLY -- never call on the eval env, whose returns must stay raw.
        """
        reward = np.asarray(reward, dtype=np.float64)
        done = np.asarray(done).astype(bool)
        self._ret = self._ret * gamma + reward
        n = self._ret.shape[0]
        batch_mean = self._ret.mean()
        batch_var = self._ret.var()
        tot = self._ret_count + n
        # running variance of the returns (mean tracked only to update var)
        delta = batch_mean - getattr(self, "_ret_mean", 0.0)
        self._ret_mean = getattr(self, "_ret_mean", 0.0) + delta * n / tot
        m2 = self._ret_var * self._ret_count + batch_var * n + delta ** 2 * self._ret_count * n / tot
        self._ret_var = m2 / tot
        self._ret_count = tot
        self._ret[done] = 0.0  # reset the discounted return at episode boundaries
        return np.clip(reward / np.sqrt(self._ret_var + 1e-8),
                       -10.0, 10.0).astype(np.float32)

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
