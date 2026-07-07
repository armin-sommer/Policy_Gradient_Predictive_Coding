"""Return computation for on-policy PC rollouts."""

import numpy as np


def compute_mc_returns(rewards, dones, gamma):
    """Monte Carlo returns over a (T, N) rollout, flattened in time-major order."""
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    if rewards.ndim == 1:
        return rewards
    t_steps, n_envs = rewards.shape
    running = np.zeros(n_envs, dtype=np.float32)
    returns = np.zeros_like(rewards)
    for t in range(t_steps - 1, -1, -1):
        running = rewards[t] + gamma * running * (1.0 - dones[t])
        returns[t] = running
    return returns.reshape(-1)
