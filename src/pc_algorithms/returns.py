"""Return / advantage computation for on-policy PC rollouts."""

import numpy as np


def compute_mc_returns(rewards, dones, gamma):
    """Monte Carlo returns over a (T, N) rollout, flattened in time-major order.

    Returns are truncated at the rollout boundary (no bootstrap), so the
    rollout must be long enough to cover the effective horizon (~1/(1-gamma)).
    """
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


def compute_gae(rewards, dones, values, next_values, gamma, lam):
    """GAE(lambda) over a (T, N) rollout.

    Bootstraps through the rollout boundary via next_values, so short rollouts
    still see the full horizon. Returns flattened (advantages, value_targets),
    where value_targets = advantages + values (the lambda-returns).
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    deltas = rewards + gamma * (1.0 - dones) * next_values - values
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = deltas[t] + gamma * lam * (1.0 - dones[t]) * running
        advantages[t] = running
    value_targets = advantages + values
    return advantages.reshape(-1), value_targets.reshape(-1)
