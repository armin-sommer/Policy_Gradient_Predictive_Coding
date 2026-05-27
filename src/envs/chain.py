"""5-state Mei-style chain MDP.

States 0..4. Actions: 0=left, 1=right. Deterministic transitions clamped
to the chain. Reward +1 on transition into state 4 (right end); 0 elsewhere.
State 4 is absorbing: once reached, the agent stays there with reward 0
until the rollout ends. Discount gamma=0.99 is used for the closed-form J
and the algorithm-side discounted returns.

J(theta) = V^pi(s=0) is computed by solving (I - gamma*P_pi) V = r_pi on
the full 5x5 state space with state 4 forced absorbing. NPG/TRPO toy
versions are bandit-only (their Fisher solve is built at a single fixed
observation), so chain experiments omit them from UPDATES.
"""
import jax
import jax.numpy as jnp

from envs._protocol import Env


OBS_DIM = 1
N_ACTIONS = 2
CHAIN_LEN = 5
GAMMA = 0.99
OBS = jnp.array([0.0])


def reset(key):
    del key
    state = jnp.int32(0)
    return state, jnp.float32(state)[None]


def step(state, action, key):
    del key
    is_done = state == CHAIN_LEN - 1
    delta = jnp.where(action == 1, 1, -1)
    next_state_raw = jnp.clip(state + delta, 0, CHAIN_LEN - 1)
    next_state = jnp.where(is_done, state, next_state_raw)
    reward = jnp.where((next_state == CHAIN_LEN - 1) & (~is_done), 1.0, 0.0)
    done = next_state == CHAIN_LEN - 1
    return next_state, jnp.float32(next_state)[None], reward, done


def J(params, *, apply):
    """V^pi(s=0) under the current policy, exact via linear solve."""
    states = jnp.arange(CHAIN_LEN)
    all_obs = states.astype(jnp.float32)[:, None]
    logits = apply(params, all_obs)
    pi = jax.nn.softmax(logits, axis=-1)

    s_left = jnp.clip(states - 1, 0, CHAIN_LEN - 1)
    s_right = jnp.clip(states + 1, 0, CHAIN_LEN - 1)
    next_state = jnp.stack([s_left, s_right], axis=1)
    rewards_sa = (next_state == CHAIN_LEN - 1).astype(jnp.float32)

    P_pi = jnp.zeros((CHAIN_LEN, CHAIN_LEN))
    for a in range(N_ACTIONS):
        P_pi = P_pi.at[jnp.arange(CHAIN_LEN), next_state[:, a]].add(pi[:, a])
    P_pi = P_pi.at[CHAIN_LEN - 1].set(0.0)
    P_pi = P_pi.at[CHAIN_LEN - 1, CHAIN_LEN - 1].set(1.0)

    r_pi = jnp.sum(pi * rewards_sa, axis=-1)
    r_pi = r_pi.at[CHAIN_LEN - 1].set(0.0)

    V = jnp.linalg.solve(jnp.eye(CHAIN_LEN) - GAMMA * P_pi, r_pi)
    return V[0]


ENV = Env(obs_dim=OBS_DIM, n_actions=N_ACTIONS,
          reset=reset, step=step, J=J)
