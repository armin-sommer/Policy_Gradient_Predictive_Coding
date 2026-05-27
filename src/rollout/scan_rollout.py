"""Pure-JAX rollout collector for vmappable envs (bandit, chain).

collect_rollout(params, key, *, env, policy, T, N) -> Rollout

Vectorises N parallel envs via vmap, scans T steps. Algorithms consume the
Rollout and never see env.step directly. For Procgen later, a separate
procgen_rollout.py keeps the same signature and Rollout shape.
"""
import distrax
import jax

from rollout._protocol import Rollout


def collect_rollout(params, key, *, env, policy, T: int, N: int) -> Rollout:
    k_reset, k_scan = jax.random.split(key)
    reset_keys = jax.random.split(k_reset, N)
    init_state, init_obs = jax.vmap(env.reset)(reset_keys)

    def one_step(carry, k):
        state, obs = carry
        k_act, k_env = jax.random.split(k)
        logits = policy.apply(params, obs)                       # (N, n_actions)
        dist = distrax.Categorical(logits=logits)
        actions = dist.sample(seed=k_act)                        # (N,)
        logp = dist.log_prob(actions)                            # (N,)

        env_keys = jax.random.split(k_env, N)
        next_state, next_obs, reward, done = jax.vmap(env.step)(
            state, actions, env_keys)

        return (next_state, next_obs), (obs, actions, reward, done, logp)

    step_keys = jax.random.split(k_scan, T)
    (_, last_obs), (obs_T, act_T, rew_T, done_T, logp_T) = jax.lax.scan(
        one_step, (init_state, init_obs), step_keys)

    return Rollout(obs=obs_T, actions=act_T, rewards=rew_T,
                   dones=done_T, logp_old=logp_T, last_obs=last_obs)
