"""Vanilla policy gradient (REINFORCE), env-agnostic via rollout_fn."""
import distrax
import jax
import jax.numpy as jnp

from rollout import discounted_returns


def make_step(rollout_fn, policy, *, lr: float, gamma: float = 0.99,
              env_J=None):
    del env_J  # unused
    apply = policy.apply

    def step(params, opt_state, key):
        roll = rollout_fn(params, key)
        T, N = roll.actions.shape
        obs = roll.obs.reshape(T * N, -1)
        actions = roll.actions.reshape(T * N)
        returns = discounted_returns(roll.rewards, gamma).reshape(T * N)

        def loss(p):
            logp = distrax.Categorical(logits=apply(p, obs)).log_prob(actions)
            return -jnp.mean(logp * returns)
        g = jax.grad(loss)(params)
        new_params = jax.tree_util.tree_map(lambda p, dp: p - lr * dp, params, g)
        mean_ep_return = roll.rewards.sum(axis=0).mean(axis=-1)
        return new_params, opt_state, {"mean_ep_return": mean_ep_return}

    return (lambda _: ()), step
