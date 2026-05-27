"""PPO-toy: clipped surrogate, K inner gradient steps on the same batch."""
import distrax
import jax
import jax.numpy as jnp

from rollout import discounted_returns


def make_step(rollout_fn, policy, *, lr: float, clip: float = 0.2,
              epochs: int = 4, gamma: float = 0.99, env_J=None):
    del env_J
    apply = policy.apply

    def step(params, opt_state, key):
        roll = rollout_fn(params, key)
        T, N = roll.actions.shape
        obs = roll.obs.reshape(T * N, -1)
        actions = roll.actions.reshape(T * N)
        returns = discounted_returns(roll.rewards, gamma).reshape(T * N)
        old_logp = roll.logp_old.reshape(T * N)

        def loss(p):
            logp = distrax.Categorical(logits=apply(p, obs)).log_prob(actions)
            ratio = jnp.exp(logp - old_logp)
            unclipped = ratio * returns
            clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * returns
            return -jnp.mean(jnp.minimum(unclipped, clipped))

        def body(p, _):
            g = jax.grad(loss)(p)
            return jax.tree_util.tree_map(lambda x, dx: x - lr * dx, p, g), None
        new_params, _ = jax.lax.scan(body, params, None, length=epochs)
        mean_ep_return = roll.rewards.sum(axis=0).mean(axis=-1)
        return new_params, opt_state, {"mean_ep_return": mean_ep_return}

    return (lambda _: ()), step
