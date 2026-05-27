"""PPO-toy: clipped surrogate, K inner gradient steps on the same batch."""
import distrax
import jax
import jax.numpy as jnp


def make_step(env, policy, *, lr: float, clip: float = 0.2,
              epochs: int = 4, batch_size: int = 64):
    logits_fn = policy.logits_fn
    sample = env.sample_batch

    def step(params, key):
        _, actions, rewards = sample(params, key, logits_fn=logits_fn,
                                     batch_size=batch_size)
        old_logp = distrax.Categorical(logits=logits_fn(params)).log_prob(actions)

        def loss(p):
            logp = distrax.Categorical(logits=logits_fn(p)).log_prob(actions)
            ratio = jnp.exp(logp - old_logp)
            unclipped = ratio * rewards
            clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * rewards
            return -jnp.mean(jnp.minimum(unclipped, clipped))

        def body(p, _):
            g = jax.grad(loss)(p)
            return jax.tree_util.tree_map(lambda x, dx: x - lr * dx, p, g), None
        new_params, _ = jax.lax.scan(body, params, None, length=epochs)
        return new_params

    return step
