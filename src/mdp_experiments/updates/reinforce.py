"""Vanilla policy gradient (REINFORCE) for small categorical policies."""
import distrax
import jax


def make_step(env, policy, *, lr: float, batch_size: int = 64):
    logits_fn = policy.logits_fn
    sample = env.sample_batch

    def grad_fn(params, actions, rewards):
        def loss(p):
            logp = distrax.Categorical(logits=logits_fn(p)).log_prob(actions)
            return -jax.numpy.mean(logp * rewards)
        return jax.grad(loss)(params)

    def step(params, key):
        _, actions, rewards = sample(params, key, logits_fn=logits_fn,
                                     batch_size=batch_size)
        g = grad_fn(params, actions, rewards)
        return jax.tree_util.tree_map(lambda p, dp: p - lr * dp, params, g)

    return step
