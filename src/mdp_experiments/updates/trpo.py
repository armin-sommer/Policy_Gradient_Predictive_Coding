"""TRPO-toy: natural gradient step with backtracking line search on KL."""
import distrax
import jax
import jax.numpy as jnp

from mdp_experiments._base import flatten, unflatten, kl_categorical
from mdp_experiments.updates._fisher import fisher_inv_g


def make_step(env, policy, *, delta: float = 0.01,
              backtrack_iters: int = 10, backtrack_decay: float = 0.5,
              batch_size: int = 64):
    logits_fn = policy.logits_fn
    n_actions = policy.n_actions
    sample = env.sample_batch
    J = env.J

    def step(params, key):
        _, actions, rewards = sample(params, key, logits_fn=logits_fn,
                                     batch_size=batch_size)
        # PG gradient (same expression as REINFORCE).
        def loss(p):
            logp = distrax.Categorical(logits=logits_fn(p)).log_prob(actions)
            return -jnp.mean(logp * rewards)
        g = jax.grad(loss)(params)

        nat_flat = fisher_inv_g(params, g,
                                logits_fn=logits_fn, n_actions=n_actions)
        gTnat = jnp.dot(flatten(g), nat_flat)
        step_size = jnp.sqrt(2 * delta / (jnp.abs(gTnat) + 1e-12))
        p_old = jax.nn.softmax(logits_fn(params))
        theta0 = flatten(params)
        J0 = J(params, logits_fn=logits_fn)

        def body(carry, _):
            accepted, theta, frac = carry
            cand = theta0 - frac * step_size * nat_flat
            cand_params = unflatten(params, cand)
            p_new = jax.nn.softmax(logits_fn(cand_params))
            ok = ((kl_categorical(p_old, p_new) <= delta) &
                  (J(cand_params, logits_fn=logits_fn) >= J0))
            take = (~accepted) & ok
            new_theta = jnp.where(take, cand, theta)
            return (accepted | ok, new_theta, frac * backtrack_decay), None

        init = (jnp.array(False), theta0, jnp.array(1.0))
        (accepted, theta_out, _), _ = jax.lax.scan(
            body, init, None, length=backtrack_iters)
        theta_final = jnp.where(accepted, theta_out, theta0)
        return unflatten(params, theta_final)

    return step
