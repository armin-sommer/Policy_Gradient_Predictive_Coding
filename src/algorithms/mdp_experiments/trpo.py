"""TRPO-toy: natural gradient step with backtracking line search on KL.

The line search uses env_J directly (closed-form). Bandit-only: the Fisher
solve is built at a single fixed observation.
"""
import distrax
import jax
import jax.numpy as jnp

from algorithms._utils import flatten, unflatten, kl_categorical
from algorithms.mdp_experiments._fisher import fisher_inv_g
from rollout import discounted_returns


def make_step(rollout_fn, policy, *, env_J=None,
              delta: float = 0.01,
              backtrack_iters: int = 10, backtrack_decay: float = 0.5,
              gamma: float = 0.99,
              fisher_damping: float = 1e-1):
    """fisher_damping controls how much the Fisher solve amplifies noise in
    low-eigenvalue directions. At 1e-3 (NPG's default, where the gradient
    is noise-free), Fisher^-1 multiplies orthogonal-to-signal sample noise
    by ~1000x. At 1e-1 the amplification is ~10x, restoring TRPO's update
    direction to near-NPG alignment on the bandit (cosine ~0.6 -> ~0.95).
    """
    if env_J is None:
        raise ValueError("TRPO-toy requires env_J for the line search")
    logits_fn = policy.logits_fn
    apply = policy.apply
    n_actions = policy.n_actions

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

        nat_flat = fisher_inv_g(params, g,
                                logits_fn=logits_fn, n_actions=n_actions,
                                damping=fisher_damping)
        gTnat = jnp.dot(flatten(g), nat_flat)
        step_size = jnp.sqrt(2 * delta / (jnp.abs(gTnat) + 1e-12))
        p_old = jax.nn.softmax(logits_fn(params))
        theta0 = flatten(params)
        J0 = env_J(params, apply=apply)

        def body(carry, _):
            accepted, theta, frac = carry
            cand = theta0 - frac * step_size * nat_flat
            cand_params = unflatten(params, cand)
            p_new = jax.nn.softmax(logits_fn(cand_params))
            ok = ((kl_categorical(p_old, p_new) <= delta) &
                  (env_J(cand_params, apply=apply) >= J0))
            take = (~accepted) & ok
            new_theta = jnp.where(take, cand, theta)
            return (accepted | ok, new_theta, frac * backtrack_decay), None

        init = (jnp.array(False), theta0, jnp.array(1.0))
        (accepted, theta_out, _), _ = jax.lax.scan(
            body, init, None, length=backtrack_iters)
        theta_final = jnp.where(accepted, theta_out, theta0)
        new_params = unflatten(params, theta_final)
        mean_ep_return = roll.rewards.sum(axis=0).mean(axis=-1)
        return new_params, opt_state, {"mean_ep_return": mean_ep_return}

    return (lambda _: ()), step
