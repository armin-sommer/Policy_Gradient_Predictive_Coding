"""REINFORCE (full): discounted returns with value baseline + entropy bonus.

Pipeline-B REINFORCE behind the rollout_fn/policy seam. Requires
policy.value_apply (actor-critic). For toy actor-only policies use
algorithms.mdp_experiments.reinforce.

Loss = -mean(logp * stop_grad(returns - baseline))
       + vf_coef * mean((returns - baseline)^2)
       - ent_coef * mean(entropy)
"""
import distrax
import jax
import jax.numpy as jnp
import optax

from rollout import discounted_returns


def make_step(rollout_fn, policy, *,
              lr: float = 3e-4,
              gamma: float = 0.99,
              ent_coef: float = 1e-4,
              vf_coef: float = 0.5,
              max_grad_norm: float = 0.5,
              env_J=None):
    del env_J
    if policy.value_apply is None:
        raise ValueError("algorithms.procgen.reinforce requires an "
                         "actor-critic policy (policy.value_apply is set)")

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )

    def init_opt_state_fn(params):
        return optimizer.init(params)

    def loss_fn(params, obs, actions, returns):
        logits = policy.apply(params, obs)
        values = policy.value_apply(params, obs)
        logp = distrax.Categorical(logits=logits).log_prob(actions)
        advantages = jax.lax.stop_gradient(returns - values)
        policy_loss = -jnp.mean(logp * advantages)
        value_loss = jnp.mean((returns - values) ** 2)
        entropy = jnp.mean(distrax.Categorical(logits=logits).entropy())
        total = policy_loss + vf_coef * value_loss - ent_coef * entropy
        return total, {"policy_loss": policy_loss,
                       "value_loss": value_loss,
                       "entropy": entropy}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def step(params, opt_state, key):
        roll = rollout_fn(params, key)
        T, N = roll.actions.shape
        obs_shape = roll.obs.shape[2:]
        B = T * N
        obs = roll.obs.reshape((B,) + obs_shape)
        actions = roll.actions.reshape(B)
        returns = discounted_returns(roll.rewards, gamma).reshape(B)

        (_, sub_metrics), grads = grad_fn(params, obs, actions, returns)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        mean_ep_return = roll.rewards.sum(axis=0).mean(axis=-1)
        metrics = {"mean_ep_return": mean_ep_return, **sub_metrics}
        return new_params, opt_state, metrics

    return init_opt_state_fn, step
