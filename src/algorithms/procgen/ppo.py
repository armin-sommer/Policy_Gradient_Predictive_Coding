"""PPO (full): Pipeline-B PPO behind the rollout_fn/policy seam.

Preserves the canonical PPO machinery from src/backprop_algorithms/ppo.py:
GAE, value head, clipped surrogate, value-loss coefficient, entropy bonus,
advantage normalisation, Adam + grad clipping, K epochs * M minibatches
per update iteration.

Requires policy.value_apply (actor-critic). For toy actor-only policies
use algorithms.mdp_experiments.ppo.

Per-iteration data flow:
    rollout_fn(params, key) -> Rollout      (T, N, ...)
    -> compute values + bootstrap
    -> GAE                                  (T, N)
    -> flatten to (T*N, ...)
    -> for epoch in range(epochs):
           shuffle, split into M minibatches, Adam update on each
"""
import distrax
import jax
import jax.numpy as jnp
import optax

from algorithms.procgen._gae import compute_gae


def _ppo_loss(params, batch, *, policy, clip, vf_coef, ent_coef):
    """Single PPO loss on a flat (B,) batch.

    params is the actor-critic dict {"actor": ..., "critic": ...}. The
    policy provides apply(actor_params, obs) -> logits and
    value_apply(critic_params, obs) -> value.
    """
    obs, actions, old_logp, advantages, vs_target = batch
    logits = policy.apply(params, obs)
    values = policy.value_apply(params, obs)

    new_logp = distrax.Categorical(logits=logits).log_prob(actions)
    log_ratio = new_logp - old_logp
    ratio = jnp.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    v_error = values - vs_target
    value_loss = 0.5 * jnp.mean(v_error * v_error)

    entropy = jnp.mean(distrax.Categorical(logits=logits).entropy())

    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": jax.lax.stop_gradient(approx_kl),
    }
    return total, metrics


def make_step(rollout_fn, policy, *,
              lr: float = 3e-4,
              clip: float = 0.2,
              epochs: int = 10,
              num_minibatches: int = 8,
              vf_coef: float = 0.5,
              ent_coef: float = 0.0,
              max_grad_norm: float = 0.5,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              normalize_advantage: bool = True,
              env_J=None):
    del env_J
    if policy.value_apply is None:
        raise ValueError("algorithms.procgen.ppo requires an actor-critic "
                         "policy (policy.value_apply is set)")

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )

    def init_opt_state_fn(params):
        return optimizer.init(params)

    loss_fn = lambda p, batch: _ppo_loss(
        p, batch, policy=policy, clip=clip, vf_coef=vf_coef, ent_coef=ent_coef)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def minibatch_update(carry, batch):
        params, opt_state = carry
        (_, metrics), grads = grad_fn(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), metrics

    def step(params, opt_state, key):
        k_roll, k_perm = jax.random.split(key)
        roll = rollout_fn(params, k_roll)
        T, N = roll.actions.shape
        obs_shape = roll.obs.shape[2:]  # preserve trailing obs dims (1D or 3D)

        # Compute on-policy values + bootstrap.
        flat_obs_T = roll.obs.reshape((T * N,) + obs_shape)
        values_T = policy.value_apply(params, flat_obs_T).reshape(T, N)
        bootstrap = policy.value_apply(params, roll.last_obs).reshape(N)

        vs, advantages = compute_gae(
            rewards=roll.rewards, values=values_T, dones=roll.dones,
            bootstrap_value=bootstrap, gamma=gamma, gae_lambda=gae_lambda)

        # Flatten the (T, N) leading axes; preserve obs trailing dims.
        B = T * N
        obs_flat = roll.obs.reshape((B,) + obs_shape)
        actions_flat = roll.actions.reshape(B)
        old_logp_flat = roll.logp_old.reshape(B)
        adv_flat = advantages.reshape(B)
        vs_flat = vs.reshape(B)

        if normalize_advantage:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        batch_size = B // num_minibatches

        def epoch_body(carry, k_epoch):
            params, opt_state = carry
            perm = jax.random.permutation(k_epoch, B)
            shuffled = (obs_flat[perm], actions_flat[perm],
                        old_logp_flat[perm], adv_flat[perm], vs_flat[perm])
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape(num_minibatches, batch_size, *x.shape[1:]),
                shuffled)
            (params, opt_state), mb_metrics = jax.lax.scan(
                minibatch_update, (params, opt_state), minibatches)
            return (params, opt_state), mb_metrics

        epoch_keys = jax.random.split(k_perm, epochs)
        (new_params, new_opt_state), all_metrics = jax.lax.scan(
            epoch_body, (params, opt_state), epoch_keys)

        mean_ep_return = roll.rewards.sum(axis=0).mean(axis=-1)
        metrics = {
            "mean_ep_return": mean_ep_return,
            "policy_loss": all_metrics["policy_loss"].mean(),
            "value_loss": all_metrics["value_loss"].mean(),
            "entropy": all_metrics["entropy"].mean(),
            "approx_kl": all_metrics["approx_kl"].mean(),
        }
        return new_params, new_opt_state, metrics

    return init_opt_state_fn, step
