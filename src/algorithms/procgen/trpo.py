"""TRPO (full): conjugate-gradient natural step + KL backtracking line search.

Pipeline-B TRPO behind the rollout_fn/policy seam. Two optimisers:
  - actor: trust-region step via Fisher-vector-products + CG + line search.
  - critic: standard Adam SGD on the GAE value targets.

Requires policy.value_apply (actor-critic). Operates on flat actor params
internally (CG and HVP are most natural in flat space); critic stays a
pytree under optax.
"""
import distrax
import jax
import jax.numpy as jnp
import optax

from algorithms.procgen._gae import compute_gae


def _flat_params(actor_params):
    leaves, _ = jax.tree_util.tree_flatten(actor_params)
    return jnp.concatenate([l.ravel() for l in leaves])


def _unflat_like(actor_params, flat):
    leaves, treedef = jax.tree_util.tree_flatten(actor_params)
    out, idx = [], 0
    for l in leaves:
        sz = l.size
        out.append(flat[idx:idx + sz].reshape(l.shape))
        idx += sz
    return jax.tree_util.tree_unflatten(treedef, out)


def make_step(rollout_fn, policy, *,
              target_kl: float = 0.01,
              cg_iters: int = 10,
              cg_damping: float = 0.1,
              line_search_iters: int = 10,
              line_search_decay: float = 0.8,
              value_lr: float = 3e-4,
              value_epochs: int = 5,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              max_grad_norm: float = 0.5,
              env_J=None):
    del env_J
    if policy.value_apply is None:
        raise ValueError("algorithms.procgen.trpo requires an actor-critic policy")

    value_opt = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(value_lr),
    )

    def init_opt_state_fn(params):
        return value_opt.init(params)

    def surrogate(actor_params, full_params, obs, actions, advantages, old_logp):
        replaced = {**full_params, "actor": actor_params}
        logits = policy.apply(replaced, obs)
        new_logp = distrax.Categorical(logits=logits).log_prob(actions)
        ratio = jnp.exp(new_logp - old_logp)
        return jnp.mean(ratio * advantages)

    def mean_kl(actor_params, full_params, obs, ref_logits):
        replaced = {**full_params, "actor": actor_params}
        new_logits = policy.apply(replaced, obs)
        ref = distrax.Categorical(logits=jax.lax.stop_gradient(ref_logits))
        new = distrax.Categorical(logits=new_logits)
        return jnp.mean(ref.kl_divergence(new))

    def fisher_vector_product(v_flat, actor_params, full_params, obs, ref_logits):
        # Hv = d/dp [ grad_kl(p) . v ] = Fisher . v.
        def kl_of_flat(p_flat):
            p = _unflat_like(actor_params, p_flat)
            return mean_kl(p, full_params, obs, ref_logits)

        def grad_dot_v(p_flat):
            grad_kl = jax.grad(kl_of_flat)(p_flat)
            return jnp.dot(grad_kl, v_flat)

        hvp = jax.grad(grad_dot_v)(_flat_params(actor_params))
        return hvp + cg_damping * v_flat

    def conjugate_gradient(Av_fn, b):
        x = jnp.zeros_like(b)
        r = b
        p = b
        rdotr = jnp.dot(r, r)

        def body(carry, _):
            x, r, p, rdotr = carry
            Ap = Av_fn(p)
            alpha = rdotr / (jnp.dot(p, Ap) + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            new_rdotr = jnp.dot(r, r)
            beta = new_rdotr / (rdotr + 1e-12)
            p = r + beta * p
            return (x, r, p, new_rdotr), None

        (x, _, _, _), _ = jax.lax.scan(body, (x, r, p, rdotr), None,
                                       length=cg_iters)
        return x

    def value_loss(params, obs, vs_target):
        v = policy.value_apply(params, obs)
        return jnp.mean((v - vs_target) ** 2)

    value_grad_fn = jax.value_and_grad(value_loss)

    def step(params, opt_state, key):
        roll = rollout_fn(params, key)
        T, N = roll.actions.shape
        obs_shape = roll.obs.shape[2:]
        B = T * N

        obs = roll.obs.reshape((B,) + obs_shape)
        actions = roll.actions.reshape(B)
        old_logp = roll.logp_old.reshape(B)

        values_T = policy.value_apply(params, obs).reshape(T, N)
        bootstrap = policy.value_apply(params, roll.last_obs).reshape(N)

        vs, advantages = compute_gae(
            rewards=roll.rewards, values=values_T, dones=roll.dones,
            bootstrap_value=bootstrap, gamma=gamma, gae_lambda=gae_lambda)

        adv_flat = advantages.reshape(B)
        vs_flat = vs.reshape(B)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        actor_params = params["actor"]
        ref_logits = jax.lax.stop_gradient(policy.apply(params, obs))

        # Natural gradient direction.
        g = _flat_params(jax.grad(lambda p: -surrogate(
            p, params, obs, actions, adv_flat, old_logp))(actor_params))
        Av = lambda v: fisher_vector_product(
            v, actor_params, params, obs, ref_logits)
        search_dir = conjugate_gradient(Av, -g)

        # Trust-region step size.
        shs = jnp.dot(search_dir, Av(search_dir))
        step_size = jnp.sqrt(2.0 * target_kl / (jnp.abs(shs) + 1e-12))

        theta0 = _flat_params(actor_params)
        L0 = surrogate(actor_params, params, obs, actions, adv_flat, old_logp)

        def line_search(carry, _):
            accepted, theta, coeff = carry
            cand_flat = theta0 + coeff * step_size * search_dir
            cand_actor = _unflat_like(actor_params, cand_flat)
            kl = mean_kl(cand_actor, params, obs, ref_logits)
            L_new = surrogate(cand_actor, params, obs, actions,
                              adv_flat, old_logp)
            ok = (kl <= target_kl) & (L_new >= L0)
            take = (~accepted) & ok
            new_theta = jnp.where(take, cand_flat, theta)
            return (accepted | ok, new_theta, coeff * line_search_decay), None

        (accepted, theta_out, _), _ = jax.lax.scan(
            line_search, (jnp.array(False), theta0, jnp.array(1.0)),
            None, length=line_search_iters)
        new_actor_flat = jnp.where(accepted, theta_out, theta0)
        new_actor = _unflat_like(actor_params, new_actor_flat)

        # Critic update: K Adam steps on MSE(returns_target - V).
        def critic_step(carry, _):
            full_params, opt_state = carry
            _, grads = value_grad_fn(full_params, obs, vs_flat)
            updates, opt_state = value_opt.update(grads, opt_state, full_params)
            full_params = optax.apply_updates(full_params, updates)
            return (full_params, opt_state), None

        params_after_actor = {**params, "actor": new_actor}
        (params_final, opt_state_final), _ = jax.lax.scan(
            critic_step, (params_after_actor, opt_state), None,
            length=value_epochs)

        mean_ep_return = roll.rewards.sum(axis=0).mean(axis=-1)
        metrics = {
            "mean_ep_return": mean_ep_return,
            "line_search_accepted": accepted.astype(jnp.float32),
            "surrogate_pre": L0,
        }
        return params_final, opt_state_final, metrics

    return init_opt_state_fn, step
