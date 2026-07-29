"""Gaussian-likelihood PC energy for the policy — the alternative to `natural_target`.

Two ways to get the natural-gradient direction into a PC update:

  A. TARGET route (`natural_target=True`, the current default path). Build the
     target by hand as mu + ts*A*(z-mu), i.e. pre-multiply the score by the
     inverse Fisher, then let PC hit it with a plain squared-error energy.
     The geometry is supplied BEFORE PC runs.

  B. ENERGY route (this module). Leave the target alone and make PC's output
     energy the policy's own negative log-likelihood, so sigma enters the
     objective PC minimises. The geometry is a property of the energy.

Only B tests the claim that PC controls the geometry implicitly; A supplies it
explicitly. `jpc` has no Gaussian likelihood loss (only "mse" and "ce"), so the
PC step is reimplemented here, mirroring jpc's structure:

    F = sum_l 0.5*||z_l - f_l(z_{l-1})||^2   (hidden layers, same as jpc)
      + output term                          (here: advantage-weighted NLL)

then dz/dt = -dF/dz for `max_t1` steps, then a weight step at the settled z.

Advantage handling. With signed advantages the energy is unbounded below (a
negative weight rewards fleeing that sample), which inference can exploit. Two
modes:

  `signed` : E_out = sum_i A_i * NLL_i           -- faithful to policy gradient,
                                                    unbounded below, relies on
                                                    finite max_t1 to stay sane.
  `exp`    : E_out = sum_i exp(A_i/tau) * NLL_i  -- weights are positive so the
                                                    energy is bounded below. This
                                                    is the reward-weighted
                                                    regression / MPO form.

Empirical note: a per-sample weight in the output energy does NOT cancel in the
weight update -- verified in `scripts/probe_pc_sample_weighting.py`, where the
contribution scales ~linearly with the weight (1.0 / 1.79 / 4.87 / 10.98 for
weights 1 / 2 / 5 / 10). That is what makes this route viable at all.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jpc

from pc_algorithms.gaussian_policy import LOG_STD_MIN, LOG_STD_MAX


def _split_out(out, action_dim, exp_std=True, min_std=0.001):
    loc, log_scale = jnp.split(out, 2, axis=-1)
    if exp_std:
        log_scale = jnp.clip(log_scale, LOG_STD_MIN, LOG_STD_MAX)
        scale = jnp.exp(log_scale)
    else:
        scale = jax.nn.softplus(log_scale) + min_std
        log_scale = jnp.log(scale)
    return loc, scale, log_scale


def _weights(advantages, mode, tau):
    if mode == "signed":
        return advantages
    if mode == "exp":
        a = advantages - jnp.max(advantages)          # stabilise the exponential
        return jnp.exp(a / tau)
    raise ValueError(f"unknown advantage mode: {mode}")


def policy_energy(model, activities, obs, pre_tanh, advantages, *,
                  action_dim, exp_std=True, adv_mode="signed", tau=1.0):
    """PC energy whose OUTPUT term is the advantage-weighted Gaussian NLL.

    Hidden-layer terms are identical to `jpc.pc_energy_fn` so the only difference
    from the target route is the output term.
    """
    out = jax.vmap(model[-1])(activities[-2])
    loc, scale, log_scale = _split_out(out, action_dim, exp_std)

    # -log N(z; mu, sigma), constant dropped
    nll = 0.5 * jnp.square((pre_tanh - loc) / scale) + log_scale
    w = _weights(advantages, adv_mode, tau)
    energy = jnp.sum(w[:, None] * nll)

    # hidden layers: z_l predicted by the layer below (same as jpc)
    for l in range(1, len(model) - 1):
        err = activities[l] - jax.vmap(model[l])(activities[l - 1])
        energy = energy + 0.5 * jnp.sum(jnp.square(err))
    err0 = activities[0] - jax.vmap(model[0])(obs)
    return energy + 0.5 * jnp.sum(jnp.square(err0))


def make_likelihood_pc_step(model, optim, opt_state, obs, pre_tanh, advantages, *,
                            action_dim, max_t1=20, dt=0.05, exp_std=True,
                            adv_mode="signed", tau=1.0):
    """One PC update using the likelihood energy. Mirrors `jpc.make_pc_step`.

    Returns a dict with the same keys the caller already uses.
    """
    def E(m, acts):
        return policy_energy(m, acts, obs, pre_tanh, advantages,
                             action_dim=action_dim, exp_std=exp_std,
                             adv_mode=adv_mode, tau=tau)

    # 1. initialise activities with a feedforward pass (as jpc does)
    activities = jpc.init_activities_with_ffwd(model=model, input=obs)

    # 2. inference: settle the activities, dz/dt = -dF/dz
    grad_z = jax.grad(lambda acts: E(model, acts))
    def body(acts, _):
        g = grad_z(acts)
        return [a - dt * gi for a, gi in zip(acts, g)], None
    activities, _ = jax.lax.scan(body, activities, None, length=max_t1)

    # 3. weight gradient at the settled activities, then optimiser step
    loss, grads = eqx.filter_value_and_grad(lambda m: E(m, activities))(model)
    updates, opt_state = optim.update(
        (grads, None), opt_state, (eqx.filter(model, eqx.is_array), None))
    model = eqx.apply_updates(model, updates[0])

    gnorms = jax.tree_util.tree_map(
        lambda g: jnp.linalg.norm(g) if eqx.is_array(g) else None, grads)
    return {"model": model, "opt_state": opt_state, "loss": loss,
            "model_grad_norms": gnorms, "activities": activities}
