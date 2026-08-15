"""Rate-corrected PC inference at production batch size.

`jpc.make_pc_step` cannot equilibrate a bench minibatch, for three compounding
reasons found by `scripts/probe_natural_gradient.py` (F1) and
`scripts/probe_inference_settling.py`:

1. **The flow runs N times too slow.** jpc's energy is batch-normalised
   (`pc_energy_fn`: `F = 1/2N sum_i ...`), so `dz/dt = -dF/dz` carries a `1/N`
   factor. The activities of different samples are *independent* variables --
   `dF/dz_i` depends on `z_i` only -- so that factor is a pure rescaling of time:
   it does not move the fixed point, it only sets how long you must integrate to
   reach it. Measured: the residual is a function of `t1/N` alone, hitting 0.39 of
   its initial value at `t1/N ~ 0.6` for every N tested. Settling needs
   `t1/N ~ 10-40`.
2. **`max_t1` cannot buy the time.** jpc hard-codes `timeout_reached = t >= 4096`
   in `steady_state_event_with_timeout`, so raising `max_t1` past 4096 does
   nothing. At N=2048 settling would need `t1 ~ 20k-80k`.
3. **The early-stop never fires.** That event's steady-state test is
   `rms_norm(y) < atol + rtol*rms_norm(y)`, i.e. `rms_norm(activities) < ~1e-3` --
   a test on the activities rather than on `dz/dt`. Activities are O(1), so it is
   effectively always false and inference just runs the clock out.

The fix here integrates the *same* ODE at per-sample rate: scale the inference
vector field by N. The weight step still uses jpc's batch-normalised
`compute_pc_param_grads`, so gradients remain a mean over the batch and learning
rates carry over unchanged.

The rate correction fixes the time-scale bug, but it is not a fixed-point
guarantee: on real ReLU MuJoCo batches, the activity-gradient residual plateaus
above zero.  The learning-relevant PC parameter gradient converges by roughly
per-sample time 3-5, so `max_t1=20` is a conservative stable-update budget rather
than evidence that `dF/dz == 0`. See `scripts/probe_settling_budget.py`.

This is a drop-in replacement for `jpc.make_pc_step` on the arguments the callers
use, returning the same keys.
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jpc


def initialize_activities(model, obs):
    """Forward-initialize every PC activity before starting inference.

    The free hidden activities must start at the model's complete feedforward
    trajectory, not at zeros or at a stale state from a previous minibatch.  Keep
    the predicted output in the tree as well: jpc's energy expects one activity
    for every model layer even though its supervised energy clamps the target at
    the final layer.  This is equivalent to
    ``jpc.init_activities_with_ffwd(model=model, input=obs)``, but makes the
    initialization invariant explicit in PCPG's inference implementation.
    """
    activities = []
    activity = obs
    for layer in model:
        activity = jax.vmap(layer)(activity)
        activities.append(activity)
    return activities


def settle_activities(model, obs, output, *, max_t1=20, rate_correction=True,
                      dt=None, rtol=1e-3, atol=1e-3, max_steps=1_000_000):
    """Integrate inference from a complete feedforward activity trajectory.

    `rate_correction=False` reproduces jpc's timescale exactly (rate 1), which is
    what the committed runs did; `True` runs the per-sample dynamics at unit speed.
    """
    acts0 = initialize_activities(model, obs)
    if max_t1 <= 0:
        return acts0, acts0

    rate = float(obs.shape[0]) if rate_correction else 1.0

    def energy(acts):
        return jpc.pc_energy_fn(params=(model, None), activities=acts,
                                y=output, x=obs)

    grad_z = jax.grad(energy)

    def vector_field(t, y, args):
        return [-rate * g for g in grad_z(y)]

    # No steady-state event: jpc's is a test on the activities, not on dz/dt, so
    # it never fires. Integrating the fixed interval is both correct and cheaper
    # than a working event would be, given the residual curve above.
    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(vector_field),
        solver=diffrax.Heun(),
        t0=0, t1=float(max_t1), dt0=dt,
        y0=acts0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=max_steps,
    )
    return [a[0] for a in sol.ys], acts0


@eqx.filter_jit          # jpc.make_pc_step is filter_jit'd too; without this the
                         # whole diffeqsolve re-traces on every call (~80x slower).
def make_pc_step_at_rate(model, optim, opt_state, output, input, *,
                         max_t1=20, rate_correction=True, grad_norms=False,
                         dt=None, rtol=1e-3, atol=1e-3, max_steps=1_000_000):
    """One PC update with rate-corrected inference.

    Signature mirrors the `jpc.make_pc_step(...)` calls in pc_actor_critic /
    pc_reinforce; returns the same keys ("model", "opt_state", "loss",
    "model_grad_norms", "activities").
    """
    activities, acts0 = settle_activities(
        model, input, output, max_t1=max_t1, rate_correction=rate_correction,
        dt=dt, rtol=rtol, atol=atol, max_steps=max_steps)

    # Weight gradients from jpc's own (batch-normalised) routine: the rate
    # correction is about the inference clock only, never about gradient scale.
    grads = jpc.compute_pc_param_grads(
        params=(model, None), activities=activities, y=output, x=input)[0]
    loss = jpc.pc_energy_fn(params=(model, None), activities=activities,
                            y=output, x=input)

    updates, opt_state = optim.update(
        (grads, None), opt_state, (eqx.filter(model, eqx.is_array), None))
    model = eqx.apply_updates(model, updates[0])

    out = {"model": model, "opt_state": opt_state, "loss": loss,
           "activities": activities}
    if grad_norms:
        # Same shape jpc returns: a per-leaf norm pytree, which _global_norm folds.
        out["model_grad_norms"] = jax.tree_util.tree_map(
            lambda g: jnp.linalg.norm(g) if eqx.is_array(g) else None, grads)
    return out


def inference_residual(model, obs, output, activities):
    """||dF/dz|| at these activities — the settling diagnostic to log online."""
    _, g = jpc.compute_pc_activity_grad(
        params=(model, None), activities=activities, y=output, x=obs)
    return jnp.sqrt(sum(jnp.sum(jnp.square(l))
                        for l in jax.tree_util.tree_leaves(g)))
