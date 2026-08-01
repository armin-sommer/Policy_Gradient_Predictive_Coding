"""Can PC inference actually settle at production batch size, and what fixes it?

jpc's energy is batch-normalised (`pc_energy_fn`: F = 1/2N sum_i ...), so the
inference flow dz/dt = -dF/dz carries a 1/N factor. But the activities of different
samples are INDEPENDENT variables -- dF/dz_i depends only on z_i -- so that factor
is a pure global rescaling of time. It does not move the fixed point; it only sets
how long you must integrate to reach it. Consequences at bench scale:

  * settling needs t1/N ~ 10-40, i.e. t1 ~ 20k-80k at N=2048;
  * `max_t1` cannot buy that, because jpc hard-codes
    `timeout_reached = t >= 4096` in `steady_state_event_with_timeout`;
  * and that event's steady-state test is `rms_norm(y) < atol + rtol*rms_norm(y)`,
    which reduces to `rms_norm(activities) < ~1e-3` -- a test on the activities,
    not on dz/dt -- so it essentially never fires and inference just runs the
    clock out.

The fix is to integrate the SAME ODE at per-sample rate: scale the vector field by
N (equivalently, integrate the un-normalised energy) during inference, while the
weight step keeps the 1/N so gradients stay a mean. This script checks the two
claims that fix rests on:

  1. RATE  -- with the correction, the residual collapses at t1 ~ 10-40 for every
     N; without it, the same collapse needs t1 ~ 10-40 x N.
  2. FIXED POINT -- the settled activities are the same either way, so the
     correction is a reparameterisation of time and not a different problem.

    python scripts/probe_inference_settling.py
    python scripts/probe_inference_settling.py --batch-sizes 32 256 2048
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax
import jpc

import probe_natural_gradient as png
from pc_algorithms.gaussian_policy import gaussian_pc_targets


def energy(model, acts, obs, targets):
    """jpc's batch-normalised PC energy, via jpc itself (not re-derived)."""
    return jpc.pc_energy_fn(params=(model, None), activities=acts,
                            y=targets, x=obs)


def settle(model, obs, targets, t1, rate):
    """Integrate dz/dt = -rate * dF/dz to t1. rate=1 is jpc's behaviour;
    rate=N runs the per-sample dynamics at unit speed."""
    acts0 = jpc.init_activities_with_ffwd(model=model, input=obs)
    if t1 <= 0:
        return acts0, acts0
    grad_z = jax.grad(lambda a: energy(model, a, obs, targets))

    def vf(t, y, args):
        g = grad_z(y)
        return [-rate * gi for gi in g]

    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(vf), solver=diffrax.Heun(),
        t0=0, t1=float(t1), dt0=None, y0=acts0,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3),
        saveat=diffrax.SaveAt(t1=True), max_steps=1_000_000,
    )
    return [a[0] for a in sol.ys], acts0


def residual(model, acts, obs, targets):
    """||dF/dz|| relative to its value at the feedforward init (1.0 = unsettled,
    0.0 = equilibrium). Uses the batch-normalised energy for both, so the ratio
    is rate-independent and comparable across corrections."""
    _, g = jpc.compute_pc_activity_grad(
        params=(model, None), activities=acts, y=targets, x=obs)
    return png._tree_norm(g)


def build(key, n, args, regime):
    k_model, k_bias, k_batch = jr.split(key, 3)
    model = png.build_model(k_model, args.obs_dim, args.action_dim,
                            args.width, args.depth, args.act_fn)
    model = png.set_log_std_bias(model, k_bias, regime, args.action_dim)
    obs, z, adv = png.make_batch(k_batch, model, n, args.obs_dim, args.action_dim)
    out = png.forward(model, obs)
    targets = gaussian_pc_targets(out, z, adv, args.action_dim, args.ts,
                                  exp_std=True, target_clip=None,
                                  natural_target=args.natural_target)
    return model, obs, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dim", type=int, default=17)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--act-fn", default="tanh")
    ap.add_argument("--ts", type=float, default=1.0)
    ap.add_argument("--natural-target", action="store_true")
    ap.add_argument("--regime", default="mixed",
                    choices=["init", "mixed", "floor", "fixed"])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--t1", type=float, nargs="+",
                    default=[1, 5, 20, 80, 320, 1280, 4096])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"jax {jax.__version__}  regime={args.regime}  "
          f"target={'natural' if args.natural_target else 'Euclidean'}")

    # --- claim 1: rate -------------------------------------------------------
    print("\n" + "=" * 78)
    print("1. RATE — relative residual ||dF/dz|| / ||dF/dz||_t0 (0 = settled)")
    print("=" * 78)
    for n in args.batch_sizes:
        model, obs, targets = build(jr.PRNGKey(args.seed), n, args, args.regime)
        r0 = residual(model, jpc.init_activities_with_ffwd(model=model, input=obs),
                      obs, targets)
        print(f"\n  N={n}")
        print(f"  {'t1':>7} {'t1/N':>8} | {'jpc (rate=1)':>13} | "
              f"{'corrected (rate=N)':>19}")
        for t1 in args.t1:
            row = []
            for rate in (1.0, float(n)):
                acts, _ = settle(model, obs, targets, t1, rate)
                row.append(residual(model, acts, obs, targets) / (r0 + 1e-300))
            print(f"  {t1:7.0f} {t1/n:8.3f} | {row[0]:13.3e} | {row[1]:19.3e}")

    # --- claim 2: same fixed point ------------------------------------------
    print("\n" + "=" * 78)
    print("2. FIXED POINT — settled activities, jpc at t1=N*T vs corrected at t1=T")
    print("=" * 78)
    print("  (if the correction were a different problem these would disagree)")
    print(f"\n  {'N':>6} {'T':>6} {'cos':>10} {'max|rel diff|':>15} "
          f"{'resid jpc':>11} {'resid corr':>11}")
    for n in args.batch_sizes:
        model, obs, targets = build(jr.PRNGKey(args.seed), n, args, args.regime)
        T = 40.0
        a_jpc, _ = settle(model, obs, targets, T * n, 1.0)   # same ODE, slow clock
        a_cor, _ = settle(model, obs, targets, T, float(n))  # ... fast clock
        f_jpc = jnp.concatenate([a.ravel() for a in a_jpc])
        f_cor = jnp.concatenate([a.ravel() for a in a_cor])
        cos = png._cos(f_jpc, f_cor)
        rel = float(jnp.max(jnp.abs(f_jpc - f_cor)) /
                    (jnp.max(jnp.abs(f_jpc)) + 1e-300))
        r0 = residual(model, jpc.init_activities_with_ffwd(model=model, input=obs),
                      obs, targets)
        print(f"  {n:6} {T:6.0f} {cos:10.6f} {rel:15.2e} "
              f"{residual(model,a_jpc,obs,targets)/r0:11.2e} "
              f"{residual(model,a_cor,obs,targets)/r0:11.2e}")

    print("\nnote: claim 2 integrates jpc's flow to t1 = 40*N by calling diffrax")
    print("directly, which the production path CANNOT do — jpc's own")
    print("steady_state_event_with_timeout hard-stops at t=4096.")


if __name__ == "__main__":
    main()
