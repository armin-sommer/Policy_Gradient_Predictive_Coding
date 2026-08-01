"""Is Innocenti's adaptive trust-region rescaling doing anything here?

`probe_natural_gradient.py` reports `cos(Delta, d_EQ)` — how well the PC update
matches the equilibrium prediction. It does NOT report how far `d_EQ` is from plain
backprop. That distinction is the whole content of the trust-region claim: PC at
equilibrium is BP on an `S^-1`-rescaled loss, with

    S_i = I + sum_l B_{l,i} B_{l,i}^T ,   B_{l,i} = d out_i / d z_{l,i}

so if `S ~ I` the rescaling is inert and "PC = adaptive trust region" predicts
nothing distinguishable from backprop. Note the sum runs over **hidden layers**,
so the effect is architectural: it grows with depth and vanishes for a shallow net.

The bench uses `jpc.make_mlp(depth=2)`, which builds ONE hidden layer.

Reports per (regime, depth): the spectrum of S, cos(d_BP, d_EQ), and the
NG ceilings — so "why don't we see a natural gradient" separates into
"S is inert here" vs "S is active but is not the Fisher".

    python scripts/probe_trust_region_metric.py
    python scripts/probe_trust_region_metric.py --depths 2 3 5 9 --regimes mixed
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
import jpc

import probe_natural_gradient as png
from pc_algorithms.gaussian_policy import gaussian_pc_targets


def s_spectrum(model, obs, out_dim):
    """Eigenvalues of S_i = I + sum_l B_l B_l^T, pooled over the batch."""
    acts = jpc.init_activities_with_ffwd(model=model, input=obs)
    n = obs.shape[0]
    s = jnp.tile(jnp.eye(out_dim, dtype=obs.dtype), (n, 1, 1))
    n_terms = 0
    for l in range(len(model) - 1):
        def tail(z_l, _l=l):
            h = z_l
            for layer in model[_l + 1:]:
                h = layer(h)
            return h
        b = jax.vmap(jax.jacrev(tail))(acts[l])
        s = s + jnp.einsum("nod,npd->nop", b, b)
        n_terms += 1
    eig = jnp.linalg.eigvalsh(s)                      # (n, out_dim), ascending
    return np.asarray(eig), n_terms


def run(args):
    print(f"jax {jax.__version__}  devices={jax.devices()}")
    print("S_i = I + sum_l B_l B_l^T   (sum over HIDDEN layers)")
    print("If S ~ I (eig ~ 1, cond ~ 1) the trust-region rescaling is inert:")
    print("d_EQ collapses onto d_BP and PC predicts nothing beyond backprop.\n")

    header = (f"{'regime':>7} {'depth':>6} {'hid':>4} {'eig_min':>9} {'eig_max':>10} "
              f"{'cond':>9} {'cos(d_BP,d_EQ)':>15} {'cos(dEQ,dNG)':>13} "
              f"{'cos(dSGD,dNG)':>14}")
    print(header)
    print("-" * len(header))

    for regime in args.regimes:
        for depth in args.depths:
            key = jr.PRNGKey(args.seed)
            k_model, k_bias, k_batch = jr.split(key, 3)
            model = png.build_model(k_model, args.obs_dim, args.action_dim,
                                    args.width, depth, args.act_fn)
            model = png.set_log_std_bias(model, k_bias, regime, args.action_dim)
            obs, z, adv = png.make_batch(k_batch, model, args.n,
                                         args.obs_dim, args.action_dim)

            geo = png.Geometry(model, obs, z, adv, args.action_dim, args.ts,
                               dampings=args.dampings, cg_iters=args.cg_iters,
                               cg_tol=args.cg_tol)
            # Euclidean family (the production default) — the target family does
            # not affect S, which is a property of the architecture alone.
            targets = gaussian_pc_targets(
                geo.out, z, adv, args.action_dim, args.ts,
                exp_std=True, target_clip=None, natural_target=False)
            d_bp, d_eq = png.energy_references(geo, model, targets)

            eig, n_hidden = s_spectrum(model, obs, 2 * args.action_dim)
            cos_bp_eq = png._cos(d_bp, d_eq)
            cos_eq_ng, _ = geo.best(d_eq, geo.d_ng)
            cos_sgd_ng, _ = geo.best(geo.d_sgd, geo.d_ng)

            print(f"{regime:>7} {depth:>6} {n_hidden:>4} "
                  f"{eig.min():9.3f} {eig.max():10.1f} "
                  f"{eig.max()/max(eig.min(),1e-30):9.1f} "
                  f"{cos_bp_eq:15.4f} {cos_eq_ng:13.4f} {cos_sgd_ng:14.4f}")

    print("\nreading:")
    print("  cos(d_BP,d_EQ) ~ 1.0  -> S is inert; PC-at-equilibrium IS backprop,")
    print("                           so no trust-region/second-order effect exists")
    print("                           to be natural-gradient-like in the first place.")
    print("  cos(dEQ,dNG) < 1      -> even a perfectly settled PC has a CEILING on")
    print("                           how natural-gradient-like it can be: S is a")
    print("                           network-Jacobian metric, the Fisher is a")
    print("                           distributional one. They are different objects.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dim", type=int, default=17)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 5, 9],
                    help="jpc.make_mlp depth; depth=2 is ONE hidden layer (bench).")
    ap.add_argument("--act-fn", default="tanh")
    ap.add_argument("--ts", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regimes", nargs="+", default=["mixed", "fixed"],
                   choices=["init", "mixed", "floor", "fixed"])
    ap.add_argument("--dampings", type=float, nargs="+",
                    default=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    ap.add_argument("--cg-iters", type=int, default=400)
    ap.add_argument("--cg-tol", type=float, default=1e-12)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
