"""Interp probe: is a policy update's mean channel scale-free in sigma?

Two questions, both answered at a snapshot (no training, CPU-only):

1. **Output Fisher.** Is F_out = diag(1/sigma^2, 2) for this repo's Gaussian
   policy? Checked by Monte-Carlo against the analytic claim. The `2` in the
   scale block assumes the scale is parameterized as u = log sigma. This repo has
   *two* parameterizations (see networks.make_networks / NormalTanhDistribution):
     - exp_std=True  (sota_init, every configs/mujoco_*_{ppo,trpo}.yaml):
       sigma = exp(u)          -> scale block should be 2
     - exp_std=False (default): sigma = softplus(r) + min_std
       -> scale block is 2*(sigma'(r)/sigma)^2, NOT 2
   The probe measures both, so the discrepancy is visible rather than assumed.

2. **Scale-freeness.** Regress log||d_mu|| on log sigma across a batch:
     vanilla PG   d_mu = -A(a-mu)/sigma^2      -> slope ~ -2 (blows up as sigma->0)
     NGD/F_out    d_mu = F_out^-1 g = -A(a-mu) -> slope ~  0 (scale-free)
   A PC scheme whose output precision is the identity behaves like the former; one
   that predicts Pi_L = F_out behaves like the latter. The slope is the
   discriminator, and it is measurable without training.

   Note this needs sigma to *vary across the batch*. On the sota_init path
   log_std is a single state-independent parameter vector, so within one batch
   sigma is constant per action dim and the regression is degenerate — reported
   explicitly rather than silently producing a meaningless slope.

Usage:
    python scripts/probe_natgrad_scaling.py
    python scripts/probe_natgrad_scaling.py --batch 4096 --mc-samples 200000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import jax
import jax.numpy as jnp

from networks.distributions import NormalTanhDistribution


# --- 1. output Fisher ------------------------------------------------------


def _neg_logp(params, raw_action, dist):
    """-log pi(raw_action) as a function of the output-layer activities z_L."""
    return -dist.log_prob(params[None, :], raw_action[None, :])[0]


def output_fisher_mc(sigma, exp_std, n_samples, key):
    """Monte-Carlo E[grad grad^T] of -log pi w.r.t. z_L = [mu, scale_param].

    Returns (mu_block, scale_block, cross_block) averaged over action dims —
    the analytic prediction is diag(1/sigma^2, 2) for the exp parameterization.
    """
    act_dim = sigma.shape[0]
    dist = NormalTanhDistribution(event_size=act_dim, exp_std=exp_std)

    mu = jnp.zeros(act_dim)
    if exp_std:
        scale_param = jnp.log(sigma)          # sigma = exp(u)
    else:
        # invert softplus(r) + min_std = sigma  ->  r = log(exp(s)-1), s = sigma - min_std
        s = sigma - dist._min_std
        scale_param = jnp.log(jnp.expm1(s))
    params = jnp.concatenate([mu, scale_param])

    grad_fn = jax.jit(jax.vmap(
        jax.grad(_neg_logp), in_axes=(None, 0, None)), static_argnums=2)

    # Sample pre-tanh actions from the model's own Gaussian (Fisher is an
    # expectation under the model, so actions must come from it).
    raw = mu + sigma * jax.random.normal(key, (n_samples, act_dim))
    g = grad_fn(params, raw, dist)                      # (n, 2*act_dim)
    g_mu, g_scale = jnp.split(g, 2, axis=-1)

    mu_block = jnp.mean(g_mu ** 2, axis=0)
    scale_block = jnp.mean(g_scale ** 2, axis=0)
    cross = jnp.mean(g_mu * g_scale, axis=0)
    return np.asarray(mu_block), np.asarray(scale_block), np.asarray(cross)


def report_fisher(args, key):
    print("=" * 78)
    print("1. OUTPUT FISHER  F_out  vs analytic diag(1/sigma^2, 2)")
    print("=" * 78)

    sigma = jnp.array(args.sigmas)
    for exp_std in (True, False):
        label = "exp_std=True  (sota_init: sigma=exp(u))" if exp_std else \
                "exp_std=False (default: sigma=softplus(r)+min_std)"
        key, sub = jax.random.split(key)
        mu_b, sc_b, cross = output_fisher_mc(sigma, exp_std, args.mc_samples, sub)

        print(f"\n  {label}")
        print(f"  {'sigma':>8} {'mu block':>12} {'want 1/s^2':>12} "
              f"{'scale block':>12} {'want':>10} {'cross':>10}")
        for i, s in enumerate(np.asarray(sigma)):
            if exp_std:
                want_scale = 2.0
            else:
                # d sigma/d r = sigmoid(r); scale block = 2*(sigma'/sigma)^2
                r = float(np.log(np.expm1(s - 0.001)))
                dsig = 1.0 / (1.0 + np.exp(-r))
                want_scale = 2.0 * (dsig / s) ** 2
            print(f"  {s:8.3f} {mu_b[i]:12.3f} {1.0/s**2:12.3f} "
                  f"{sc_b[i]:12.3f} {want_scale:10.3f} {cross[i]:10.3f}")

        # The block-diagonality claim: cross terms are odd moments -> 0.
        max_cross = float(np.max(np.abs(cross)))
        rel_mu = float(np.max(np.abs(mu_b - 1.0 / np.asarray(sigma) ** 2)
                              / (1.0 / np.asarray(sigma) ** 2)))
        print(f"  -> max |cross| = {max_cross:.4f} (block-diagonal if ~0)")
        print(f"  -> mu block max rel. error = {rel_mu:.1%}")
    return key


# --- 2. scale-freeness of the mean update ----------------------------------


def mean_updates(sigma, advantage, raw_action, mode):
    """Per-sample mean-channel update d_mu under the given output precision.

    vanilla : d_mu = -A (a-mu)/sigma^2            (Pi_L = I in output space)
    natural : d_mu = F_out^-1 g = -A (a-mu)       (Pi_L = F_out)
    """
    g_mu = -advantage[:, None] * raw_action / sigma[:, None] ** 2  # mu = 0
    if mode == "vanilla":
        d_mu = g_mu
    elif mode == "natural":
        d_mu = sigma[:, None] ** 2 * g_mu
    else:
        raise ValueError(mode)
    return jnp.linalg.norm(d_mu, axis=-1)


def _binned_slope(sigma, values, n_bins):
    """Regress log(mean value per log-sigma bin) on log sigma.

    Binning first is what makes this readable: per-sample ||d_mu|| carries the
    noise of both the sampled action and the advantage, so an unbinned fit has
    the right slope but r^2 ~ 0.25. Bin means cancel that noise.
    """
    x_all = np.log(np.asarray(sigma))
    v_all = np.asarray(values)
    edges = np.linspace(x_all.min(), x_all.max(), n_bins + 1)
    idx = np.clip(np.digitize(x_all, edges) - 1, 0, n_bins - 1)

    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 8:
            continue
        xs.append(x_all[m].mean())
        ys.append(np.log(v_all[m].mean() + 1e-30))
    xs, ys = np.array(xs), np.array(ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    r2 = 1.0 - np.var(ys - (slope * xs + intercept)) / np.var(ys)
    return slope, r2


def report_scale_freeness(args, key):
    print()
    print("=" * 78)
    print("2. SCALE-FREENESS  slope of log||d_mu|| vs log sigma")
    print("=" * 78)
    print(f"\n  heteroscedastic batch (n={args.batch}), sigma log-uniform in "
          f"[{args.sigma_lo}, {args.sigma_hi}]")

    act_dim = args.act_dim
    key, k_s, k_a, k_adv = jax.random.split(key, 4)

    # One sigma per sample (state-dependent sigma, i.e. the exp_std=False path),
    # spanning the mixed regime.
    log_lo, log_hi = np.log(args.sigma_lo), np.log(args.sigma_hi)
    sigma = jnp.exp(jax.random.uniform(k_s, (args.batch,), minval=log_lo, maxval=log_hi))
    # On-policy: a ~ N(mu, sigma^2), so (a-mu) = sigma * n and carries its OWN
    # factor of sigma. This is why the raw-unit slopes below are -1/+1 and not
    # the -2/0 you get if (a-mu) is held fixed.
    raw = sigma[:, None] * jax.random.normal(k_a, (args.batch, act_dim))
    # Advantages: unit-variance, sigma-independent (isolates the geometry).
    adv = jax.random.normal(k_adv, (args.batch,))

    # Two metrics. Raw action units is what you would naively regress; sigma
    # units (d_mu / sigma) is the KL-relevant one, since a Gaussian's KL depends
    # on the mean shift measured in standard deviations.
    metrics = {
        "raw  ||d_mu||":       (lambda n, s: n,     {"vanilla": -1.0, "natural": +1.0}),
        "sigma-units ||d_mu||/sigma": (lambda n, s: n / s, {"vanilla": -2.0, "natural": 0.0}),
    }

    results = {}
    for metric_name, (transform, expected) in metrics.items():
        print(f"\n  metric: {metric_name}")
        print(f"  {'mode':>10} {'slope':>9} {'expected':>9} {'r^2':>7}")
        for mode in ("vanilla", "natural"):
            norms = mean_updates(sigma, adv, raw, mode)
            slope, r2 = _binned_slope(sigma, transform(norms, sigma), args.n_bins)
            results[(metric_name, mode)] = slope
            print(f"  {mode:>10} {slope:9.3f} {expected[mode]:9.1f} {r2:7.3f}")

    su = "sigma-units ||d_mu||/sigma"
    print("\n  READING:")
    print(f"  - In KL-relevant sigma units the contrast is the sharp one:")
    print(f"      vanilla PG / Pi_L=I : slope {results[(su,'vanilla')]:+.2f}  "
          f"(step in sigma units explodes as sigma -> 0)")
    print(f"      NGD / Pi_L=F_out    : slope {results[(su,'natural')]:+.2f}  "
          f"(scale-free -- sigma cancels exactly)")
    print(f"  - In RAW action units neither is flat "
          f"({results[('raw  ||d_mu||','vanilla')]:+.2f} vs "
          f"{results[('raw  ||d_mu||','natural')]:+.2f}): on-policy (a-mu) ~ sigma")
    print(f"    contributes its own power of sigma. A probe that regresses raw")
    print(f"    ||d_mu|| and expects 0 for correct NGD will read a false negative.")
    print(f"  - precision gap at the floor: F_out/I = 1/sigma^2 = "
          f"{1.0/args.sigma_lo**2:.1f}x at sigma={args.sigma_lo}")

    # What the repo's frozen configs actually do.
    print()
    print("-" * 78)
    print("  NOTE on the committed MuJoCo runs: every configs/mujoco_*_{ppo,trpo}.yaml")
    print("  sets sota_init: true -> SOTAPolicyMLP's log_std is a single")
    print("  state-INDEPENDENT parameter vector. sigma therefore has no")
    print("  across-batch spread, so this regression is degenerate on those runs;")
    print("  the heteroscedastic Pi_L(z_L) concern does not apply to them.")
    print("-" * 78)
    return key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--act-dim", type=int, default=6)
    p.add_argument("--mc-samples", type=int, default=100_000)
    p.add_argument("--sigmas", type=float, nargs="*",
                   default=[0.14, 0.5, 1.0, 1.65],
                   help="sigmas at which to measure F_out (mixed-regime span).")
    p.add_argument("--sigma-lo", type=float, default=0.14)
    p.add_argument("--sigma-hi", type=float, default=1.65)
    p.add_argument("--n-bins", type=int, default=24,
                   help="log-sigma bins for the binned slope fit.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"jax {jax.__version__}  devices={jax.devices()}")
    key = jax.random.PRNGKey(args.seed)
    key = report_fisher(args, key)
    report_scale_freeness(args, key)


if __name__ == "__main__":
    main()
