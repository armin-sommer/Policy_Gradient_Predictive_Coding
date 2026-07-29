"""Does a per-sample weight in the PC output energy survive the weight update?

This decides whether an advantage-weighted likelihood energy is viable for PCPG.

  If the weight CANCELS  -> the advantage would cancel too, so putting the policy
                            likelihood in PC's energy cannot express policy
                            gradient. The target-based route (natural_target) is
                            the only option.
  If the weight SURVIVES -> the likelihood energy is viable, and "PC supplies the
                            geometry implicitly" becomes directly testable.

Method: two identical samples, differing only in their energy weight. Run the PC
inference to convergence, take the weight gradient, and compare each sample's
contribution across weightings. No training, no GPU.

Run:  python scripts/probe_pc_sample_weighting.py
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import jpc

jax.config.update("jax_enable_x64", True)


def build(key, in_dim=3, hid=4, out_dim=2):
    """Two-layer linear PC network (linear so the result is checkable by hand)."""
    k1, k2 = jax.random.split(key)
    return [
        eqx.nn.Linear(in_dim, hid, use_bias=False, key=k1),
        eqx.nn.Linear(hid, out_dim, use_bias=False, key=k2),
    ]


def energy(params, activities, y, x, w):
    """PC energy with a PER-SAMPLE weight on the output term only."""
    model, _ = params
    pred = jax.vmap(model[-1])(activities[-2])
    eL = y - pred
    E = 0.5 * jnp.sum(w[:, None] * eL ** 2)          # <-- the weight under test
    e0 = activities[0] - jax.vmap(model[0])(x)        # hidden term
    E = E + 0.5 * jnp.sum(e0 ** 2)
    return E


def settle(params, y, x, w, steps=400, dt=0.05):
    """Run inference dz/dt = -dE/dz to (near) convergence."""
    model, _ = params
    acts = jpc.init_activities_with_ffwd(model=model, input=x)
    grad_z = jax.grad(energy, argnums=1)
    for _ in range(steps):
        g = grad_z(params, acts, y, x, w)
        acts = [a - dt * gi for a, gi in zip(acts, g)]
    return acts


def per_sample_contribution(params, y, x, w):
    """Weight-gradient contribution of each sample, at the settled activities.

    Settle once with the full batch (as in training), then isolate sample i by
    slicing EVERY term to row i — masking only the output term would leave the
    hidden term summed over the batch and contaminate the comparison.
    """
    acts = settle(params, y, x, w)                     # settle with the FULL w
    out = []
    for i in range(y.shape[0]):
        sl = slice(i, i + 1)
        acts_i = [a[sl] for a in acts]
        g = jax.grad(energy)(params, acts_i, y[sl], x[sl], w[sl])
        leaves = [l for l in jax.tree_util.tree_leaves(g) if eqx.is_array(l)]
        out.append(float(jnp.sqrt(sum(jnp.sum(l ** 2) for l in leaves))))
    return out


def main():
    key = jax.random.PRNGKey(0)
    model = build(key)
    params = (model, None)

    # two IDENTICAL samples, so any difference comes only from the weight
    x = jnp.array([[1.0, 0.5, -0.3], [1.0, 0.5, -0.3]])
    y = jnp.array([[0.7, -0.4], [0.7, -0.4]])

    print("Two identical samples; only sample 1's energy weight changes.\n")
    print(f"{'weights':>16}  {'sample0':>10}  {'sample1':>10}  {'ratio s1/s0':>12}")
    print("-" * 54)
    base = None
    for wv in [1.0, 2.0, 5.0, 10.0]:
        w = jnp.array([1.0, wv])
        c0, c1 = per_sample_contribution(params, y, x, w)
        if base is None:
            base = c1
        print(f"{'[1.0, %.1f]' % wv:>16}  {c0:>10.6f}  {c1:>10.6f}  {c1/c0:>12.4f}")

    print("\nreading:")
    print("  ratio grows ~linearly with the weight -> weight SURVIVES (batch-level)")
    print("                                           => advantage survives, likelihood route viable")
    print("  ratio stays ~1.0 regardless of weight  -> weight CANCELS (per-sample)")
    print("                                           => advantage cancels, likelihood route dead")


if __name__ == "__main__":
    main()
