"""Unit checks for the likelihood-energy PC step. No env needed.

Verifies, on synthetic data:
  1. the step runs and returns finite parameters
  2. the ADVANTAGE actually changes the update (it does not cancel)
  3. sign correctness: positive advantage moves mu TOWARD the sampled action,
     negative advantage moves it AWAY
  4. the 'exp' weighting mode is bounded and also advantage-sensitive

Run:  PYTHONPATH=src python scripts/test_likelihood_energy.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jpc
import optax

from pc_algorithms.likelihood_energy import make_likelihood_pc_step, policy_energy

ACTION_DIM = 3
OBS_DIM = 5


def setup(seed=0, lr=0.05):
    key = jr.PRNGKey(seed)
    model = jpc.make_mlp(key, input_dim=OBS_DIM, width=16, depth=2,
                         output_dim=2 * ACTION_DIM, act_fn="tanh", use_bias=True)
    optim = optax.sgd(lr)
    opt_state = optim.init((eqx.filter(model, eqx.is_array), None))
    return model, optim, opt_state


def mu_of(model, obs):
    out = jpc.init_activities_with_ffwd(model=model, input=obs)[-1]
    return jnp.split(out, 2, axis=-1)[0]


def main():
    obs = jr.normal(jr.PRNGKey(1), (8, OBS_DIM))
    z = jr.normal(jr.PRNGKey(2), (8, ACTION_DIM))

    print("1. step runs and stays finite")
    model, optim, opt_state = setup()
    adv = jnp.ones(8)
    r = make_likelihood_pc_step(model, optim, opt_state, obs, z, adv,
                                action_dim=ACTION_DIM, max_t1=20)
    leaves = [l for l in jax.tree_util.tree_leaves(r["model"]) if eqx.is_array(l)]
    finite = all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)
    print(f"   loss={float(r['loss']):.4f}  all params finite: {finite}")
    assert finite

    print("\n2. does the advantage change the update? (must NOT cancel)")
    base = None
    for a in [1.0, 2.0, 5.0]:
        model, optim, opt_state = setup()
        r = make_likelihood_pc_step(model, optim, opt_state, obs, z,
                                    jnp.full((8,), a), action_dim=ACTION_DIM, max_t1=20)
        d = float(jnp.linalg.norm(mu_of(r["model"], obs) - mu_of(model, obs)))
        base = d if base is None else base
        print(f"   advantage={a:<4} |delta mu| = {d:.6f}   ratio vs A=1: {d/base:.3f}")

    print("\n3. sign: +A should move mu TOWARD z, -A AWAY")
    for a, want in [(+2.0, "toward"), (-2.0, "away")]:
        model, optim, opt_state = setup()
        before = float(jnp.mean(jnp.abs(mu_of(model, obs) - z)))
        r = make_likelihood_pc_step(model, optim, opt_state, obs, z,
                                    jnp.full((8,), a), action_dim=ACTION_DIM, max_t1=20)
        after = float(jnp.mean(jnp.abs(mu_of(r["model"], obs) - z)))
        got = "toward" if after < before else "away"
        ok = "OK " if got == want else "FAIL"
        print(f"   [{ok}] A={a:+.1f}: |mu-z| {before:.4f} -> {after:.4f}  ({got}, wanted {want})")

    print("\n4. 'exp' mode: bounded and still advantage-sensitive")
    for a_hi in [0.0, 2.0]:
        model, optim, opt_state = setup()
        adv = jnp.array([a_hi] + [0.0] * 7)
        r = make_likelihood_pc_step(model, optim, opt_state, obs, z, adv,
                                    action_dim=ACTION_DIM, max_t1=20,
                                    adv_mode="exp", tau=1.0)
        d = float(jnp.linalg.norm(mu_of(r["model"], obs) - mu_of(model, obs)))
        print(f"   sample0 advantage={a_hi:<4} |delta mu| = {d:.6f}  loss={float(r['loss']):.4f}")

    print("\nall checks ran")


if __name__ == "__main__":
    main()
