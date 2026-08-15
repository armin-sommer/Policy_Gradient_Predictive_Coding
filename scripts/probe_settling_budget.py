"""How much inference time does a real PCPG run need before it is settled?

`probe_inference_settling.py` answers the *rate* question on a synthetic batch: with
jpc's batch-normalised energy the residual is a function of t1/N, so the per-sample
inference time is tau = t1/N and the rate correction makes tau = t1. It reports that
the residual falls below 1e-2 by tau ~ 5 and concludes max_t1=20 is enough.

Two things that argument does not establish, and this probe measures:

1. **It was measured on a randomly-initialised net and a synthetic batch.** The
   settling budget depends on the energy landscape, which changes as weights grow
   during training. So the sweep here runs on the *real* bench pipeline -- real Brax
   rollouts, real GAE advantages, real `gaussian_pc_targets`, N = 2048 -- and repeats
   it at several points along a real training trajectory.
2. **The residual is not what learning consumes.** Training only sees the weight
   gradient that the settled activities produce. Activities can still be moving while
   the gradient direction has stopped, or the reverse. The headline metric here is
   therefore `cos(g(tau), g*)` -- the PC weight gradient at inference budget tau
   against the fully-settled one -- with the residual reported alongside.

Every quantity is reported against tau in PER-SAMPLE units, so the production
operating point is included on the same axis as the corrected one: an uncorrected run
at max_t1=20 and N=2048 sits at tau = 20/2048 ~ 0.0098.

    python scripts/probe_settling_budget.py                     # bench, at init
    python scripts/probe_settling_budget.py --measure-at 0 20 60
    python scripts/probe_settling_budget.py --warmup-arm unsettled
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jpc
import optax

from env import make_vec_env
import probe_natural_gradient as ng_probe
# The standalone geometry probe enables x64 for CPU CG accuracy at import time.
# Brax requires its rollout carry to remain float32, so restore that mode before
# creating environments; the Fisher construction itself is unchanged.
jax.config.update("jax_enable_x64", False)
from utils.utils import EnvConfig
from pc_algorithms.gaussian_policy import (
    gaussian_pc_targets,
    sample_gaussian_action,
)
from pc_algorithms.returns import compute_gae
from pc_algorithms.precision_energy import (
    settle_precision_outputs,
    trust_region_precision,
)

# Thresholds the summary reports a required tau for. 0.99 is "same direction to
# within a degree of arc"; 0.999 is the stricter reading.
COS_TARGETS = (0.99, 0.999)
RESID_TARGET = 1e-2


def _flat(tree):
    leaves = [l for l in jax.tree_util.tree_leaves(tree) if eqx.is_array(l)]
    return jnp.concatenate([l.ravel() for l in leaves])


def _cos(a, b):
    na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
    return float(jnp.dot(a, b) / (na * nb + 1e-30))


def settle(model, obs, targets, tau, rtol=1e-3, atol=1e-3):
    """Integrate the per-sample inference dynamics to tau; return (acts, n_steps).

    Same ODE and solver settings as `pc_algorithms.inference.settle_activities`
    (Heun + PID, rate = N), but it also reports the solver step count, which is the
    hardware-independent cost of a given tau, and takes the tolerances as arguments
    so the residual floor can be tested against them.
    """
    acts0 = jpc.init_activities_with_ffwd(model=model, input=obs)
    if tau <= 0:
        return acts0, 0

    rate = float(obs.shape[0])

    def vector_field(t, y, args):
        g = jax.grad(lambda a: jpc.pc_energy_fn(
            params=(model, None), activities=a, y=targets, x=obs))(y)
        return [-rate * gi for gi in g]

    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(vector_field),
        solver=diffrax.Heun(),
        t0=0, t1=float(tau), dt0=None, y0=acts0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=1_000_000,
    )
    return [a[0] for a in sol.ys], int(sol.stats["num_steps"])


def residual(model, acts, obs, targets):
    """||dF/dz|| under the batch-normalised energy, so ratios are rate-independent."""
    _, g = jpc.compute_pc_activity_grad(
        params=(model, None), activities=acts, y=targets, x=obs)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(l))
                              for l in jax.tree_util.tree_leaves(g))))


def residual_per_layer(model, acts, obs, targets):
    _, g = jpc.compute_pc_activity_grad(
        params=(model, None), activities=acts, y=targets, x=obs)
    return [float(jnp.linalg.norm(l)) for l in jax.tree_util.tree_leaves(g)]


def param_grad(model, acts, obs, targets):
    return jpc.compute_pc_param_grads(
        params=(model, None), activities=acts, y=targets, x=obs)[0]


def parameter_space_npg_energy(geo, damping):
    """Solve the parameter-displacement NPG energy with the exact Fisher FVP.

    For a policy-loss gradient g, this quadratic is
      1/2 delta.T (F + lambda I) delta + g.T delta.
    Its KKT condition is (F + lambda I) delta + g = 0, so its minimizer is
    precisely the damped natural-gradient descent direction.  This is PC-like
    inference over *shared parameters*, not ordinary PC's per-sample activities.
    """
    delta = geo.d_ng[damping]
    kkt = geo.fvp(delta) + damping * delta + geo.g
    rel_kkt = float(jnp.linalg.norm(kkt) / (jnp.linalg.norm(geo.g) + 1e-30))
    energy = float(0.5 * delta @ (geo.fvp(delta) + damping * delta)
                   + geo.g @ delta)
    return delta, rel_kkt, energy


def natural_gradient_check(model, obs, pre_tanh, advantages, action_dim,
                           target_scale, tau, tau_ref, precision_beta,
                           precision_radius):
    """Compare a converged real PC update with the exact Gaussian Fisher CG solve.

    `Geometry` builds the policy-gradient Fisher on this exact rollout minibatch;
    it is not a diagonal or empirical approximation.  We run both target families
    because the configured Euclidean target and the optional output-Fisher target
    make distinct claims about natural-gradient behaviour.
    """
    geo = ng_probe.Geometry(
        model, obs, pre_tanh, advantages, action_dim, target_scale,
        dampings=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        cg_iters=800, cg_tol=1e-10)
    param_delta, param_kkt, param_energy = parameter_space_npg_energy(geo, 1.0)
    print("\n  parameter-space NPG energy (lambda=1): "
          f"relative KKT residual={param_kkt:.3e}, E={param_energy:.4e}")
    rows = []
    print("\n  exact Fisher/CG check on the real policy minibatch")
    print(f"  {'family':>20} {'cos(update,ref)':>15} {'cos_NG@1e-4':>12} "
          f"{'cosF@1e-4':>11} {'cos_NG*':>10} {'lambda*':>9}")
    families = []
    for label, natural_target in (("euclidean", False),
                                  ("natural_target", True)):
        target = gaussian_pc_targets(
            geo.out, pre_tanh, advantages, action_dim, target_scale,
            exp_std=True, natural_target=natural_target)
        families.append((label, target, {}))

    npg_output = settle_precision_outputs(
        geo.out, pre_tanh, advantages, action_dim=action_dim,
        beta=jnp.asarray(precision_beta, dtype=geo.out.dtype),
        max_t1=tau, target_scale=target_scale, exp_std=True)
    families.append(("precision_npg", npg_output["targets"], {
        "output_beta": float(npg_output["beta"]),
        "output_radius": float(npg_output["radius"]),
        "output_residual_rms": float(npg_output["residual_rms"]),
    }))

    tr_beta = trust_region_precision(
        geo.out, pre_tanh, advantages, action_dim, precision_radius,
        target_scale=target_scale, exp_std=True)
    tr_output = settle_precision_outputs(
        geo.out, pre_tanh, advantages, action_dim=action_dim,
        beta=tr_beta, max_t1=tau, target_scale=target_scale, exp_std=True)
    families.append(("precision_tr", tr_output["targets"], {
        "output_beta": float(tr_output["beta"]),
        "output_radius": float(tr_output["radius"]),
        "output_residual_rms": float(tr_output["residual_rms"]),
    }))

    near_damping = 1e-4
    for label, targets, metadata in families:
        acts, _ = settle(model, obs, targets, tau)
        acts_ref, _ = settle(model, obs, targets, tau_ref)
        update = -_flat(param_grad(model, acts, obs, targets))
        update_ref = -_flat(param_grad(model, acts_ref, obs, targets))
        cos_ng, lam = geo.best(update, geo.d_ng)
        cos_f = geo.cos_f(update, geo.d_ng[lam])
        cos_ng_near = _cos(update, geo.d_ng[near_damping])
        cos_f_near = geo.cos_f(update, geo.d_ng[near_damping])
        update_cos = _cos(update, update_ref)
        cos_parameter_energy = _cos(update, param_delta)
        print(f"  {label:>20} {update_cos:15.6f} {cos_ng_near:12.4f} "
              f"{cos_f_near:11.4f} {cos_ng:10.4f} {lam:9.0e}  "
              f"cosF_NG*={cos_f:.4f} cos(PC,param-E@1)="
              f"{cos_parameter_energy:.4f}")
        rows.append({"family": label, "update_ref_cos": update_cos,
                     "cos_ng": cos_ng, "cos_f_ng": cos_f,
                     "damping": lam,
                     "cos_ng_near_undamped": cos_ng_near,
                     "cos_f_ng_near_undamped": cos_f_near,
                     "near_undamped_damping": near_damping,
                     "cos_parameter_energy_lambda1": cos_parameter_energy,
                     "parameter_energy_lambda1_kkt": param_kkt,
                     **metadata})
    return rows


def diagnose_floor(model, obs, targets, tau, tols=(1e-3, 1e-4, 1e-5)):
    """Why does ||dF/dz|| stall well above zero once tau is past the knee?

    Two candidate causes, and they are distinguishable. If the floor is set by the
    solver's error control, tightening rtol/atol lowers it. If instead one layer
    contributes a constant the dynamics cannot remove, the per-layer breakdown shows
    the floor sitting in a single layer and tightening changes nothing.

    Tested at a tau just past the knee, not at the reference: step count grows like
    tau/tol, so tightening tolerance at a large tau costs enormously and measures
    the same floor.
    """
    r0 = residual(model, jpc.init_activities_with_ffwd(model=model, input=obs),
                  obs, targets)
    print(f"\n  residual floor diagnosis at tau={tau:g}")
    print(f"  {'rtol=atol':>10} {'resid/resid_0':>14} {'steps':>7}")
    for tol in tols:
        z, steps = settle(model, obs, targets, tau, rtol=tol, atol=tol)
        print(f"  {tol:10.0e} {residual(model, z, obs, targets)/r0:14.3e} {steps:7d}")
    z, _ = settle(model, obs, targets, tau)
    per = residual_per_layer(model, z, obs, targets)
    per0 = residual_per_layer(
        model, jpc.init_activities_with_ffwd(model=model, input=obs), obs, targets)
    print("  per-layer ||dF/dz_l||  (settled vs feedforward init):")
    for i, (a, b) in enumerate(zip(per, per0)):
        print(f"    layer {i}: {a:.4e}  (init {b:.4e})")


def sweep(model, obs, targets, taus, tau_ref, label):
    """Measure residual and weight-gradient convergence vs inference budget tau."""
    n = obs.shape[0]

    z_ref, steps_ref = settle(model, obs, targets, tau_ref)
    g_ref = _flat(param_grad(model, acts=z_ref, obs=obs, targets=targets))
    f_ref = _flat(z_ref)
    r0 = residual(model, jpc.init_activities_with_ffwd(model=model, input=obs),
                  obs, targets)
    resid_ref = residual(model, z_ref, obs, targets)

    # Is the reference itself settled? If cos(g(2*ref), g(ref)) is not ~1 the whole
    # table is measured against a moving target and means nothing.
    z_chk, _ = settle(model, obs, targets, 2.0 * tau_ref)
    g_chk = _flat(param_grad(model, acts=z_chk, obs=obs, targets=targets))
    ref_cos = _cos(g_chk, g_ref)

    print(f"\n  {label}:  N={n}  ||dF/dz||_ffwd={r0:.4e}")
    print(f"  reference tau={tau_ref:g} ({steps_ref} solver steps), "
          f"resid/resid_0={resid_ref/r0:.2e};  "
          f"cos(g(2*ref), g(ref))={ref_cos:.6f}")
    if ref_cos < 0.9999:
        print("  WARNING: reference is not itself converged -- raise --tau-ref")

    print(f"\n  {'tau':>9} {'max_t1 equiv':>13} {'resid/resid_0':>14} "
          f"{'cos(z,z*)':>10} {'cos(g,g*)':>11} {'|g|/|g*|':>9} "
          f"{'steps':>6} {'ms':>7}")
    rows = []
    for tau in taus:
        z, steps = settle(model, obs, targets, tau)
        g = _flat(param_grad(model, acts=z, obs=obs, targets=targets))
        jax.block_until_ready(g)
        # Time a second, warm call: the first traces and compiles this tau, which
        # at small tau costs orders of magnitude more than the solve itself.
        t0 = time.perf_counter()
        jax.block_until_ready(settle(model, obs, targets, tau)[0])
        ms = (time.perf_counter() - t0) * 1e3
        rr = residual(model, z, obs, targets) / r0
        cz = _cos(_flat(z), f_ref)
        cg = _cos(g, g_ref)
        gn = float(jnp.linalg.norm(g) / (jnp.linalg.norm(g_ref) + 1e-30))
        rows.append({"tau": tau, "resid_ratio": rr, "cos_z": cz, "cos_g": cg,
                     "gnorm_ratio": gn, "solver_steps": steps, "ms": ms})
        print(f"  {tau:9.4f} {tau*n:13.1f} {rr:14.3e} {cz:10.6f} {cg:11.6f} "
              f"{gn:9.3f} {steps:6d} {ms:7.1f}")

    print()
    for target in COS_TARGETS:
        hit = next((r["tau"] for r in rows if r["cos_g"] >= target), None)
        print(f"  cos(g,g*) >= {target:<6}: " +
              (f"tau >= {hit:g}  (max_t1 >= {hit*n:.0f} at this N)" if hit
               else "not reached on this grid"))
    hit = next((r["tau"] for r in rows if r["resid_ratio"] <= RESID_TARGET), None)
    print(f"  resid/resid_0 <= {RESID_TARGET:<6}: " +
          (f"tau >= {hit:g}" if hit else "not reached on this grid"))

    return {"label": label, "n": n, "resid_ffwd": r0, "tau_ref": tau_ref,
            "ref_self_cos": ref_cos, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--num-minibatches", type=int, default=4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--act-fn", default="relu")
    ap.add_argument("--target-scale", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=3.0e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--measure-at", type=int, nargs="+", default=[0, 20, 60],
                    help="training-update counts at which to run the tau sweep")
    ap.add_argument("--warmup-arm", default="settled",
                    choices=["settled", "unsettled"],
                    help="how the warmup updates are run: 'settled' uses the rate "
                         "correction at --warmup-max-t1, 'unsettled' reproduces the "
                         "committed runs (tau = max_t1/N)")
    ap.add_argument("--warmup-max-t1", type=float, default=20.0)
    ap.add_argument("--warmup-policy-geometry", default="euclidean",
                    choices=["euclidean", "precision_npg", "precision_tr"],
                    help="policy target family used to produce warmup checkpoints")
    ap.add_argument("--precision-beta", type=float, default=24.0)
    ap.add_argument("--precision-radius", type=float, default=0.01)
    ap.add_argument("--tau", type=float, nargs="+",
                    default=[0.0098, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 40])
    ap.add_argument("--tau-ref", type=float, default=100.0)
    ap.add_argument("--diagnose-floor", action="store_true",
                    help="test whether the residual floor is solver-tolerance-set")
    ap.add_argument("--floor-tau", type=float, default=5.0,
                    help="tau at which to test the residual floor (just past the knee)")
    ap.add_argument("--ng-probe", action="store_true",
                    help="compare converged policy updates to exact Fisher/CG "
                         "natural gradients on the same rollout minibatch")
    ap.add_argument("--ng-tau", type=float, default=20.0,
                    help="rate-corrected inference time for the Fisher check")
    ap.add_argument("--ng-tau-ref", type=float, default=100.0,
                    help="long-horizon update reference for the Fisher check")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" /
                                        "settling_budget" / "settling_budget.json"))
    args = ap.parse_args()

    print(f"jax {jax.__version__}  devices={jax.devices()}")
    print(f"env={args.env} num_envs={args.num_envs} rollout={args.rollout} "
          f"minibatches={args.num_minibatches} width={args.width} "
          f"depth={args.depth} act={args.act_fn} ts={args.target_scale}")
    print(f"warmup arm: {args.warmup_arm} (max_t1={args.warmup_max_t1:g}); "
          f"measuring at updates {args.measure_at}")

    key = jr.PRNGKey(args.seed)
    key, k_policy, k_value, k_envs = jr.split(key, 4)

    envs = make_vec_env(EnvConfig(env_name=args.env, num_envs=args.num_envs,
                                  episode_length=1000))
    envs.seed(int(k_envs[0]))
    env_state = envs.reset()
    action_size = envs.action_space.n
    obs_dim = int(np.prod(env_state.obs.shape[1:]))
    print(f"obs_dim={obs_dim} action_dim={action_size}")

    policy_model = jpc.make_mlp(k_policy, input_dim=obs_dim, width=args.width,
                               depth=args.depth, output_dim=2 * action_size,
                               act_fn=args.act_fn, use_bias=True)
    value_model = jpc.make_mlp(k_value, input_dim=obs_dim, width=args.width,
                               depth=args.depth, output_dim=1,
                               act_fn=args.act_fn, use_bias=True)
    policy_optim = optax.adam(args.learning_rate)
    policy_opt_state = policy_optim.init(
        (eqx.filter(policy_model, eqx.is_array), None))
    value_optim = optax.adam(args.learning_rate)
    value_opt_state = value_optim.init(
        (eqx.filter(value_model, eqx.is_array), None))

    @eqx.filter_jit
    def pcn_forward(model, obs):
        return jpc.init_activities_with_ffwd(model=model, input=obs)[-1]

    def flat_obs(obs, update=True):
        raw = np.asarray(obs).reshape(obs.shape[0], -1).astype(np.float32)
        try:
            return envs.normalize_obs(raw, update=update)
        except TypeError:
            return envs.normalize_obs(raw)

    # The warmup step reuses the production step function so the trajectory this
    # probe measures on is produced by the same code the runs use.
    from pc_algorithms.inference import make_pc_step_at_rate
    settled_warmup = args.warmup_arm == "settled"

    def pc_step(model, optim, opt_state, output, inp):
        if settled_warmup:
            return make_pc_step_at_rate(model=model, optim=optim,
                                        opt_state=opt_state, output=output,
                                        input=inp, max_t1=args.warmup_max_t1)
        return jpc.make_pc_step(model=model, optim=optim, opt_state=opt_state,
                                output=output, input=inp,
                                max_t1=args.warmup_max_t1)

    def collect():
        """One real rollout -> (observations, pre_tanh, advantages, value_targets)."""
        nonlocal env_state, key
        obs_buf, pre_buf, rew_buf, done_buf, next_buf = [], [], [], [], []
        for _ in range(args.rollout):
            obs = flat_obs(env_state.obs)
            params = pcn_forward(policy_model, jnp.asarray(obs))
            key, k_act = jr.split(key)
            actions, pre_tanh = sample_gaussian_action(
                k_act, params, action_size, exp_std=True)
            nstate = envs.step(np.asarray(actions))
            reward = nstate.reward
            if hasattr(envs, "normalize_reward"):
                reward = envs.normalize_reward(reward, nstate.done, args.gamma)
            obs_buf.append(obs)
            pre_buf.append(np.asarray(pre_tanh))
            rew_buf.append(reward)
            done_buf.append(nstate.done)
            next_buf.append(flat_obs(nstate.obs, update=False))
            env_state = nstate

        obs_arr = np.stack(obs_buf)
        observations = obs_arr.reshape(-1, obs_arr.shape[-1])
        next_observations = np.stack(next_buf).reshape(-1, obs_arr.shape[-1])
        values = np.asarray(pcn_forward(
            value_model, jnp.asarray(observations))).squeeze(-1)
        next_values = np.asarray(pcn_forward(
            value_model, jnp.asarray(next_observations))).squeeze(-1)
        adv, vtarg = compute_gae(
            np.stack(rew_buf), np.stack(done_buf),
            values.reshape(args.rollout, args.num_envs),
            next_values.reshape(args.rollout, args.num_envs),
            args.gamma, args.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return (observations, np.stack(pre_buf).reshape(-1, action_size), adv, vtarg)

    def minibatch(data, idx):
        observations, pre_tanh, adv, vtarg = data
        mb_obs = jnp.asarray(observations[idx])
        mb_adv = jnp.asarray(adv[idx])
        params_mb = pcn_forward(policy_model, mb_obs)
        mb_pre_tanh = jnp.asarray(pre_tanh[idx])
        if args.warmup_policy_geometry == "precision_npg":
            policy_targets = settle_precision_outputs(
                params_mb, mb_pre_tanh, mb_adv, action_dim=action_size,
                beta=jnp.asarray(args.precision_beta, dtype=params_mb.dtype),
                max_t1=args.warmup_max_t1, target_scale=args.target_scale,
                exp_std=True)["targets"]
        elif args.warmup_policy_geometry == "precision_tr":
            beta = trust_region_precision(
                params_mb, mb_pre_tanh, mb_adv, action_size,
                args.precision_radius, target_scale=args.target_scale,
                exp_std=True)
            policy_targets = settle_precision_outputs(
                params_mb, mb_pre_tanh, mb_adv, action_dim=action_size,
                beta=beta, max_t1=args.warmup_max_t1,
                target_scale=args.target_scale, exp_std=True)["targets"]
        else:
            policy_targets = gaussian_pc_targets(
                params_mb, mb_pre_tanh, mb_adv, action_size,
                args.target_scale, exp_std=True)
        value_targets = jnp.asarray(vtarg[idx])[:, None]
        return (mb_obs, policy_targets, value_targets,
                jnp.asarray(pre_tanh[idx]), mb_adv)

    out = {"config": vars(args), "obs_dim": obs_dim,
           "action_dim": action_size, "measurements": []}
    updates_done = 0

    for target_update in sorted(args.measure_at):
        while updates_done < target_update:
            data = collect()
            batch = data[0].shape[0]
            mb = max(1, args.num_minibatches)
            usable = (batch // mb) * mb
            for _ in range(2):                       # update_epochs = 2
                key, k_perm = jr.split(key)
                perm = np.asarray(jr.permutation(k_perm, batch))[:usable]
                for idx in perm.reshape(mb, -1):
                    mb_obs, p_targ, v_targ, _, _ = minibatch(data, idx)
                    vr = pc_step(value_model, value_optim, value_opt_state,
                                 v_targ, mb_obs)
                    value_model, value_opt_state = vr["model"], vr["opt_state"]
                    pr = pc_step(policy_model, policy_optim, policy_opt_state,
                                 p_targ, mb_obs)
                    policy_model, policy_opt_state = pr["model"], pr["opt_state"]
            updates_done += 1
            if updates_done % 10 == 0:
                print(f"  ... warmup update {updates_done}")

        data = collect()
        batch = data[0].shape[0]
        mb = max(1, args.num_minibatches)
        idx = np.arange((batch // mb) * mb).reshape(mb, -1)[0]
        mb_obs, p_targ, v_targ, mb_pre_tanh, mb_adv = minibatch(data, idx)

        print("\n" + "=" * 98)
        print(f"after {updates_done} real updates "
              f"({updates_done * args.num_envs * args.rollout:,} env steps), "
              f"warmup arm = {args.warmup_arm}")
        print("=" * 98)
        pol = sweep(policy_model, mb_obs, p_targ, args.tau, args.tau_ref, "policy")
        if args.diagnose_floor:
            diagnose_floor(policy_model, mb_obs, p_targ, args.floor_tau)
        val = sweep(value_model, mb_obs, v_targ, args.tau, args.tau_ref, "value")
        if args.diagnose_floor:
            diagnose_floor(value_model, mb_obs, v_targ, args.floor_tau)
        ng = None
        if args.ng_probe:
            ng = natural_gradient_check(
                policy_model, mb_obs, mb_pre_tanh, mb_adv, action_size,
                args.target_scale, args.ng_tau, args.ng_tau_ref,
                args.precision_beta, args.precision_radius)
        out["measurements"].append({"updates": updates_done,
                                    "env_steps": updates_done * args.num_envs * args.rollout,
                                    "policy": pol, "value": val,
                                    "natural_gradient": ng})

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
