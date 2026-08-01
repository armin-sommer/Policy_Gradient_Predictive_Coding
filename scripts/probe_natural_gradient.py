"""Does the PC weight update implement a natural gradient? A direction-level probe.

The claim under test ("PC update = natural gradient at inference convergence")
is a statement about the RAW weight gradient that comes out of settled PC
inference — the direction SGD realises. This script measures that direction on
a frozen (policy, batch) and compares it against every reference direction it
could plausibly be tracking. No training, no env, CPU-only.

Why this decomposition. For the Gaussian policy head the parameter-space
Fisher of the policy-gradient surrogate

    L(theta) = -(ts/N) sum_i A_i log N(z_i; mu_theta(s_i), sigma_theta(s_i)^2)

factors as F = (1/N) sum_i J_i^T F_out,i J_i with J_i = d out_i / d theta and a
closed-form output metric F_out = diag(1/sigma^2, 2) (mu / log_std halves). A
natural gradient F^-1 grad L therefore needs two ingredients, and the two
knobs of this codebase supply (or fail to supply) one each:

  * output metric F_out^-1  <- the TARGET family. Euclidean targets
    (natural_target=False) keep the 1/sigma^2 variance division = raw score;
    natural targets (natural_target=True) pre-multiply by F_out^-1.
  * network factor          <- INFERENCE, which reshapes how the output error
    maps into parameters as activities settle.

Reference directions, all on the same frozen (theta, batch), all descent
directions:

  policy-gradient side (what the update SHOULD be, per hypothesis):
    d_SGD = -grad L                        vanilla policy gradient (Euclidean)
    d_OUT = -J^T P u,  P = [sigma^2, 1/2]  output-space natural ONLY: exactly
                                           what the natural target encodes
    d_NG  = -(F + lam I)^-1 grad L         the natural gradient (both factors)
  energy side (what the PC machinery could actually compute):
    d_BP  = -grad_theta E_composed         backprop on the target energy (the
                                           Millidge "PC ~ BP" endpoint); equals
                                           the intended gradient up to the
                                           log_std target-clip truncation
    d_EQ  = -grad_theta [ (1/2N) sum_i     the equilibrium prediction: PC at
            r_i^T S_i^-1 r_i ],            settled inference = BP on a RESCALED
            r_i = target_i - out_i,        loss with output metric
            S_i = I + sum_l B_l,i B_l,i^T  S = I + sum_l J_l J_l^T, B_l,i =
            (S_i stop-gradiented)          d out_i / d z_l,i. Exact for linear
                                           nets (Innocenti et al., 2305.18188),
                                           first-order for tanh.

F is the exact Fisher (deterministic GGN — no action sampling), applied
matrix-free via jvp/vjp + CG; the damping lam is swept and the best-fit lam is
part of the measurement, not a nuisance. S_i is built exactly per sample
(output dim is small).

What Delta(t1) = -dE/dtheta(settled activities) does mechanically, measured:

  t1 = 0        At exactly-feedforward activities every hidden error is zero,
                so the update touches ONLY the final layer: a delta rule
                pushing the output layer at the (clipped) targets. This is the
                anchor: share_last = 1.0000 and the final-layer block equals
                d_BP's final-layer block exactly.
  t1/N small    jpc's energy is batch-normalised (F = 1/2N sum_i ...), so the
                inference ODE dz/dt = -dF/dz runs N times SLOWER than its
                per-sample dynamics: reaching per-sample time T needs
                t1 = N*T. Hidden layers acquire gradient mass ~ t1/N. THE
                BENCH OPERATING POINT (minibatch N=2048, max_t1=20) IS
                t1/N ~ 0.01: a last-layer delta rule with ~1% corrections.
  t1/N >~ 10    Inference equilibrates (resid -> 0) and the update should
                track d_EQ if the linearised equilibrium theory holds. jpc
                hard-stops integration at t1=4096
                (steady_state_event_with_timeout), so t1 beyond that is
                silently truncated — and equilibrating a bench minibatch
                would need t1 ~ 20k-80k, far beyond the stop.

Hypothesis grid — what "has a natural gradient" would require:

                          | Euclidean target       | natural target
  inference inert (t/N~0) | last-layer delta rule  | last-layer delta rule
  fixed-prediction BP     | d_BP ~ d_SGD           | d_BP ~ d_OUT
  equilibrium, linearised | d_EQ                   | d_EQ
  the claim               | d_EQ ~ d_NG?           | d_EQ ~ d_NG?  <- printed
                                                      as the 'ceiling' line

The Fisher-metric cosine cos_F is the parameterisation-invariant alignment
(cos_F(Delta, d_NG) = 1 iff Delta buys the most L-improvement per unit KL);
Euclidean cosines are reported for interpretability.

Everything mirrors the production path: targets from
`gaussian_policy.gaussian_pc_targets` (target_clip=None — the non-clipped
version), inference and weight gradients via jpc's own
`init_activities_with_ffwd -> solve_inference -> compute_pc_param_grads`
(Heun + PID(1e-3,1e-3): the exact `make_pc_step` chain minus the optimizer).
Note `jpc.make_mlp(depth=2)` — the bench Config — builds ONE hidden layer
(obs -> width -> out), so there is a single free activity layer.

Regimes position sigma where the geometry does / does not discriminate:
  init  : sigma ~ 1     -> F_out ~ [1, 2]: output metrics nearly collinear;
                           the network factor still separates the references.
  mixed : per-dim log_std bias U[-1.8, 0.3] -> sigma spread ~ [0.17, 1.35];
                           the late-training mix of floored and healthy dims.
  floor : log_std bias -1.95 -> sigma ~ 0.14: the 1/sigma^2 ~ 49x amplifier
                           regime where Euclidean and natural maximally differ.

Run:  python scripts/probe_natural_gradient.py                  # full pilot
      python scripts/probe_natural_gradient.py --regimes floor --n 64
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jpc
from jax.flatten_util import ravel_pytree

from pc_algorithms.gaussian_policy import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    gaussian_pc_targets,
    split_gaussian_params,
)

jax.config.update("jax_enable_x64", True)

JPC_T_HARD_STOP = 4096  # jpc's steady_state_event_with_timeout fires at t>=4096


def to_f64(tree):
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, tree)


def forward(model, obs):
    """Feedforward pass, identical to jpc.init_activities_with_ffwd()[-1]."""
    h = obs
    for layer in model:
        h = jax.vmap(layer)(h)
    return h


def build_model(key, obs_dim, action_dim, width, depth, act_fn):
    model = jpc.make_mlp(key, input_dim=obs_dim, width=width, depth=depth,
                         output_dim=2 * action_dim, act_fn=act_fn, use_bias=True)
    return to_f64(model)


def set_log_std_bias(model, key, regime, action_dim):
    """Position the sigma head: the regime is the independent variable."""
    if regime == "init":
        return model
    if regime == "fixed":
        # CONTROL: state-INDEPENDENT sigma, i.e. PPO's parameterisation and the
        # one the experiment log 3.4 (`state_indep_std`) switches on. Zeroing the
        # log_std rows of the final weight makes sigma = exp(bias) constant across
        # states, so the output precision F_out = diag(1/sigma^2, 2) is no longer a
        # function of the inference variable. bias = 0 -> sigma = 1, comfortably
        # inside [LOG_STD_MIN, LOG_STD_MAX], so the clamp is inactive and the F5
        # fidelity loss is switched off too. This is the clean case (no
        # parameter-dependent precision, no clamp truncation) against which the
        # heteroscedastic regimes' cost is read.
        lin_w = model[-1].layers[1].weight
        model = eqx.tree_at(
            lambda m: m[-1].layers[1].weight, model,
            lin_w.at[action_dim:, :].set(0.0))
        bias0 = model[-1].layers[1].bias
        return eqx.tree_at(
            lambda m: m[-1].layers[1].bias, model,
            bias0.at[action_dim:].set(0.0))
    if regime == "mixed":
        delta = jr.uniform(key, (action_dim,), minval=-1.8, maxval=0.3)
    elif regime == "floor":
        delta = jnp.full((action_dim,), -1.95)
    else:
        raise ValueError(f"unknown regime: {regime}")
    bias = model[-1].layers[1].bias
    new_bias = bias.at[action_dim:].set(delta.astype(bias.dtype))
    return eqx.tree_at(lambda m: m[-1].layers[1].bias, model, new_bias)


def make_batch(key, model, n, obs_dim, action_dim):
    """On-policy synthetic batch: obs ~ N(0,1) (Welford-normalised inputs),
    z from the policy itself, advantages standardised (bench normalises)."""
    k_obs, k_act, k_adv = jr.split(key, 3)
    obs = jr.normal(k_obs, (n, obs_dim))
    out = forward(model, obs)
    loc, scale, _ = split_gaussian_params(out, action_dim, exp_std=True)
    z = loc + scale * jr.normal(k_act, loc.shape)
    adv = jr.normal(k_adv, (n,))
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return obs, z, adv


def logp_sum(out, z, action_dim):
    """log N(z; mu, sigma^2) (const dropped), SAME clipped parameterisation as
    production sampling/targets, so grad wrt out matches the Euclidean offsets."""
    loc, scale, log_scale = split_gaussian_params(out, action_dim, exp_std=True)
    return jnp.sum(-0.5 * jnp.square((z - loc) / scale) - log_scale, axis=-1)


def _cos(a, b):
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-300))


def _tree_norm(tree):
    return float(jnp.sqrt(sum(
        jnp.sum(jnp.square(l)) for l in jax.tree_util.tree_leaves(tree))))


def layer_blocks(flat, unravel):
    """Split a flat parameter vector back into per-model-layer blocks."""
    return [ravel_pytree(sub)[0] for sub in unravel(flat)]


class Geometry:
    """Frozen (theta, batch) geometry: forward jvp/vjp, exact Fisher matvec,
    policy-gradient reference directions, damped CG solves."""

    def __init__(self, model, obs, z, adv, action_dim, ts,
                 dampings, cg_iters, cg_tol):
        self.dampings, self.cg_iters, self.cg_tol = dampings, cg_iters, cg_tol
        self.arrays, self.static = eqx.partition(model, eqx.is_array)
        flat0, self.unravel = ravel_pytree(self.arrays)
        self.n_params, self.n = flat0.size, obs.shape[0]
        self.obs = obs

        def fwd(a):
            return forward(eqx.combine(a, self.static), obs)

        self.fwd = fwd
        out, self._vjp = jax.vjp(fwd, self.arrays)
        self.out = out

        def loss_out(o):
            return -ts * jnp.mean(adv * logp_sum(o, z, action_dim))

        self.u = jax.grad(loss_out)(out)                    # dL/dout per sample
        self.g = ravel_pytree(self._vjp(self.u)[0])[0]      # grad_theta L

        _, scale, _ = split_gaussian_params(out, action_dim, exp_std=True)
        raw_ls = out[..., action_dim:]
        in_bounds = ((raw_ls > LOG_STD_MIN)
                     & (raw_ls < LOG_STD_MAX)).astype(out.dtype)
        # exact head Fisher wrt raw outputs (clip has zero grad outside bounds)
        fdiag = jnp.concatenate(
            [1.0 / jnp.square(scale), 2.0 * in_bounds], axis=-1)
        # production's F_out^-1 = [sigma^2, 1/2], UNMASKED (gaussian_pc_targets)
        precond = jnp.concatenate(
            [jnp.square(scale), 0.5 * jnp.ones_like(scale)], axis=-1)

        self.precond = precond
        self.d_sgd = -self.g
        self.d_out = -ravel_pytree(self._vjp(precond * self.u)[0])[0]
        self.frac_ls_out_of_bounds = float(1.0 - in_bounds.mean())
        self.sigma_min, self.sigma_max = float(scale.min()), float(scale.max())

        def matvec(v, metric):
            tangent = jax.jvp(fwd, (self.arrays,), (self.unravel(v),))[1]
            return ravel_pytree(self._vjp(metric * tangent / self.n)[0])[0]

        self.fvp = jax.jit(lambda v: matvec(v, fdiag))

        self.d_ng = {lam: self._solve(self.fvp, self.g, lam)
                     for lam in dampings}

    def _solve(self, op, rhs, lam):
        sol, _ = jax.scipy.sparse.linalg.cg(
            lambda v: op(v) + lam * v, rhs, tol=self.cg_tol,
            maxiter=self.cg_iters)
        return -sol

    def cos_f(self, a, b):
        """Fisher-metric cosine — the parameterisation-invariant alignment."""
        fa, fb = self.fvp(a), self.fvp(b)
        return float(a @ fb / (jnp.sqrt((a @ fa) * (b @ fb)) + 1e-300))

    def best(self, delta, sols):
        return max((_cos(delta, d), lam) for lam, d in sols.items())


def energy_references(geo, model, targets):
    """d_BP (composed backprop on the target energy, via jpc's own energy fn)
    and d_EQ (the linearised equilibrium prediction: BP on the S^-1-rescaled
    loss, S_i = I + sum_l B_l,i B_l,i^T built exactly per sample)."""

    def e_composed(a):
        m = eqx.combine(a, geo.static)
        acts = jpc.init_activities_with_ffwd(model=m, input=geo.obs)
        return jpc.pc_energy_fn(params=(m, None), activities=acts,
                                y=targets, x=geo.obs)

    d_bp = -ravel_pytree(jax.grad(e_composed)(geo.arrays))[0]

    acts = jpc.init_activities_with_ffwd(model=model, input=geo.obs)
    out_dim = targets.shape[-1]
    s = jnp.tile(jnp.eye(out_dim, dtype=targets.dtype),
                 (geo.n, 1, 1))
    # free activities are z_0..z_{L-2}; z_{L-1} never enters jpc's energy
    for l in range(len(model) - 1):
        def tail(z_l, _l=l):
            h = z_l
            for layer in model[_l + 1:]:
                h = layer(h)
            return h
        b = jax.vmap(jax.jacrev(tail))(acts[l])        # (n, out, width_l)
        s = s + jnp.einsum("nod,npd->nop", b, b)
    s_inv = jax.lax.stop_gradient(jnp.linalg.inv(s))

    def e_rescaled(a):
        r = targets - geo.fwd(a)
        return 0.5 * jnp.mean(jnp.einsum("no,nop,np->n", r, s_inv, r))

    d_eq = -ravel_pytree(jax.grad(e_rescaled)(geo.arrays))[0]
    return d_bp, d_eq


def pc_update(model, obs, targets, t1):
    """The production update direction: jpc's make_pc_step chain minus the
    optimizer. Returns (Delta flat, activity residual, energy, z-shift)."""
    acts0 = jpc.init_activities_with_ffwd(model=model, input=obs)
    acts = acts0
    if t1 > 0:
        acts = jpc.solve_inference(
            params=(model, None), activities=acts0, output=targets,
            input=obs, max_t1=int(t1))
        acts = jax.tree_util.tree_map(lambda a: a[0], acts)
    grads = jpc.compute_pc_param_grads(
        params=(model, None), activities=acts, y=targets, x=obs)[0]
    delta = -ravel_pytree(eqx.filter(grads, eqx.is_array))[0]
    energy, a_grad = jpc.compute_pc_activity_grad(
        params=(model, None), activities=acts, y=targets, x=obs)
    shift = (_tree_norm([a - b for a, b in zip(acts, acts0)])
             / (_tree_norm(acts0) + 1e-300))
    return delta, _tree_norm(a_grad), float(energy), shift


def target_fidelity(out, targets, u_intended, action_dim, n):
    """How faithfully the (clipped) target encodes its INTENDED gradient.
    u_eff = (out - target)/N is the output error the energy actually descends;
    u_intended is the family's intended preconditioned score (u for Euclidean,
    P*u for natural). Reported per output half — the log_std clip in
    gaussian_pc_targets truncates the log_std channel only."""
    u_eff = (out - targets) / n
    d = action_dim
    cos_mu = _cos(u_eff[:, :d].ravel(), u_intended[:, :d].ravel())
    cos_ls = _cos(u_eff[:, d:].ravel(), u_intended[:, d:].ravel())
    return cos_mu, cos_ls


def run_probe(args):
    for regime in args.regimes:
        key = jr.PRNGKey(args.seed)
        k_model, k_bias, k_batch = jr.split(key, 3)
        model = build_model(k_model, args.obs_dim, args.action_dim,
                            args.width, args.depth, args.act_fn)
        model = set_log_std_bias(model, k_bias, regime, args.action_dim)
        obs, z, adv = make_batch(k_batch, model, args.n,
                                 args.obs_dim, args.action_dim)

        out0 = forward(model, obs)
        ffwd = jpc.init_activities_with_ffwd(model=model, input=obs)[-1]
        assert float(jnp.max(jnp.abs(out0 - ffwd))) < 1e-10, \
            "forward() does not match jpc's feedforward pass"

        geo = Geometry(model, obs, z, adv, args.action_dim, args.ts,
                       args.dampings, args.cg_iters, args.cg_tol)

        print(f"\n{'=' * 100}")
        print(f"regime={regime}  seed={args.seed}  n={args.n}  "
              f"params={geo.n_params}  layers={len(model)}  "
              f"sigma=[{geo.sigma_min:.3f}, {geo.sigma_max:.3f}]  "
              f"raw log_std out of bounds: "
              f"{100 * geo.frac_ls_out_of_bounds:.1f}%")
        lam_mid = args.dampings[len(args.dampings) // 2]
        print(f"policy-gradient reference separations (lam={lam_mid:g}):  "
              f"cos(d_SGD,d_NG)={_cos(geo.d_sgd, geo.d_ng[lam_mid]):.3f}  "
              f"cos(d_OUT,d_NG)={_cos(geo.d_out, geo.d_ng[lam_mid]):.3f}  "
              f"cos(d_SGD,d_OUT)={_cos(geo.d_sgd, geo.d_out):.3f}")

        for natural in (False, True):
            targets = gaussian_pc_targets(
                out0, z, adv, args.action_dim, args.ts, exp_std=True,
                target_clip=None, natural_target=natural)
            fam = ("natural (no 1/sigma^2)" if natural
                   else "Euclidean (with 1/sigma^2)")
            intended = geo.d_out if natural else geo.d_sgd
            intended_name = "d_OUT" if natural else "d_SGD"

            d_bp, d_eq = energy_references(geo, model, targets)
            u_intended = geo.precond * geo.u if natural else geo.u
            cos_mu, cos_ls = target_fidelity(
                out0, targets, u_intended, args.action_dim, args.n)
            c_ceil, lam_ceil = geo.best(d_eq, geo.d_ng)
            c_ceil_bp, lam_ceil_bp = geo.best(d_bp, geo.d_ng)

            print(f"\n-- target family: {fam}")
            print(f"   target fidelity (u_eff vs dL/dout): mu-half "
                  f"{cos_mu:.4f}, log_std-half {cos_ls:.4f} "
                  f"(log_std target-clip truncation)")
            print(f"   cos(d_BP, {intended_name})={_cos(d_bp, intended):.4f}   "
                  f"ceilings: cos(d_EQ, d_NG)={c_ceil:.4f} "
                  f"(lam*={lam_ceil:.0e}), cos(d_BP, d_NG)={c_ceil_bp:.4f} "
                  f"(lam*={lam_ceil_bp:.0e})")
            print(f"{'t1':>6} {'t1/N':>8} {'resid':>10} {'energy':>11} "
                  f"{'z-shift':>8} {'|Delta|':>10} {'shL':>6} "
                  f"{'cos_BP':>8} {'cos_EQ':>8} "
                  f"{'cos_SGD':>8} {'cos_OUT':>8} {'cos_NG*':>8} {'lam*':>8} "
                  f"{'cosF_NG*':>9}")
            resid0, delta = None, None
            for t1 in args.t1:
                if t1 > JPC_T_HARD_STOP:
                    print(f"{t1:>6} skipped: jpc hard-stops inference at "
                          f"t={JPC_T_HARD_STOP}")
                    continue
                try:
                    delta, resid, energy, shift = pc_update(
                        model, obs, targets, t1)
                except Exception as e:  # stiff regimes can exhaust the solver
                    print(f"{t1:>6} solver failed: {type(e).__name__}: {e}")
                    continue
                if resid0 is None:
                    resid0 = resid if resid > 0 else 1.0
                blocks = layer_blocks(delta, geo.unravel)
                sh_last = float(jnp.linalg.norm(blocks[-1])
                                / (jnp.linalg.norm(delta) + 1e-300))
                c_ng, lam_n = geo.best(delta, geo.d_ng)
                cf = geo.cos_f(delta, geo.d_ng[lam_n])
                print(f"{t1:>6} {t1 / args.n:>8.3f} {resid / resid0:>10.2e} "
                      f"{energy:>11.4e} {shift:>8.4f} "
                      f"{float(jnp.linalg.norm(delta)):>10.3e} {sh_last:>6.3f} "
                      f"{_cos(delta, d_bp):>8.4f} {_cos(delta, d_eq):>8.4f} "
                      f"{_cos(delta, geo.d_sgd):>8.4f} "
                      f"{_cos(delta, geo.d_out):>8.4f} "
                      f"{c_ng:>8.4f} {lam_n:>8.1e} {cf:>9.4f}")
                if t1 == 0:
                    # anchor: at ffwd activities the update is final-layer only
                    bp_blocks = layer_blocks(d_bp, geo.unravel)
                    print(f"       anchor t1=0: share_last={sh_last:.6f} "
                          f"(expect 1.0), cos(last-layer block vs d_BP)="
                          f"{_cos(blocks[-1], bp_blocks[-1]):.6f} "
                          f"(expect 1.0)")
            if delta is not None:
                blocks = layer_blocks(delta, geo.unravel)
                ng_blocks = layer_blocks(geo.d_ng[lam_mid], geo.unravel)
                eq_blocks = layer_blocks(d_eq, geo.unravel)
                per_layer = "  ".join(
                    f"L{i}[share {float(jnp.linalg.norm(b) / (jnp.linalg.norm(delta) + 1e-300)):.2f} "
                    f"cosEQ {_cos(b, eb):.3f} cosNG {_cos(b, nb):.3f}]"
                    for i, (b, eb, nb) in enumerate(
                        zip(blocks, eq_blocks, ng_blocks)))
                print(f"       per-layer at t_last: {per_layer}")
                lam_line = "  ".join(
                    f"{lam:.0e}:{_cos(delta, geo.d_ng[lam]):.3f}"
                    for lam in args.dampings)
                print(f"       cos(Delta(t_last), d_NG[lam]) by lam:  {lam_line}")

    print(f"""
reading:
  anchor   : at t1=0 the update must be final-layer-only (share_last = 1.0)
             with the final-layer block exactly on d_BP's — validates the
             harness against jpc's own energy.
  fidelity : cos(d_BP, d_SGD/d_OUT) < 1 measures how much the log_std target
             clip distorts what the target family means to encode — a property
             of the production target, not of the probe.
  timescale: t1/N is per-sample inference time; the energy is batch-normalised
             so settling needs t1/N ~ 10-40. The bench operating point is
             t1/N = 20/2048 = 0.01, and jpc hard-stops at t1={JPC_T_HARD_STOP},
             so a bench minibatch (N=2048) CANNOT be equilibrated in-training.
  claim    : "PC update = natural gradient" requires (a) rows moving onto d_EQ
             as resid -> 0 (equilibrium theory holds), AND (b) the d_EQ~d_NG
             ceiling being high — for the natural family. If rows never leave
             the t1=0 anchor, inference contributes nothing and any 'natural
             gradient' is exactly the target preconditioning, nothing more.""")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # bench-winner geometry: halfcheetah, width 64, depth 2 (ONE hidden layer),
    # tanh, ts=1.0, exp_std
    ap.add_argument("--obs-dim", type=int, default=17)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--act-fn", default="tanh")
    ap.add_argument("--ts", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=256,
                    help="batch size; inference timescale is proportional to "
                         "this (bench minibatch is 2048)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t1", type=int, nargs="+",
                    default=[0, 1, 5, 20, 80, 320, 1280, 4096],
                    help="inference integration times; 20 = bench operating "
                         "point; jpc hard-stops at 4096")
    ap.add_argument("--regimes", nargs="+",
                    default=["init", "mixed", "floor", "fixed"],
                    choices=["init", "mixed", "floor", "fixed"],
                    help="'fixed' is the state-independent-sigma control.")
    ap.add_argument("--dampings", type=float, nargs="+",
                    default=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    ap.add_argument("--cg-iters", type=int, default=400)
    ap.add_argument("--cg-tol", type=float, default=1e-12)
    run_probe(ap.parse_args())


if __name__ == "__main__":
    main()
