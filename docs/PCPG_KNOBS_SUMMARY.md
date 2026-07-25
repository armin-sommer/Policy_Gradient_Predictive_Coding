# PCPG HalfCheetah knob summary

A consolidated map of every knob probed on the HalfCheetah PCPG actor-critic
benchmark, what each one *means*, and what the sweeps *told us*. Sources: the
stability-flags reference ([`PCPG_STABILITY_FLAGS.md`](PCPG_STABILITY_FLAGS.md)),
the four `results/trust_region_kl_*` sweeps, the `trust_region_kl_natural`
benchmark table, and the seed-level collapse analysis of the `_nat` runs.

All numbers are HalfCheetah, bench tier (1M steps, [64,64] net, 256 envs, tanh),
3 seeds unless noted. "collapse" = a seed whose final return craters after peaking.

## The core PC update

Every knob acts at one point in this chain:

```
advantage A ─▶ target = μ + ts·A·(z−μ)/σ²  ─▶ PC inference (max_t1 steps) ─▶ weight grad ─▶ optimizer step
                    │          │        │                                          │           │
             natural_target  target_  log_std_min                          max_grad_norm   adam/sgd
                             clip     (σ floor)                            (SGD only)
```

PCPG converts "move the policy along the advantage" into a **target** for the
network outputs, settles the latent with a few **predictive-coding inference
steps**, then steps the weights toward it. Knobs either reshape the target,
control inference, or control the step.

## Master knob table

| Knob | Meaning | Values tried | What it told us |
|---|---|---|---|
| `optimizer` | How weight-grads become steps. **Adam** rescales each parameter adaptively (renormalizes magnitudes); **SGD** takes the raw scaled step. | adam, sgd | **Pivotal.** With the natural target, SGD is the only clean recipe (`781±47`, 0/3). Adam keeps collapsing. |
| `natural_target` (`_nat`) | Drops the `1/σ²` amplifier from the target → Fisher-preconditioned ("natural gradient") target; acts as an implicit trust region. | on/off | **The unlock — but only under SGD.** SGD+nat: KL≈0.04, no collapse. Adam+nat: KL≈0.30, still collapses. An **interaction**, not a standalone fix. |
| `max_t1` | Number of predictive-coding **inference iterations** per update (how long the latent settles). More = "more inference." | 10, 20, 40, 80 | **More inference is not the lever.** 20 is the sweet spot; 80 just adds variance (SGD mt80 = `181/−62/954`). Consistent across all sweeps. |
| `target_scale` (`ts`) | How far the target shifts the mean per unit advantage — target aggressiveness. | 0.3–1.0 | Higher = faster but hotter. ts=1.0 works at bench; flagged as a scaling risk (try 0.7 at 5M). |
| `learning_rate` (actor) | Policy-net step size. Naming: **`lr003`=0.03**, `lr0003`=0.003. | 3e-4 … 3e-2 | SGD needs 0.03 to learn; that's why it's aggressive at scale (test 0.01 before 5M). |
| `value_learning_rate` (critic) | Value-net step size. | 1e-4 … 3e-4 | Under-examined, but **SGD's failure mode is critic divergence** (value-EV→−3) → prime suspect knob. *(sweep pending — see below)* |
| `target_clip` (`_tclip`) | Hard cap on the per-coord mean-target offset — an *output-space* trust region. Optimizer-agnostic. | 2, 5 | Lifts Adam's peaks (~1000) but **does not remove collapse** (still 1/3). Trust region in the wrong space. |
| `target_clip_rel` | Makes the cap relative to σ (`|Δμ|≤clip·σ`). | on/off | Not separately conclusive here. |
| `log_std_min` (`_smin`) | Floor on policy std σ. Caps the `1/σ²` amplifier and stops exploration vanishing. | −2 (default) | Directly targets Adam's entropy/saturation drift. *(sweep pending — see below)* |
| `max_grad_norm` (`_clip`) | Global-norm clip on the policy gradient. **Adam renormalizes it away**; mainly bites SGD. | None, 0.5, 1 | **Kills SGD** (returns → ~−11, learns nothing); near no-op for Adam. Wrong tool. |
| `state_indep_std` (`_stdglobal`) | Single global σ vector (PPO-style) instead of a per-state σ head. | on/off | **Catastrophic.** Adam → 2–3/3 collapse, negative returns; SGD → learns nothing. Per-state std matters for PCPG. |
| `act_fn` | Hidden nonlinearity. | tanh, relu | tanh is the working default. |
| `width`/`depth` | Net capacity (bench [64,64] vs SOTA [256,256]). | 64/256, depth 2 | Capacity-match removes the confound vs PPO/TRPO; not the stability driver. |
| `num_envs` | Parallel envs = batch. | 256 (bench), 1024 (SOTA) | Scale knob, not stability. |
| `total_steps` | Training length. | 1M (bench), 5M (SOTA) | **Matters: all collapses were late (~780k/1M)** → 5M is a real stress test. |
| `normalize_advantages`/`normalize_rewards` | Standardize advantages / running-normalize rewards. | on | On throughout; hygiene, not varied. |

## Trust-region family scoreboard

Four ways to impose a "trust region" were tried. Only one works, and only with SGD:

| Strategy (flag) | Mechanism | Adam mt20 | SGD mt20 | Verdict |
|---|---|---|---|---|
| `natural_target` | drop `1/σ²`, natural-grad target | 425 ± 491 (1/3) | **781 ± 47 (0/3)** | **Winner — SGD only** |
| `target_clip` (2/5) | cap target offset (output space) | 618–691 (1/3) | — | peaks↑, collapse stays |
| `max_grad_norm` (1) | clip gradient norm | 649 ± 644 (1/3) | −11 (dead) | breaks SGD |
| `state_indep_std` | global σ | −599 (2/3) | ~0 (dead) | catastrophic |

**Lesson:** *where* the trust region lives matters. A metric-space one (natural
target, KL-shaped) works; a gradient-norm one destroys SGD; a parameterization
change (global std) destroys everything.

## Diagnostics (used to explain the collapses)

| Diagnostic | Meaning | What it flagged |
|---|---|---|
| `policy_kl_max/mean` | exact `D_KL(π_old‖π_new)` per update — the trust-region quantity | Adam spikes ~0.30 (shared shocks); SGD stays ~0.04 |
| `value_explained_var` | critic fit `1 − Var(returns−value)/Var(returns)` | **SGD collapse = diverges to −3** (broken critic) |
| `pretanh_sat_frac` | fraction of actions with `|z|>2` (tanh saturation) | **Adam collapse = rises** (overconfident, saturating policy) |
| `policy_drift_max` | `|μ_post−μ_pre|/σ` per update | tracks the KL shocks |
| `mu_target_mag` | raw target-offset magnitude (pre-clip) | natural vs Euclidean regime |
| `log_std_mean`, `frac_std_at_min/max` | where σ sits vs its clamps | entropy-collapse tracking |
| `policy_grad_norm_max/mean` | policy-grad norm, pre-clip | *not* exploding (~0.2 steady; the 0.41 "max" is just the init step) |

## Two failure modes, opposite fixes

The single most important thing the knobs taught us:

- **Adam collapses policy-side:** tanh-saturation + entropy drift → fragile →
  a shared ~0.30 KL shock (all seeds get it ~780k) knocks the drifting run over.
  It doesn't recover. Fix lever = `log_std_min` / `target_clip`.
- **SGD collapses critic-side:** `value_explained_var` diverges to strongly
  negative and stays there → slow monotonic decay, no KL shock, low saturation.
  Fix lever = `value_learning_rate` / target normalization.
- **`natural_target` + SGD** sidesteps both by keeping KL naturally small — the
  current best recipe (`781±47`, 0 collapse) and the promotion candidate.

## Open cells (reasoned about, not yet run)

Two knobs in the table are motivated by the failure analysis but never swept.
The `scripts/run_pcpg_knob_fill.py` sweep fills exactly these:

1. **`log_std_min`** on the collapsing **Adam nat mt20** config (`{−1.5, −1.0}`
   vs the −2 baseline) — does raising the σ floor stop the saturation collapse?
2. **`value_learning_rate`** on the diverging **SGD nat mt80** config
   (`{1e-4, 1e-3}` vs the 3e-4 baseline) — is the critic divergence controllable?
