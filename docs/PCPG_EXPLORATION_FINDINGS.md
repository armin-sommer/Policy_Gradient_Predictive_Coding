# Gaussian PCPG on HalfCheetah — Exploration Findings

**Status: exploratory / pre-registration.** These are results from an informal
tuning exploration, not the formal, preregistered protocol in
[PCPG_TUNING_METHODOLOGY.md](PCPG_TUNING_METHODOLOGY.md). No collapse definition,
gates, or selection rules were fixed in advance; seed counts are small (n=1–3);
and some comparisons are confounded (noted inline). Treat everything here as
**hypotheses generated from evidence**, not established claims.

- Date: 2026-07-18
- Branch: `feature/mujoco-halfcheetah-pcpg`
- Setup: `pc_actor_critic`, bench tier ([64,64], 256 envs, **1M** steps), Brax/MJX
  HalfCheetah. Config copied from the Hopper PCPG config (`target_scale=1.0`,
  `learning_rate=3e-4`, `max_t1=20`, `normalize_advantages/rewards=true`).
- Diagnostics used (already logged by the code): `eval/mean_score`,
  `diag/policy_drift_max` (max change in policy mean per update, in σ),
  `diag/mu_target_mag_max`, `diag/mu_abs_mean`, `diag/frac_std_at_min`,
  `diag/value_explained_var` (critic quality), `diag/pretanh_sat_frac`.

---

## 1. Baseline (default config: `target_scale=1.0`, Adam, ReLU)

**PCPG can learn HalfCheetah, but the default regime is not training-stable —
performance peaks and then frequently collapses.**

- The n=2 learning curve climbs from random (~−14) to a mean peak of ~**1,100**
  (single seed ~1,900) by ~0.25M steps, then crashes, oscillates between positive
  and **negative**, and ends near random (~30).
- Final-step diagnostics (seed 1): `policy_drift_max ≈ 7.4σ`, `mu_abs_mean ≈ 2.68`
  (→ tanh(2.68)≈0.99, means slammed into action-bound saturation),
  `mu_target_mag_max ≈ 20`, `value_explained_var` collapsing toward **−0.9** on
  some seeds.

**Candidate failure mechanisms (heterogeneous across seeds; plausible, not
established):**
- **NOT σ-collapse** (measured) — `frac_std_at_min ≈ 0` throughout; exploration is
  fine.
- **Target explosion / μ-runaway** (measured) — drift spikes to 7–10σ,
  `mu_abs_mean` rises to ~2.7 (means driven into tanh saturation),
  `mu_target_mag_max ≈ 20`.
- **Actor–critic death spiral** (candidate — *not* established): the critic
  collapses (`value_explained_var` → negative), plausibly producing noisy
  advantages that drive bad policy updates that whipsaw returns and further wreck
  the critic. Negative EV shows *correlation*, not causal direction. To support it,
  check the temporal order per seed — does EV decline *before* the drift spikes and
  the return collapse? (step of first major EV decline vs first drift spike vs
  return collapse).

---

## 2. `target_scale` is the mean-target knob (1.0 → 0.3)

**Same-seed A/B (seed 1, only `target_scale` changed) — a clean controlled result:**

| final-step diag | ts=1.0 | ts=0.3 |
|---|---|---|
| `policy_drift_max` | **7.40** | **0.31** |
| `mu_abs_mean` | 2.68 | 1.73 |
| `value_explained_var` (end) | 0.25 *(other seed → −0.9)* | 0.27 (stable) |
| eval return (end) | ~30 (≈random) | **642** |

Lowering `target_scale` linearly shrinks the mean-target and **removed the
*catastrophic* collapse** — every seed now clearly learns.

**But it did NOT produce stable training.** Over n=3 at ts=0.3:

| metric (n=3) | value |
|---|---|
| best (mean ± std) | 889 ± 365 |
| final (mean ± std) | **321 ± 448** |

`final ≪ best` and `final_std > final_mean` → the runs **peak then degrade to
scattered endpoints.** Per-seed: seed 1 ends 642 (ok); seed 2 = critic collapse
(EV −0.67, drift 10σ); seed 3 = critic *healthy* (EV 0.83) yet policy wanders to
**−313**.

**Diagnosis:** the residual failure is **occasional violent updates** (drift
8–10σ). Lowering `target_scale` reduced *typical* drift but was **insufficient to
prevent rare large targets** — `mu_target_mag_max` still reached ~20 at ts=0.3.
(To be precise: `target_scale` *does* linearly scale a given sample's target
offset — it is not that it fails to scale the tail. Rather, a single global scalar
was insufficient because the remaining amplification — large advantages, small σ,
and other quantities that vary across updates — can still produce extreme
offsets.)

**Note:** `gaussian_policy.py` clips the σ-target but leaves `loc_target`
**unclipped**. The branch history had a mean-target ("mu-offset") clip that "cuts
the destructive tail" (commit `c9dd80f`), **removed** in the final commit
(`8ac34fd`). An explicit clip was deliberately *not* pursued here — the goal is an
**implicit** trust region from the PC dynamics (see §3), not a bolted-on clamp.

---

## 3. The implicit trust region: SGD + tanh (Innocenti natural regime)

Innocenti et al. (arXiv:2305.18188) interpret predictive-coding learning as
interpolating between the backprop gradient and an inference-generated
trust-region direction, for **plain SGD on the equilibrated energy**. The default
configs use **Adam**. *Hypothesis* (motivated by that paper, **not** demonstrated
here — see §4): Adam's adaptive preconditioning interferes with the update
geometry that PC inference induces. This does not imply that any SGD-based PC
implementation automatically has a bounded policy update, least of all in a
nonlinear actor–critic RL setting.

### Bug found (blocking the test)
`--overrides train.learning_rate=1e-3` was parsed by PyYAML as the **string**
`"1e-3"` (its float resolver requires a decimal point) → `optax.sgd` did
`"1e-3" * grad` → `TypeError: Only integer scalar arrays can be converted to a
scalar index`. **Fix: always write LRs with a decimal point** (`1.0e-3`) or as
decimals (`0.001`). Latent footgun in `run_train.py`'s override parsing.

### LR screen (SGD+tanh, ts=1.0, 300k steps)

| LR | `drift_max` | best | verdict |
|---|---|---|---|
| 0.01 | **0.12** | −6 | trust region holds, but **too slow to learn** |
| 0.03 | **0.47** | **605** | ★ bounded drift **and** learns |
| 0.1 | **291** | (crashed) | drift explodes → the trust region **breaks at too-high LR** |

→ **Consistent with an LR-dependent, trust-region-*like* stabilization effect.**
At LR=0.03, measured policy drift stays substantially smaller while the agent
learns (reaching 605 at 300k where Adam learned nothing, −8); at LR=0.1, stability
is lost. The target stays large (uncapped, ~25–43) while measured drift stays ~0.5
— consistent with the realized step being limited by the dynamics rather than by
the target, though we have **not** verified that the Innocenti mechanism is what
produces this.

### Full run (SGD+tanh, LR=0.03, 1M, n=3)

| seed | `drift_max` | best | final | outcome |
|---|---|---|---|---|
| 1 | 2.42 | 605 | 267 | degrades (stays +) |
| 2 | 2.76 | 410 | **−285** | collapses |
| 3 | **0.55** | 953 | **855** | **HOLDS** ★ |

**Partial win.** Drift is down ~3× vs Adam (2.4–2.8σ vs 7–10σ), and seed 3 is the
cleanest result to date — climbs to 953 and *holds* at 855 with tight drift (0.55).
**But not solved:** seeds 1–2 still degrade (seed 2 collapses). In these three
runs, **lower maximum drift was associated with the stable outcome** — suggestive,
not predictive at n=3. (`drift_max` can be dominated by a single transient; better
measures — drift 95th percentile, mean of the final 10% of updates, count of
updates above 1–2σ — would separate one spike from persistent instability, and a
correlation over more seeds is needed before calling drift predictive.)

---

## 4. Mechanistic notes (with a confound flagged)

- **Adam (hypothesis, not demonstrated):** one hypothesis is that Adam's adaptive
  preconditioning interferes with the update geometry PC inference induces. But
  (a) Adam does **not** generally normalize every update to exactly LR magnitude;
  (b) we never measured Adam's effective update norms, moment estimates, or their
  relation to the PC energy; and (c) optimizer and activation were changed
  together. Attributing the instability specifically to Adam needs the
  effective-update-norm measurement **and** the full optimizer×activation ablation.
- **tanh vs ReLU:** theory (and the code's framing) says the natural regime needs a
  smooth, bounded activation — tanh's smooth derivative and bounded range keep the
  energy well-behaved; ReLU's kinked 0/1 derivative, dead units, and unbounded
  outputs degrade it. **CONFOUND:** every comparison changed *both* optimizer and
  activation together (`adam+relu` vs `sgd+tanh`). We tested only **2 of the 4
  corners**. We have shown *"the sgd+tanh package beats the adam+relu package"* —
  **not** that tanh (or sgd) individually matters. Needs the 2×2 ablation
  (`sgd+relu`, `adam+tanh`).
- **Robustness to a weak critic:** in the natural regime the critic is often *poor*
  (`value_explained_var` negative), yet the policy still learns — the bounded steps
  make it **robust to noisy advantages**. Adam took big steps on that same noise and
  self-destructed.

---

## 5. Compute cost (both efficiency axes)

Measured wall-clock (HalfCheetah, from run logs):

| run | env steps | wall-clock/run | effective steps/s |
|---|---|---|---|
| PPO sota | 30M | ~35 min | ~14,300 |
| TRPO sota | 30M | ~95 min | ~5,300 |
| **PCPG bench** | 1M | ~7 min | **~2,400** |

**PCPG is the slowest *per step*** (~6× slower than PPO), because of the `max_t1=20`
inference inner loop that backprop lacks. It only *finishes* fast because bench is
1M steps; at PPO's 30M budget it would take ~3.5 h — longer than both baselines.
Any PCPG-vs-baseline claim must report **both** sample efficiency (env-steps) and
compute efficiency (wall-clock/FLOPs).

*Caveat:* PPO/TRPO used 1024 envs, PCPG bench 256, and evaluation overhead,
compilation, and device conditions were not matched. Treat the steps/s figures as
**approximate / order-of-magnitude**, not a controlled per-step benchmark.

---

## 6. Seed variance / fragility

The three §3 full runs differ **only in the seed** (same config, optimizer,
activation, LR). Outcomes range from **855 (stable)** to **−285 (collapse)**. The
seed controls network init, env init, action sampling, and minibatch order — so
these are *the same algorithm with different luck*. **This high seed-variance is
itself the core fragility finding**, and it makes n=2–3 untrustworthy.

---

## 7. Open questions / next steps

1. **`max_t1` (equilibration)** — untested. Tests whether *incomplete inference*
   contributes to the residual drift spikes. It **will** cost wall-clock (~linear
   in `max_t1`), **may** alter learning dynamics / sample efficiency, and is **not**
   guaranteed to tighten drift — evaluate it with an inference-convergence
   diagnostic and the compute cost. Use seed 2 (the collapser) only as a
   *diagnostic*; do **not** select the final hyperparameter on the known-bad seed
   (overfits to it) — confirm on a fixed/fresh seed set afterward.
2. **Resolve the optimizer×activation confound** — run the 2×2 so `sgd+tanh` is
   justified, not assumed.
3. **Critic (hypothesis)** — if `max_t1` doesn't help, the residual *may* be
   critic-driven. The value net does inherit the same optimizer and activation
   (confirmed in code: `make_mlp(act_fn=Config.act_fn)`,
   `make_optim(value_learning_rate)`), but keeps `value_learning_rate=3e-4` — so in
   the sgd+tanh runs the *policy* LR was raised to 0.03 while the *critic* stayed at
   3e-4. Raising `value_learning_rate` is **one** candidate; but negative EV can
   also come from too-high value LR, nonstationary targets, reward normalization,
   architecture, or a bug. Verify the cause (underfitting vs unstable targets)
   before acting.
4. **Then formalize** — freeze the regime, fill in the Appendix A preregistration
   (collapse definition, thresholds, selection rule), and run Stage 2→4 properly
   with adequate seeds.

## Bottom line — what this can honestly conclude

The Gaussian PCPG implementation **can** learn HalfCheetah, but learning is
currently **fragile and highly seed-dependent**. Lower `target_scale` and the
SGD+tanh regime substantially reduce policy drift relative to the original
Adam+ReLU configuration, and one SGD+tanh seed maintained meaningful performance
(~855) through 1M steps. These results **motivate** further testing of inference
equilibration (`max_t1`), the optimizer×activation ablation, and critic stability
— but they do **not** yet establish an implicit trust-region *mechanism* or a
robust final configuration.

Does the PC approach work? **Yes**, in the limited but real sense that it produces
genuine learning on HalfCheetah. **Not yet** in the stronger sense of being
reliable, stable, or competitive.

---

## 8. Known bugs / gotchas
- **LR override string bug** (§3) — use `1.0e-3`/`0.001`, never `1e-3`.
- **`--skip-complete`** skips any log containing `TRAINING END`, so a short smoke
  run "completes" a seed and blocks the real run — delete the smoke log first.
- `Failed to import warp` on the pod is **harmless** (Brax optional backend; we use
  `mjx`).
