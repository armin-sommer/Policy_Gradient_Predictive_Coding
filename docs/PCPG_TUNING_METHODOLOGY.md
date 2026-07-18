# PCPG Hyperparameter & Config Testing Methodology

Scope: a defensible protocol for testing PCPG (`pc_actor_critic`, `pc_reinforce`)
configurations on Brax/MJX continuous control, starting with HalfCheetah, and for
comparing PCPG against the backprop baselines (PPO, TRPO). This document is the
plan of record; it supersedes earlier informal notes.

Status of the evidence base: **we currently have no PCPG runs to calibrate
against.** Every threshold, fidelity level, and sensitivity ranking below is a
*hypothesis to be measured*, not an established fact. The protocol is therefore
front-loaded with a calibration stage whose job is to produce that evidence
before any aggressive pruning or gating is trusted.

---

## 1. Principles (invariants the protocol must not violate)

1. **Verify before pruning.** Early-stopping and diagnostic gates are only valid
   once we have shown, on real PCPG runs, that (a) the gate predicts eventual
   failure with a low false-positive rate, and (b) performance at a reduced
   budget is rank-correlated with performance at the final budget. Until then,
   pruning is unjustified.
2. **Hard gates catch only unequivocal numerical failure.** Softer diagnostic
   signals are ranking/diagnosis *features*, not auto-reject rules.
3. **Separate engineering tuning from research ablations.** Hyperparameters used
   to obtain the benchmark result are tuned in one experiment; algorithmic
   variants (optimizer, activation, std parameterization, natural-gradient/
   trust-region target rules) are tested in a separate track. Mixing them means
   the method mutates during tuning and the final comparison becomes
   uninterpretable.
4. **Tuning seeds are disjoint from evaluation seeds.** RL hyperparameter
   landscapes can be seed-dependent; selecting on the test seeds inflates results
   (Eimer et al. 2023).
5. **Failures stay in the performance distribution.** A genuine algorithmic
   divergence is part of PCPG's behavior and is reported as such. A run is
   excluded only when it is a *proven* infrastructure failure (crash unrelated to
   the algorithm, preemption, etc.), documented case by case.
6. **Report both efficiency axes.** PC inference iterations (`max_t1`,
   `pc_steps_per_update`) change per-update cost, and the backprop baselines have
   no inference inner loop at all. Neither axis alone is sufficient: environment
   steps measure *sample* efficiency, while wall-clock/compute measures
   *computational* efficiency. Every PCPG-vs-baseline comparison is shown on both
   (§7).

---

## 2. Available signals (from the current code)

Both `pc_actor_critic.py` and `pc_reinforce.py` already log, per update:

- `eval/mean_score`, `eval/std_score`, `eval/mean_episode_length` (deterministic-
  action eval via `pc_eval.py`).
- `training/*`: pc loss(es), mean reward, mean |advantage|.
- `diag/*` (continuous policies): `log_std_mean`, `log_std_min`,
  `frac_std_at_min`, `frac_std_at_max`, `mu_abs_mean`, `policy_drift_mean`,
  `policy_drift_max`, `mu_target_mag_mean`, `mu_target_mag_max`,
  `pretanh_sat_frac`, and (actor-critic) `diag/value_explained_var`.

These are the raw material for both gating and ranking. Which of them predict
collapse — and at what thresholds — is an output of Stage 0, not an input.

### 2.1 Hard gates

A run is killed immediately if any of the following occur. They fall into two
tiers with different justification burdens.

**Tier A — intrinsic numerical invalidity (no empirical calibration needed).**
Unambiguous by definition:

- NaN / Inf in loss, parameters, gradients, or emitted actions.
- Non-finite or invalid distribution parameters (e.g. non-positive scale).
- Invalid actions, under a **declared meaning** of "invalid" (non-finite; outside
  the action space before clipping; outside after clipping; or rejected by the
  env). The tanh-squashed policy can make tiny floating-point excursions past the
  nominal range, so the boundary must be explicit. Action-clipping frequency is
  logged — silent clipping can conceal instability while letting a run continue.

**Tier B — implementation-specific emergency stops (parameters must be declared
and mechanically validated in Stage 0A before use).** These are almost certainly
pathological in this implementation, but each embeds a threshold or window that
must be fixed *a priori*, not chosen after seeing results:

- PC target magnitude beyond a **declared emergency bound** (derived from the
  representable numeric range with headroom, or a provisional emergency
  threshold — not the word "plausible").
- Sustained **complete** variance-bound saturation (`frac_std_at_min == 1.0` or
  `frac_std_at_max == 1.0`) over a **declared window length**. Note the bounds are
  implementation clamps, so this is an emergency stop, not an intrinsic invalid.
- Dead-update condition: drift below a **declared tolerance** with flat loss over
  a **declared duration** (small updates can be legitimate near convergence).

A Tier-B stop terminates the run for safety but does **not** by itself determine
the run's final classification. The cause — genuine algorithmic divergence, an
over-conservative bound, instrumentation error, or an implementation bug — is
adjudicated under the pre-declared collapse definition (§6, Appendix A) and
retained in the experiment record.

### 2.2 Soft features (ranking / diagnosis; calibrated in Stage 0)

Partial std saturation, transient negative `value_explained_var`, moderate
`policy_drift`, `pretanh_sat_frac`, and target-magnitude *trends* are informative
but not proof of failure (e.g. negative explained variance can be transient;
partial saturation can be an adaptive response). They enter the ranking metric
and the collapse-prediction analysis; they do not trigger auto-rejection until
Stage 0 has estimated their false-positive/false-negative rates.

---

## 3. Search space

The priorities below are **hypotheses** motivated by PCPG's structure and by the
branch's collapse history; the sensitivity itself is measured during
calibration/screening, not assumed.

### 3.1 Engineering hyperparameters (tuned for the benchmark result)

| Knob | Hypothesized role | Initial range |
|---|---|---|
| `target_scale` | Scales advantage → PC target magnitude; the branch's collapse-and-revert history centers on target magnitude. | log-uniform, ~0.1–1.0 |
| `learning_rate` | Weight-update size; dominant HP in conventional on-policy RL. | log-uniform, ~1e-4–1e-3 |
| `max_t1` | PC-specific: inference steps to equilibrate activities; too few biases the target. **Also a compute knob** (§7); whether it is tuned or fixed is an open decision (§6). | small discrete set, e.g. {10, 20, 40} |
| one normalization decision | Only if there is a concrete hypothesis after §5 distribution checks. | {true, false} |

### 3.2 Frozen at conventional defaults (provisional baselines)

`gamma=0.99`, `gae_lambda` (0.95 AC / 0.97 TRPO-style), `depth=2`, `width` by
tier, `rollout_length` by algorithm (32 GAE / 256 MC). These are frozen to reduce
dimensionality using established on-policy defaults **as provisional baselines,
not because they are known-optimal for a predictive-coding algorithm** — PCPG's
target construction and inference dynamics may shift the sensitivity landscape
(the intended reading of Andrychowicz et al. 2020, whose thesis is that many
implementation choices matter). Any of these can be revisited if Stage 0/2
evidence implicates it.

### 3.3 Research ablations (separate track, never inside the benchmark sweep)

`optimizer` (Adam vs SGD — Innocenti et al. trust-region property for plain GD),
`act_fn` (ReLU vs tanh), `exp_std` (exp vs softplus), and any natural-gradient /
trust-region target-rule variants. Each is a scientific question answered with
its own controlled experiment against a fixed engineering configuration.

Multiple-comparison discipline: **all** pre-specified ablations are reported, not
only the ones that won; primary and exploratory analyses are labelled as such; no
variant is promoted into the main method solely because it won on a single
environment; and the more variants tried, the more cautiously a lone "win" is
interpreted.

---

## 4. Staged protocol

Each stage names its verification criterion. A stage is not "done" because it ran;
it is done when its criterion is met.

Because no PCPG data exists, Stage 0 produces the initial evidence required to
justify the funnel. It is split into two cheap sub-stages with different jobs;
neither requires dozens of full-length runs.

### Stage 0A — Mechanical failure tests (validate the hard gates)

A handful of intentionally broken configurations, **short runs only** (~10k–50k
steps). Purpose: confirm the system actually detects and halts on the §2.1
failures — both Tier A (NaN/Inf, invalid std, invalid actions) and Tier B (target
explosion past the declared bound, complete saturation over the declared window,
the dead-update condition), plus logging/termination faults. Tier-B thresholds and
windows are declared *before* Stage 0A (Appendix A); this stage validates the
**implementation** of the gates under those declared settings — it does not choose
them. A failed test may justify revising the protocol and rerunning Stage 0A, but
the values are never tuned against observed experimental behavior. Stage 0A does
**not** validate prediction of naturally-occurring collapse (an engineered NaN
looks nothing like a config that learns for 400k steps and then destabilizes).

Verify: each hard gate fires on its corresponding broken config and the run halts
cleanly; no gate fires on a **conservative control configuration that remains
numerically valid over the short test horizon** (a non-failing short run is not
necessarily healthy in the learning sense — it only shows no false trigger).

### Stage 0B — Pilot calibration (soft signals, rungs, variance)

A modest, diverse sample of **plausible** configurations. This calibration is
**PCPG-only** (`pc_actor_critic` and `pc_reinforce`, which may have different
signatures and so warrant **separate** pilot budgets rather than sharing one
total); the backprop baselines need no PC-diagnostic calibration and enter only at
Stage 4. Run the pilot **adaptively rather than at a fixed size**, on plausible configs
drawn **per materially different PCPG algorithm** (here exactly two —
`pc_actor_critic` and `pc_reinforce`; architecture-only variants such as the width
tiers do not by themselves trigger a separate pilot). Launch in **waves of ~4–6
configs**, two seeds each, initially 300k–500k steps, and stop adding waves when
either the pre-declared **pilot conclusiveness criterion (Appendix A6)** is met or
a pre-declared **maximum pilot budget** is reached — expect on the order of 8–12
configs per algorithm, but let the stopping rule, not a fixed number, decide.
Extend only a promising subset to 1M. Sequential sampling is more efficient than a
fixed *N* for the "is the early→final signal there yet?" question this pilot
exists to answer.

**Stage 0B is for decision calibration, not reliable statistical estimation.**
With a pilot this size every number below is provisional and is reported with its
uncertainty: Spearman correlation on a small extended subset will be unstable, and
soft-diagnostic false-positive rates are essentially uninterpretable if only one
or two runs collapse. **No formal soft gate is authorized from Stage 0B unless the
pre-declared pilot conclusiveness criterion (Appendix A6) is met** — making the
go/no-go for pruning itself preregistered, rather than a judgment call after
seeing the pilot; otherwise the fallback below applies.

Do **not** force a balanced healthy/collapsed dataset — hand-balancing makes the
calibration distribution artificial and its accuracy numbers misleading. Sample
the realistic search region (ranges for `target_scale`/`learning_rate` informed
by the branch history). The goal is to understand behavior where tuning will
actually operate. Log all `diag/*` every update and `eval/mean_score` on a dense
checkpoint grid (≥ 100k / 250k / 500k, plus 1M on the extended subset).

Estimate from these runs:
1. Whether performance at a reduced budget predicts later performance (Spearman
   rank correlation across 100k/250k/500k vs the extended 1M subset). This sets
   the **minimum trustworthy fidelity** and thus the ASHA rung levels.
2. Which soft diagnostics precede poor outcomes, with what lead time and — on
   *plausible* configs — what false-positive rate. Calibration goals are low FPR
   (don't kill promising runs), useful lead time, and honest uncertainty on a
   small dataset, **not** balanced-classification accuracy.
3. Approximate seed variance, to inform later seed counts (§5).
4. The empirical **distributions** (not just means/maxima) of normalized
   advantages and resulting PC targets, to test rather than assume whether
   `target_scale` transfers from Hopper (§5, §8).

Graceful degradation: if the plausible region yields too few natural collapses to
calibrate a soft predictor, or early/final correlation is weak at every reduced
budget, do **not** manufacture positives. Fall back to hard-failure-only pruning
with longer minimum budgets, and treat soft diagnostics as diagnosis rather than
gates. An inconclusive pilot is an acceptable, informative outcome.

Verify: either (a) a soft-signal gate with documented low FPR and lead time plus
an evidence-based rung, or (b) an explicit decision to run hard-gates-only with
longer rungs.

### Stage 1 — Candidate preflight (integration/numerical, optional)

An integration and numerical preflight, not an experimental stage: one seed,
50–100k steps, applying the §2.1 hard gates validated in Stage 0A, to catch shape,
logging, and integration faults before committing a config to distributed,
costly Stage-2 machines. Do **not** rank viable configurations here. It **may be
skipped** for configurations generated from a fully validated template (Stage 2
itself terminates on hard failures).

Verify: candidate runs end-to-end with no §2.1 violation.

### Stage 2 — Low-fidelity tuning

Two fixed tuning seeds per configuration, ~250k–500k steps (or the Stage-0
minimum fidelity, whichever is larger). Search the §3.1 engineering knobs only.
Apply ASHA **only** with the rung levels and reduction factor justified by Stage 0;
otherwise use fixed-budget random search over the reduced space.

Stage 2 is a **feasibility filter plus a ranking**, using a pre-declared
lexicographic rule (fixed before any Stage-2 run, to remove the post-hoc
flexibility of a weighted "composite"):

1. exclude hard failures (§2.1);
2. require **both** tuning seeds to clear the pre-declared **viability** threshold.
   Beating a random-action baseline is only a feasibility test — in continuous
   control many dysfunctional policies clear random return — so viability is *not*
   evidence of competitive performance; a stronger **promotion** threshold
   (relative to a validated simple baseline or a task-specific reference) is
   declared in Appendix A;
3. rank survivors by **normalized AUC** (defined in Appendix A);
4. use the pre-declared soft-diagnostic stability score **only as a tie-breaker**.

Stage 2 ranks on the *same* normalized-AUC quantity Stage 3 optimizes, so a config
is never promoted for one property and selected under another.

Verify: ≥1 configuration clears the viability threshold stably across both tuning
seeds.

### Stage 3 — Medium-budget confirmation

Top 3–5 configurations, three tuning seeds, 1M steps. Select on the **exact
primary tuning objective** declared in Appendix A — the same normalized-AUC
quantity Stage 2 ranked on, so promotion and final selection agree. Compute enters
as a **hard budget constraint or an explicit second objective**, not an arbitrary
penalty: unless a penalty's functional form and coefficient are justified in
advance, prefer a Pareto rule — among non-collapsing configs on the return–compute
frontier, choose the lowest-compute config within a pre-declared practical-return
margin of the best. Do not tune per environment unless the scientific claim
explicitly permits environment-specific tuning (§6).

"Stable" is defined by a **pre-declared decision rule**, not a subjective judgment
— three seeds can reveal obvious seed-dependence but cannot statistically rule it
out. The default policy is deliberately **risk-averse**, a value choice rather than
a statistical necessity: the winner satisfies the preferred stability rule iff it
(a) ranks first on ≥2 of 3 seeds, (b) is not exceeded by any runner-up by more than
a declared practical margin on aggregate AUC, and (c) shows no algorithmic collapse
on any of the three tuning seeds — so a single collapse prevents a config from
*satisfying the preferred rule* even if its expected performance is higher.
Failing the preferred rule is **not** automatic exclusion: selection may still
proceed by aggregate score, with the seed-dependence recorded as a known risk
carried into Stage 4.

Verify: the winner satisfies the decision rule, or the failure to do so is
documented.

### Stage 4 — Locked final evaluation

Freeze the configuration. Evaluate on **disjoint** final seeds. Compare PCPG and
baselines under identical conditions: same env-step budget, same eval checkpoints
and eval-policy convention, paired environment seeds where appropriate, equal
*tuning* budget across algorithms, and compute/wall-clock reporting. Include all
runs, including genuine algorithmic collapses.

Reporting (§5).

---

## 5. Statistics and reporting

- **Per environment:** always **show every seed outcome** and the learning
  curves. Report a pre-declared interval estimator appropriate to the sample size;
  with only 5–10 seeds a nonparametric bootstrap interval is itself unstable and
  is interpreted **descriptively**, not as a precise coverage guarantee — it is
  not automatically superior to raw seeds plus a descriptive range. Do **not**
  compute IQM over a handful of single-environment seeds.
- **Across the task suite:** aggregate **normalized** scores with IQM and
  stratified bootstrap confidence intervals (much better motivated at suite level),
  and show performance profiles (Agarwal et al. 2021). This is the setting rliable
  was designed for. The normalization convention (§6) must be fixed before Stage 4,
  since it can change aggregate rankings.
- **Pairwise (PCPG vs a baseline):** report probability of improvement across
  tasks and seeds where the design supports it.
- **Seed count is derived, not declared.** Set it from a pilot variance estimate
  or a sequential-uncertainty criterion for the specific effect size and claim.
  Provisional scaffold for planning only: 3 tuning seeds; 5 final seeds during
  development; 10 for the main comparison if variance remains high — each subject
  to the pilot estimate.
- **Advantage/target distributions:** report the full distributions used in the
  §4 Stage-0 transfer check, not point summaries.

---

## 6. Open decisions that shape the design

These are judgment calls not derivable from the repo. Each must be fixed before
the stage noted; they are not settled by this document.

- **Claim scope — per-environment or aggregate across a suite?** Drives whether
  environment-specific tuning is admissible (Stage 3), the seed-count design, and
  whether the primary report is per-env intervals or suite-level IQM/performance
  profiles. *Before Stage 3.*
- **Definition of collapse.** The operational criterion that separates a genuine
  algorithmic divergence from ordinary low return, used consistently by the gates
  (§2), the selection penalty (Stage 3), and reporting (§5). *Before Stage 0B.*
- **Tuning objective.** Final return, area under the eval curve, stability-
  adjusted return, or compute-adjusted return — the single pre-declared metric
  Stage 3 selects on. *Before Stage 2.*
- **Hyperparameter scope.** Global across environments or tuned per environment
  (coupled to claim scope). *Before Stage 3.*
- **`max_t1` status.** Part of the algorithm definition (fixed) or a tunable
  compute budget — determines whether it sits in §3.1 (tuned) or §3.2 (frozen)
  and how it is read on the compute axis (§7). *Before Stage 2.*
- **Baseline provenance and acceptance.** For PPO/TRPO: own implementation,
  adapted repo, or an established reference; whether baseline bugs/underperformance
  are fixed first; and whether each receives method-appropriate tuning. A weak
  local TRPO is not a credible baseline merely because it is labelled TRPO. The
  acceptance criterion is **formal** (Appendix A): environment/implementation
  match, mean/median return within a declared literature/reference range, plausible
  learning-curve shape, no known implementation defect open, and evaluation
  convention aligned with PCPG. *Before Stage 4.*
- **Tuning-budget parity — equal by what unit?** "Equal tuning budget" is
  ambiguous (configs / env-steps / accelerator-hours / seed-runs / updates). Pre-
  declare the unit; the right choice depends on the claim — e.g. equal
  accelerator-hours + identical tuning-seed protocol (report configs and env-steps
  explored), or equal total env-interaction budget with compute reported
  separately for a sample-efficiency study. *Before Stage 4.*
- **Task normalization convention.** How per-task scores are normalized for
  suite-level IQM / performance profiles (random-to-reference, random-to-expert,
  task-baseline, rank, or raw where scales are comparable). Alters aggregate
  rankings. *Before Stage 4.*
- **Seed pairing across methods.** Whether environment seeds are paired between
  PCPG and baselines (enables paired tests / probability-of-improvement). *Before
  Stage 4.*
- **Compute measurement unit.** The primary axis for §7 (wall-clock on identical
  hardware/software vs a measured-FLOPs/accelerator-hour proxy). *Before Stage 4.*

---

## 7. Compute accounting

`max_t1` and `pc_steps_per_update` multiply inner compute per update, so two
configurations at equal environment steps are **not** equal-cost. Neither axis is
sufficient alone: environment steps measure sample efficiency, wall-clock/compute
measures computational efficiency, and both are scientifically meaningful.

Wall-clock and operation counts are **not** interchangeable (wall-clock depends on
hardware, compilation, vectorization, batch size, implementation quality, and
logging/eval overhead; operation counts are harder to define fairly across PC and
backprop). Every run therefore records, at minimum:

- wall-clock under **identical hardware/software conditions**;
- environment steps;
- number of updates;
- `max_t1` / PC inference steps (and `pc_steps_per_update`);
- peak memory, where relevant.

Report two families and keep them distinct:

- **Algorithmic compute proxies** (implementation-independent): number of updates,
  PC inner iterations (≈ `max_t1` activity-relaxation steps per PC update ×
  `pc_steps_per_update` PC updates per minibatch), and — with credible
  instrumentation — measured FLOPs.
- **Realized system cost**: wall-clock and accelerator-hours on identical
  hardware/software.

Wall-clock combines algorithmic efficiency with **implementation maturity**, so it
can penalize a less-optimized code path rather than the underlying method — a real
risk here, since the PCPG and PPO/TRPO paths may differ in JAX compilation and
vectorization maturity. Report both families; neither alone stands in for
"efficiency." The primary compute unit is a pre-declared choice (§6).

---

## 8. Threats to validity / assumptions to verify (not assert)

- **Hopper → HalfCheetah transfer.** Reward scale and termination behavior must
  be read from the actual Brax/MJX env definitions and from logs, not assumed.
  Episode *returns* for the two tasks are of similar order (baseline logs), so
  earlier per-step "≈10×" phrasings are not reliable. `normalize_advantages=true`
  does not by itself make `target_scale` equivalent across environments —
  advantage tails, action dimensionality, policy σ, critic error, and horizon all
  differ; this is settled by the §4 distribution comparison.
- **Early/final rank correlation** is the load-bearing assumption for the whole
  funnel and is measured in Stage 0, not presumed.
- **Diagnostic thresholds** are calibrated, with reported error rates, before use
  as gates.

---

## 9. Reproducibility and selection controls

Good statistical reporting can still be undermined by informal reruns and post-hoc
exclusions. Every stage therefore records:

- the **fixed code commit** used (one commit per stage; no mid-stage changes);
- **immutable config files** (config edits create new files, not in-place edits);
- environment, wrapper, and dependency versions (Brax/MJX/JAX pins per
  `pyproject.toml`), and the **hardware type**;
- all seeds, evaluation frequency, and the **checkpoint-selection rule** (which
  checkpoint's return is reported, fixed in advance);
- **every attempted configuration, including failures** — nothing is silently
  dropped;
- **no silent reruns** of unlucky seeds; a rerun is logged with its reason and the
  original is retained;
- **final evaluation seeds are generated and recorded before Stage 4, are not
  inspected during tuning, and are not replaced or supplemented based on observed
  outcomes** — any additional evaluation batch is reported separately as a new
  analysis;
- a clear separation between **tuning** dashboards/seeds and **evaluation**
  dashboards/seeds (§1.4).

---

## Appendix A — Preregistration manifest (fill before the relevant stage)

The methodology above becomes *executable* once the following are committed to a
versioned manifest (a single file under version control), each **before** the
stage that consumes it. The functional forms are fixed here; bracketed `<...>`
values are the project's to declare, and must not be changed after seeing the data
they gate. This is the small preregistration appendix that turns a defensible plan
into a runnable one.

**A1 — Collapse classification** *(before Stage 0B; rationale in §6).* A run is
classified `collapsed` iff `<operational criterion>` — e.g. eval return falls below
`<frac>` × its own running maximum for `<N>` consecutive checkpoints *after* having
exceeded the viability threshold, or a Tier-B stop is adjudicated as genuine
divergence. Every Tier-B stop and every candidate collapse is adjudicated under
this single definition and recorded (§2.1).

**A2 — Tier-B emergency thresholds/windows** *(before Stage 0A; §2.1).*
- target-magnitude emergency bound: `<value>` (from the representable numeric range
  with headroom);
- complete-saturation window: `frac_std_at_{min,max} == 1.0` for `<K>` consecutive
  updates;
- dead-update: `policy_drift < <ε>` with `|Δloss| < <δ>` sustained over `<M>`
  updates.

**A3 — Normalized AUC and early-termination handling** *(before Stage 2; §4).*
Trapezoidal AUC of eval return over a **common, pre-declared env-step checkpoint
grid** (equally spaced at `<Δ>` steps, linear interpolation for missing points),
normalized by `<return normalizer>` and by the maximum step budget. From the first
adjudicated collapse onward the curve takes the pre-declared **failure score**
`<value>` (a collapsed run is scored, *not* dropped — §1.5). Thresholds:
viability = `<above-random margin>`; promotion = `<reference return>`.

**A4 — Stage-3 return-vs-compute selection** *(before Stage 3; §4, §7).* Primary
objective = the A3 normalized AUC. Compute treatment = `<hard budget cap C>` **or**
Pareto rule with practical-return margin `<m>`. Stability decision-rule margin =
`<margin>` (Stage 3).

**A5 — Baseline acceptance** *(before Stage 4; §6).* Each of PPO/TRPO is accepted
iff: implementation source = `<...>`; mean/median return reaches
`<literature/reference range>` on `<validation env(s)>`; learning-curve shape
plausible; no known defect open; evaluation convention aligned with PCPG.

**A6 — Pilot conclusiveness criterion** *(before Stage 0B; §4 Stage 0B).* The
adaptive pilot stops adding waves, and a soft-diagnostic gate is authorized, iff:
≥ `<K>` adjudicated collapses have been observed **and** the bootstrap CI on
Spearman ρ(early, final) at the candidate rung excludes `<ρ_min>`. Maximum pilot
budget before forced stop = `<budget>` (configs or accelerator-hours). If the
budget is exhausted before the criterion is met, the pilot is declared
inconclusive and the protocol falls back to hard-failure-only pruning with longer
minimum budgets (§4 Stage 0B, graceful degradation) — the rank-correlation and
seed-variance estimates are still retained and used.

Also fix, per §6 (not repeated here): claim scope, hyperparameter scope, `max_t1`
status, tuning-budget unit, task-normalization convention, seed pairing, and the
primary compute measurement unit.

---

## References

- Andrychowicz et al., *What Matters in On-Policy Reinforcement Learning? A
  Large-Scale Empirical Study*, 2020. arXiv:2006.05990.
- Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical
  Precipice* (rliable), NeurIPS 2021. arXiv:2108.13264.
- Eimer et al., *Hyperparameters in Reinforcement Learning and How To Tune Them*,
  ICML 2023.
- Li et al., *A System for Massively Parallel Hyperparameter Tuning* (ASHA), 2018.
  arXiv:1810.05934.
- Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization
  Framework*, 2019. arXiv:1907.10902.
- Innocenti et al., arXiv:2305.18188 (predictive-coding trust-region property for
  plain gradient descent; as cited in `pc_actor_critic.py` / `pc_reinforce.py`).
