# PCPG Benchmark Plan

Status: benchmark planning document. This translates the current research
decisions into a systematic route for comparing PCPG against PPO/TRPO on MuJoCo.

Primary claim:

```text
PCPG can learn HalfCheetah and can be competitive with PPO/TRPO.
```

Secondary claim:

```text
PCPG behavior should be understood on both sample efficiency and compute cost.
```

---

## 1. Algorithms

Fixed baselines:

- `ppo`: considered credible and working.
- `trpo`: considered credible and working.
- `reinforce`: implementation fixed, but not yet a working competitive baseline.

PCPG algorithms:

- `pc_actor_critic`: main PCPG candidate.
- `pc_reinforce`: secondary PCPG candidate.

Important distinction: PPO/TRPO are fixed baselines. PCPG implementations may
still be updated during the mechanism/tuning phase, but once final benchmark
configs are declared, PCPG code and configs must be frozen for final evaluation.

---

## 2. Environments

Benchmark order:

1. HalfCheetah.
2. Hopper.

HalfCheetah is the development environment. Hopper is the first transfer check.
Do not tune a new PCPG method on Hopper until the HalfCheetah protocol is clear;
otherwise the benchmark becomes environment-specific tuning.

---

## 3. Metrics

Primary metric for the development benchmark:

```text
final eval return at the fixed step budget
```

Required secondary metrics:

- best eval return;
- area under the eval learning curve;
- collapse/degradation count;
- wall-clock time;
- environment steps;
- individual seed curves;
- all PCPG diagnostics.

Why final return is primary: current PCPG runs often learn and then degrade. Best
return alone would overstate the method by hiding instability.

Why AUC is still reported: a method that learns quickly but degrades behaves
differently from one that never learns. AUC preserves that information.

---

## 4. Seed Policy

Current testing phase:

```text
3 seeds per algorithm/config
```

This is enough to expose obvious fragility and compare routes cheaply. It is not
enough for a strong statistical claim. Therefore:

- use 3 seeds for development/tuning;
- report every seed;
- do not rerun unlucky seeds;
- do not select a config because it won on one seed;
- if a config is promoted as "final", mark the conclusion as provisional unless
  it later gets more seeds.

Suggested seed split:

```text
tuning/development seeds: 1, 2, 3
final locked seeds:       11, 12, 13
```

If compute allows later:

```text
robustness seeds: 11-20
```

---

## 5. Collapse / Degradation Definition

Because the current behavior includes "learn then collapse", define collapse
before final benchmarking.

Recommended operational definition:

```text
A run is collapsed if:
  1. it first exceeds a viability threshold;
  2. then its eval return falls below 30% of its own previous best;
  3. and stays below that level for at least 3 consecutive eval checkpoints.
```

For HalfCheetah development, use:

```text
viability threshold = 300 eval return
```

Also report "degradation" separately:

```text
degradation ratio = final_return / best_return
```

Interpretation:

```text
ratio near 1.0: held performance
ratio 0.3-0.7: degraded
ratio below 0.3 after learning: collapsed
negative final after positive best: severe collapse
```

This definition is intentionally simple and transparent. It is not a SOTA
standard; it is a project-specific stability label designed for the observed PCPG
failure mode.

---

## 6. Required PCPG Diagnostics

Every PCPG benchmark table should include or link to:

- `diag/policy_drift_max`;
- drift p95 or final-window drift, if parsed from logs;
- `diag/mu_target_mag_max`;
- target-magnitude p95, if parsed from logs;
- `diag/value_explained_var`;
- `training/value_pc_loss`;
- `training/policy_pc_loss`;
- `diag/mu_abs_mean`;
- `diag/pretanh_sat_frac`;
- `diag/log_std_mean`, `diag/log_std_min`;
- `diag/frac_std_at_min`, `diag/frac_std_at_max`;
- wall-clock and steps/sec.

Reason: final return tells whether the algorithm worked; these diagnostics tell
which failure mode occurred.

---

## 7. Recommended Route

### Stage A: Freeze the baseline comparison

Confirm PPO and TRPO have fixed configs, fixed budgets, fixed eval cadence, and
credible HalfCheetah results. These are already considered credible, so this
stage is mostly documentation.

Output:

```text
locked PPO config
locked TRPO config
baseline summary table
```

### Stage B: Mechanism tests for PCPG

Do this before claiming competitiveness.

Required tests:

```text
optimizer x activation:
  adam + relu
  adam + tanh
  sgd  + relu
  sgd  + tanh

target scale:
  0.1
  0.3
  1.0
```

Do not run more `max_t1=40` as a rescue fix; it worsened the known-collapsing
seed and increased drift.

Decision rule:

```text
promote the PCPG config with the best final return among configs
that do not collapse on more than 1 of 3 development seeds
and whose wall-clock is recorded.
```

If no config passes this rule, the correct conclusion is not "benchmark failed";
it is:

```text
PCPG learns transiently but is not stable enough for a competitiveness claim yet.
```

### Stage C: Locked HalfCheetah benchmark

Freeze:

- code commit;
- config files;
- seeds;
- eval cadence;
- step budget;
- checkpoint-selection rule;
- collapse definition.

Run:

```text
ppo
trpo
pc_actor_critic
pc_reinforce
optional: reinforce, clearly marked as weak/noncompetitive if still broken
```

Report:

```text
final return mean +/- std
best return mean +/- std
AUC mean +/- std
collapse count
wall-clock mean
all seed curves
diagnostic summary for PCPG
```

### Stage D: Hopper transfer

Run the same PCPG regime on Hopper with minimal retuning. If retuning is needed,
label it explicitly:

```text
HalfCheetah-tuned PCPG transfer to Hopper
```

or

```text
environment-specific tuned PCPG on Hopper
```

Do not mix these claims.

---

## 8. Why this route is best

This route separates three things that are otherwise easy to confuse:

1. **Can PCPG learn at all?**
   Answered by best return and AUC.

2. **Can PCPG hold performance?**
   Answered by final return, degradation ratio, and collapse count.

3. **Is PCPG competitive with PPO/TRPO?**
   Answered only after configs and seeds are locked.

It also keeps the current research honest: the evidence already shows high seed
variance and collapse after learning, so the benchmark must not be based on best
checkpoint alone.

---

## 9. Current Project State

| Item | State |
|---|---|
| PPO/TRPO baselines | fixed and credible |
| REINFORCE baseline | fixed but weak/not yet working |
| PC actor-critic | learns, but fragile |
| PC-REINFORCE | implemented; needs systematic MuJoCo readout |
| HalfCheetah | active development benchmark |
| Hopper | next transfer benchmark |
| `target_scale` | important; 0.3 helps but does not solve |
| `max_t1` | negative as a fix; 40 worsened seed 2 |
| optimizer/activation | unresolved confound |
| critic tuning | unresolved; diagnose before changing |
| benchmark readiness | not final yet; mechanism tests first |

---

## 10. References for Reporting Norms

- Schulman et al. 2015, TRPO: trust-region policy updates were introduced to
  prevent overly large policy changes that can collapse performance.
- Schulman et al. 2017, PPO: PPO was proposed as a simpler approximate
  trust-region style method with strong sample-efficiency and wall-time behavior.
- Agarwal et al. 2021, RLiable / Statistical Precipice: report uncertainty,
  robust aggregates, performance profiles, and probability of improvement rather
  than relying on point estimates.
- Patterson et al. 2024, Empirical Design in RL: benchmark conclusions require
  explicit choices about metrics, seeds, hyperparameter bias, and stability.
