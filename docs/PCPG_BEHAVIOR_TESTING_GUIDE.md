# PCPG Behavior and Systematic Testing Guide

Status: working research guide. This document explains how the current PCPG
implementation behaves, what is likely problematic, and how to test it
systematically without turning every surprising seed into a new ad hoc theory.

Related docs:

- `docs/MUJOCO_IMPLEMENTATION.md` explains the Brax/MJX environment and backprop
  baselines.
- `docs/PCPG_EXPLORATION_FINDINGS.md` records empirical HalfCheetah findings.
- `docs/PCPG_TUNING_METHODOLOGY.md` gives the formal staged tuning protocol.

---

## 1. What the current PCPG algorithm is doing

The actor-critic PCPG implementation is in
`src/pc_algorithms/pc_actor_critic.py`. It has two independent predictive-coding
MLPs:

```text
policy_model: observation -> Gaussian policy parameters [mu, log_std]
value_model:  observation -> scalar value V(s)
```

For MuJoCo, the environment has continuous actions, so the policy is Gaussian:

```text
z ~ Normal(mu, std)
action = tanh(z)
```

The environment receives `tanh(z)`, but the learning target is built in the
pre-tanh Gaussian space. After a rollout, the critic computes GAE advantages and
value targets. Then the algorithm performs two predictive-coding updates:

```text
critic PC update:
  input  = observation
  output = value target
  model  = value_model

actor PC update:
  input  = observation
  output = Gaussian policy target
  model  = policy_model
```

The PC update itself is delegated to `jpc.make_pc_step(...)`. This repo does not
manually implement hidden-activity inference or precision matrices; it creates
the RL-derived output targets and asks `jpc` to fit the PC network to them.

---

## 2. Where the PC part enters

In ordinary actor-critic, the code would define losses:

```text
policy loss = -log pi(a|s) * advantage
value loss  = (V(s) - value_target)^2
```

and use backpropagation to update weights.

In this PCPG code, the RL losses are converted into supervised output targets.
Then predictive coding updates the network weights toward those targets.

For the critic, the target is direct regression:

```text
V(s) -> value_target
```

For the actor, the target is the Gaussian policy-gradient direction:

```text
target_mu =
  mu + target_scale * advantage * (z - mu) / std^2

target_log_std =
  log_std + target_scale * advantage * (((z - mu) / std)^2 - 1)
```

The code clips the `log_std` target but does not clip the `mu` target. That
matters: `mu` targets can become very large when advantages are large or `std` is
small. This is one of the central instability risks.

---

## 3. What PC inference means here

Predictive coding treats layer activities as variables. For one PC update:

```text
fixed during inference:
  input activity
  output target
  weights

moving during inference:
  hidden-layer activities
```

The network begins with a feedforward pass. Then hidden activities are adjusted
to reduce prediction-error energy:

```text
prediction_error_l =
  actual_activity_l - prediction_from_layer_below_l

energy =
  sum over layers of weighted squared prediction errors
```

In the general theory, the weighting can be written with precision matrices:

```text
E = sum_l 1/2 * error_l^T Precision_l error_l
```

In the current implementation, no explicit precision matrix is passed from this
repo into `jpc.make_pc_step`. Practically, treat the precision as fixed/default
inside `jpc` rather than as a learned or tuned object in this project.

After hidden activities settle, weights are updated so each layer better predicts
the settled activity above it. The activity update is the inference phase; the
weight update is the learning phase.

The main PC controls exposed by this repo are:

```text
max_t1:
  number of internal inference/settling steps per PC update

pc_steps_per_update:
  number of PC weight updates per minibatch

optimizer:
  optax optimizer used for the final weight update, usually adam or sgd

learning_rate / value_learning_rate:
  actor and critic weight-update step sizes
```

---

## 4. Current likely problems

The current implementation can learn HalfCheetah, but it is not robust. The
main problems are likely these.

### 4.1 Unbounded mean targets

The Gaussian `log_std` target is clipped, but `target_mu` is not. Because:

```text
target_mu offset = advantage * (z - mu) / std^2
```

rare samples can produce large actor targets. A global `target_scale` reduces the
average size but does not eliminate tails.

Observed consequence: policy drift spikes, `mu_abs_mean` grows, tanh actions
saturate, and performance can collapse after a promising peak.

### 4.2 Inference is not automatically a trust region

Increasing `max_t1` was tested on the known-collapsing seed 2:

```text
max_t1=20: best 410, final -285, drift_max 2.76
max_t1=40: best  71, final -476, drift_max 14.46
```

This refutes the narrow hypothesis that under-equilibration was causing the
collapse for that seed. More inference made the actor chase the bad target more
completely. So `max_t1` is not currently a rescue lever.

### 4.3 Critic diagnostics are ambiguous

Bad `value_explained_var` often appears near collapse, but this does not prove
the critic is the root cause. In local seed-2 logs, `value_pc_loss` can become
small while `value_explained_var` later becomes very negative. This suggests a
nonstationary or poorly scaled value-target problem, not simply "the critic loss
is too high."

Critic questions to separate:

```text
underfitting:
  value_pc_loss stays high and EV stays poor

unstable/nonstationary targets:
  value_pc_loss is low or falling, but EV oscillates or goes negative

actor-driven critic damage:
  policy drift/return collapse occurs first, then value estimates become bad
```

### 4.4 Optimizer and activation are confounded

The best partial regime found so far is `sgd+tanh+lr=0.03`, but this changed two
things at once relative to `adam+relu`. We do not yet know whether the effect is
from SGD, tanh, their interaction, or the larger actor learning rate.

### 4.5 High seed variance may be the finding

The same configuration produced outcomes ranging from a stable positive final
return to collapse. That means single-seed knob tests are useful for mechanism
checks, but not enough to declare a configuration solved.

---

## 5. Where we stand on parameter tests

Current evidence should be read as exploratory, not final.

| Lever | Status | Current interpretation |
|---|---|---|
| `target_scale: 1.0 -> 0.3` | useful but insufficient | Reduces typical policy drift and catastrophic collapse, but rare large targets remain. |
| `max_t1: 20 -> 40` | negative result | More inference worsened seed 2; do not spend more on this as a fix. |
| `sgd+tanh`, actor LR screen | partial win | `lr=0.03` gave bounded drift and learning; `0.01` too slow; `0.1` unstable. |
| optimizer x activation | unresolved | Need 2x2: `adam+relu`, `adam+tanh`, `sgd+relu`, `sgd+tanh`. |
| critic LR / critic capacity | unresolved | Diagnose temporal order first; do not blindly raise value LR. |
| `pc_steps_per_update` | mostly untested | Likely increases target chasing and compute; test only after target scale/drift behavior is understood. |
| advantage normalization | enabled in main runs | Needed for scale control, but may hide heavy tails; inspect full advantage distributions. |
| reward normalization | enabled in main runs | Stabilizes scale, but makes critic targets nonstationary; test only with careful logs. |
| mean-target clipping | not in current final method | Likely stabilizes tails, but changes the algorithm; treat as a separate ablation. |

---

## 6. What to test next

The next tests should answer specific questions in order. Avoid changing more
than one mechanism per test unless the comparison is explicitly factorial.

### Step 1: Parse existing logs into a diagnostic table

For every run, extract:

```text
best eval return
final eval return
return AUC
max / p95 / final-window policy_drift_max
max / p95 mu_target_mag_max
mean / final-window value_explained_var
value_pc_loss trend
pretanh_sat_frac final and max
log_std_mean/min and saturation fractions
wall-clock
```

Do this before launching more jobs. The first target is temporal ordering:

```text
Does EV fail before drift?
Does drift spike before eval collapse?
Does tanh saturation rise before eval collapse?
Do target tails precede all of the above?
```

### Step 2: Resolve the optimizer x activation confound

Run the 2x2 on the same budget and seeds:

```text
adam + relu
adam + tanh
sgd  + relu
sgd  + tanh
```

Keep the same `target_scale`, `max_t1`, rollout length, batch settings, and eval
frequency. This answers whether the observed improvement belongs to SGD, tanh, or
their interaction.

### Step 3: Actor target magnitude test

If drift spikes remain the strongest predictor, test actor target control:

```text
target_scale = 0.1, 0.3, 1.0
```

Optionally add a separate research ablation:

```text
clip target_mu offset to K sigma, e.g. K in {1, 2, 5}
```

Do not mix target clipping into the main method unless it is explicitly declared
as a modified algorithm.

### Step 4: Critic mechanism test

Only after log ordering suggests critic involvement, test:

```text
value_learning_rate x {0.3, 1, 3, 10}
value pc steps x {1, 2}
critic width/depth if underfitting is clear
gae_lambda x {0.9, 0.95, 0.97}
```

Read results by EV trend and actor stability, not just final return.

### Step 5: Final robustness check

For any candidate that looks good:

```text
run at least 5 seeds before calling it stable
report all seeds
report best, final, AUC, collapse count, and wall-clock
```

Given current variance, 3 seeds can reveal fragility but cannot establish
robustness.

---

## 7. Suggested experiment matrix

Use HalfCheetah bench tier first. Freeze:

```text
env: halfcheetah
budget: 1M env steps for medium confirmation
rollout_length: 32 for actor-critic
num_envs: 256 or current bench value
normalize_advantages: true
normalize_rewards: true
eval cadence: fixed across all runs
```

Minimal next matrix:

| Question | Configs | Seeds | Decision |
|---|---|---:|---|
| Optimizer/activation | 4 corners | 2-3 | Pick mechanism, not winner only. |
| Target scale | 0.1, 0.3, 1.0 | 3 | Check drift tail vs learning speed. |
| Critic LR | 0.3x, 1x, 3x, 10x | 2-3 | Only if critic temporal evidence supports it. |
| Candidate robustness | top 1-2 configs | 5-10 | Estimate collapse frequency. |

Stop criteria:

```text
stop max_t1 testing as a fix;
stop any config family whose drift tails grow while returns degrade;
stop tuning if all promising configs remain seed-fragile after target/critic tests;
then formalize fragility as the result.
```

---

## 8. How to write the comprehensive algorithm behavior report

The report should not be organized as "we tried knobs." It should be organized by
mechanism:

1. **Algorithm statement**
   - Actor and critic are PCNs.
   - RL produces output targets.
   - PC inference and learning fit the networks to those targets.

2. **Continuous-control target derivation**
   - Why MuJoCo needs a Gaussian policy.
   - How `mu`, `log_std`, `z`, and `advantage` produce the target.
   - Why `std` and target tails matter.

3. **PC learning dynamics**
   - Input/output clamping.
   - Hidden-activity inference.
   - Prediction-error energy.
   - Weight update through `jpc.make_pc_step`.
   - Exposed implementation knobs: `max_t1`, `pc_steps_per_update`, optimizer, LR.

4. **Environment and data pipeline**
   - Brax/MJX vectorized HalfCheetah.
   - Observation normalization.
   - Reward normalization.
   - Rollout buffer.
   - GAE and value targets.

5. **Observed behavior regimes**
   - No learning / too small updates.
   - Learning then collapse.
   - Learning then slow degradation.
   - Stable partial success.

6. **Diagnostic interpretation**
   - Drift as realized policy movement.
   - `mu_target_mag` as desired target pressure.
   - `value_explained_var` as critic quality, with caveats.
   - `pretanh_sat_frac` and `mu_abs_mean` as action saturation.
   - `log_std` saturation as exploration pathology.

7. **Ablations and conclusions**
   - What each parameter test can and cannot prove.
   - Which hypotheses were refuted.
   - Which remain open.
   - Which conclusions are robust vs exploratory.

8. **Final recommendation**
   - Whether to pursue a stabilized algorithmic variant or report the current
     method as fragile/high-variance.

---

## 9. Current bottom line

The implementation is coherent as a PCPG actor-critic: the actor and critic are
both updated by predictive coding toward RL-derived targets. The likely problem
is not a missing PC call or an obviously wrong environment wrapper. The problem
is behavioral: the Gaussian actor target can create large unconstrained mean
targets, and PC inference does not automatically bound the realized policy
update. Current evidence says `target_scale` and the SGD/tanh regime help, but
the algorithm remains seed-fragile. The next decisive work is diagnostic log
analysis, the optimizer/activation 2x2, target-tail control, and only then critic
tuning.
