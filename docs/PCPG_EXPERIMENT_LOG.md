# PCPG on HalfCheetah — complete experiment log

Every run, what it asked, what it showed. All numbers extracted from the seed logs
in `results/`, not from notes. Claims that rest on timing are marked **[verified]**
with the exact steps.

**Context.** PCPG = predictive-coding policy gradient. The theoretical claim is that
the PC weight update approximates a *natural gradient* at inference convergence. The
empirical goal on HalfCheetah is not to beat PPO (PPO ≈ 4.4k @ 1M, PCPG ≈ 0.9k) but
to get the continuous-control update **reliable across seeds** — the prerequisite for
testing where PC beats backprop.

**Status in one line:** we have the first 0-collapse 3-seed result
(`SGD + natural target + max_t1=20`, **781 ± 47**), we know four stabilisation
strategies that *don't* work and why, and the mechanism of the remaining collapses
is **not yet established** — only a signature.

---

## 1. The update, and where each knob acts

```
advantage A ─▶ target = μ + ts·A·(z−μ)/σ²  ─▶ PC inference (max_t1 steps) ─▶ weight grad ─▶ optimizer step
                    │          │        │                                          │           │
             natural_target  target_  log_std_min                          max_grad_norm   adam/sgd
                             clip     (σ floor)                            (SGD only)
```

Actions are `a = tanh(z)`, `z ~ N(μ,σ)` — the *same* squashed-Gaussian PPO uses in
this repo (`NormalTanhDistribution`, `src/networks/distributions.py`).

---

## 2. Headline results

| recipe | final | collapse | note |
|---|---|---|---|
| **SGD + natural + mt20** | **781 ± 47** | **0/3** | best stable result at 1M |
| Adam + natural + mt20 | 425 ± 491 | 1/3 | same target, unstable |
| Adam Euclidean mt20 (baseline) | 649 ± 644 | 1/3 | higher peak, unreliable |
| SGD Euclidean mt20 | 279 ± 466 | 2/3 | `1/σ²` is corrosive under SGD |
| Adam + global σ (PPO-style) | −599 ± 968 | 2/3 | catastrophic |
| Adam capacity-matched 5M | up to **2937** | 2/3 | best peak ever seen; unreliable |

---

## 3. Sweeps, in order

### 3.1 `trust_region_kl` — the baseline matrix (16 configs, 48 runs, 1M)

*Question: optimizer × inference length × algorithm, no stabilisers.*

| config | final | collapse | kl_max | value_ev |
|---|---|---|---|---|
| adam mt10 | 649 ± 440 | 1/3 | 0.49 | +0.39 |
| adam mt20 | 649 ± 644 | 1/3 | 0.42 | +0.30 |
| adam mt40 | 391 ± 547 | 1/3 | 0.46 | +0.38 |
| adam mt80 | 699 ± 441 | 0/3 | 2.91 | +0.34 |
| sgd mt10 | 406 ± 475 | 1/3 | 3.14 | −0.72 |
| sgd mt20 | 279 ± 466 | 2/3 | 5.15 | −0.75 |
| sgd mt40 | −62 ± 535 | 1/3 | **58.5** | −0.59 |
| sgd mt80 | −99 ± 405 | 2/3 | **295.8** | −0.71 |
| pc_reinforce (all 8) | 8–66 | 0/3 | 0.04–0.08 | n/a |

**Findings.** (a) The Euclidean `1/σ²` target under SGD produces *enormous* policy
jumps — `kl_max` up to **296** — and gets worse with more inference. (b) Adam's
adaptive rescaling masks this (`kl_max` ≈ 0.4). (c) Adam has positive value-EV,
SGD strongly negative: two different critic regimes. (d) **PC-REINFORCE without a
critic does not learn at bench scale** (finals 8–66) — the value head is essential.

### 3.2 `trust_region_kl_clip` — gradient clipping (9 configs, 26 runs)

*Question: does `max_grad_norm` stabilise?*

| config | final | collapse | vs baseline |
|---|---|---|---|
| adam mt10 clip0.5 | 579 ± 154 | 0/2 | changed |
| adam mt10 clip1.0 | 855 ± 161 | 0/3 | 2 of 3 seeds bit-identical |
| adam mt20 clip1.0 | 649 ± 644 | 1/3 | **bit-identical** |
| adam mt40 clip1.0 | 391 ± 547 | 1/3 | **bit-identical** |
| adam mt80 clip1.0 | 875 ± 91 | 0/3 | changed (kl_max 2.91→1.43) |
| sgd mt10–80 clip1.0 | **−11 ± 2** | 0/3 | **dead** |

**[verified]** Adam mt20 and mt40 with `clip=1.0` produce **bit-identical finals** to
no-clip (`[1121, 1088, −261]` and `[675, 870, −374]`) — the clip *never fired*,
because gradient norms sit at ~0.2–0.4. Earlier I attributed this to Adam
renormalising the clip away; the real reason is simpler and stronger: **the threshold
was above the gradient distribution.**

**[verified] The SGD clip runs are confounded.** `config.yaml` shows
`learning_rate: 0.0003` versus the working SGD baseline's `0.03` — **100× too small**.
They died of the learning rate, not the clipping. This invalidated the sweep and
motivated §3.6.

### 3.3 `trust_region_kl_tclip` — output-space target clip (6 configs, 18 runs)

| config | final | best | collapse |
|---|---|---|---|
| adam mt20 tclip2 | 618 ± 566 | 1027 | 1/3 |
| adam mt20 tclip5 | 691 ± 269 | 945 | 1/3 |
| adam mt40 tclip2 | 705 ± 664 | 1069 | 1/3 |
| adam mt80 tclip2 | 781 ± 547 | 1039 | 1/3 |

**Finding.** Capping the target offset **raises peaks** (up to 1069, the best at bench
scale) but **never removes the collapse** — 1/3 in every cell. An output-space bound
is not the missing constraint.

### 3.4 `trust_region_kl_stdglobal` — PPO-style global σ (8 configs, 24 runs)

| config | final | collapse | kl_max | sat_max |
|---|---|---|---|---|
| adam mt10 | −647 ± 548 | **3/3** | 48.9 | 0.86 |
| adam mt20 | −599 ± 968 | 2/3 | **151.9** | 0.61 |
| adam mt40 | −524 ± 47 | 1/3 | 169.8 | 0.91 |
| adam mt80 | −451 ± 84 | 1/3 | 170.2 | 0.73 |
| sgd (all) | −46 … −0 | 0/3 | 0.6–1.0 | 0.06 |

**Finding.** A **state-independent** `log_std` — exactly the parameterisation PPO
uses successfully here — is **catastrophic for PCPG**: `kl_max` up to 170, saturation
to 0.91, negative returns. The per-state σ head is load-bearing. This is the
strongest evidence that PCPG's instability is *not* a shared "MuJoCo/tanh" issue: the
same parameterisation is fine for PPO in this codebase.

### 3.5 `trust_region_kl_natural` — the natural target ⭐ (4 configs, 12 runs)

*Question: does dropping the `1/σ²` amplifier (Fisher-preconditioned target) fix it?*

| config | final | best | AUC | collapse | kl_max |
|---|---|---|---|---|---|
| **sgd mt20 nat** | **781 ± 47** | 862 ± 52 | **441 ± 18** | **0/3** | **0.04** |
| sgd mt80 nat | 357 ± 433 | 802 ± 114 | 384 ± 71 | 0/3 | 0.04 |
| adam mt20 nat | 425 ± 491 | 753 ± 28 | 284 ± 116 | 1/3 | 0.30 |
| adam mt80 nat | 369 ± 480 | 812 ± 116 | 341 ± 118 | 1/3 | 0.30 |

Seeds for the winner: **759 / 738 / 846** — unusually tight for this project.

**Finding.** The natural target is the only strategy that produces a clean 3-seed
result — **but only under SGD**. Same target under Adam still spikes to `kl_max` 0.30
and still collapses 1/3. So this is an **optimizer × target interaction**, not a
property of the target alone. `mt80` adds seed variance without benefit.

![natural target](../results/trust_region_kl_natural/collapse_anatomy_sgd_tanh_ts10_bench_lr003_mt20_nat.png)

### 3.6 `gradclip_probe` — Marco's check, done properly (4 configs, 10 runs)

*Question: are the collapses rare numerical gradient spikes that a guard would catch?
Run on the **Euclidean** update so `1/σ²` is retained.*

Measured tail (clip-free): **p50 0.671, p90 0.941, p99 4.387, p99.9 10.24** → tail
ratio **6.5×**. A real heavy tail exists, as Marco predicted.

| clip | bind rate | kl_max | final | collapse |
|---|---|---|---|---|
| none | 0% | 6.25 | 267 (n=1) | 1/1 |
| 10.24 (p99.9) | 0.07% | 12.1 / 8.9 / 0.31 | 294 ± 466 | 2/3 |
| 4.39 (p99) | 0.47% | 3.49 / 2.48 / 0.31 | 280 ± 503 | 2/3 |
| 1.46 (p99/3) | **1.56%** | **0.46 / 0.53 / 0.37** | 272 ± 475 | 1/3 |

**Finding.** The guard **worked and was not enough.** At clip 1.46 it fired on 1.56%
of updates and cut `kl_max` from 6–12 to ~0.4 — a >10× reduction in realised step
size — and returns did not move (267 → 294 → 280 → 272). Per-seed, outcome is fixed
by seed identity, not clip: seed 2 collapses under every clip (−285/−371/−257), seed
3 is fine under every clip (855/855/895).

**Conclusion: structural, not numerical.** Bounding step *magnitude* is not what the
natural target provides (272 vs 781). Not ruled out: gradient *direction*, Adam
preconditioning, per-parameter spikes, sample-level gradients.

![gradclip](../results/gradclip_probe/collapse_anatomy_sgd_euclid_mt20_lr003_clip1p4623.png)

### 3.7 `knob_fill_smin_vlr` — the two rescue knobs (4 configs, 12 runs)

| knob | value | final | best | collapse |
|---|---|---|---|---|
| `log_std_min` (Adam nat mt20) | −2 (base) | 425 ± 491 | 753 | 1/3 |
| | −1.5 | 514 ± 576 | 866 | 1/3 |
| | −1.0 | 496 ± 551 | **893** | 1/3 |
| `value_lr` (SGD nat mt80) | 1e-4 | 310 ± 381 | 662 | 0/3 |
| | 3e-4 (base) | 357 ± 433 | 802 | 0/3 |
| | 1e-3 | 496 ± 419 | 674 | 0/3 |

**Finding.** Both negative for stability. The σ floor lifts peaks (753→893) and is
demonstrably active (`log_std_mean` −0.28 vs −0.79 baseline) yet collapse stays 1/3.
Critic LR moves the mean but not the ±420 spread. Neither beats SGD+natural+mt20.

### 3.8 5M-scale sweeps

**`benchmark_halfcheetah_pcpg_5m_27runs_20260721`** (9 configs, 27 runs, 5M,
[256,256], 1024 envs) — *note: `results/overnight_halfcheetah_sota_sweep{,_01}/` are
empty scaffolding for this same sweep; the data is here.*

| config | final | best | collapse |
|---|---|---|---|
| adam ts05 lr0002 | 1349 ± 1537 | 1821 | 2/3 |
| adam ts07 lr0002 | 1313 ± 784 | 1560 | 3/3 |
| adam ts07 lr0001 | 1097 ± 500 | 1442 | 2/3 |
| adam ts06 | 848 ± 887 | 1538 | 2/3 |
| sgd ts05 lr001 | 676 ± 1038 | 719 | 0/3 |
| sgd ts05 lr0003 | 164 ± 267 | 176 | 0/3 |
| **sgd ts05 lr003** | **−192 ± 55** | 1348 | **3/3** |

**`benchmark_halfcheetah_capacity_5m`** — single best seed ever: **2937** (adam ts07),
but 2/3 collapse; ts05 gave 2902 then −604/−480.

**Finding.** Capacity raises the ceiling (≈2900 vs ≈1100) and does **not** fix
reliability. Critically, **SGD at lr=0.03 — the bench winner's LR — collapses 3/3 at
5M**, so the §3.5 recipe is *not* safe to promote unchanged. Lower LR (0.01) is
0/3 but weak.

**`pcr_sota`** (PC-REINFORCE, 8 configs, 24 runs, 5M): peaks to **2139** then
catastrophic collapse — finals ≈ −500 to −600 on most seeds, saturation to **1.000**.
Confirms §3.1: no critic ⇒ high ceiling, no floor.

---

## 4. Verified claims and their evidence

| claim | evidence | status |
|---|---|---|
| SGD+natural+mt20 is 0-collapse at 1M | seeds 759/738/846 | **holds (n=3)** |
| Natural target alone doesn't stabilise | Adam+nat still 1/3, kl_max 0.30 | **holds** |
| Gradient clipping doesn't rescue | 1.56% bind, kl 6→0.4, returns flat | **holds** |
| Adam clip1.0 was a no-op in 2 cells | bit-identical finals | **[verified]** |
| Old SGD clip runs are invalid | `lr=0.0003` vs `0.03` in config.yaml | **[verified]** |
| Global σ is catastrophic | −599, kl_max 152 | **holds** |
| Transformation matches PPO | same `NormalTanhDistribution`; Jacobian in `log_prob`; no `μ,σ` dependence in tanh correction | **holds (code)** |
| ~~KL shock causes the collapse~~ | see below | **RETRACTED** |
| Deterministic-eval explains it | see below | **REFUTED** |

### 4.1 Retracted: "a shared KL shock knocks the run over"

**[verified]** Across 16 collapsing runs, the `kl_max` spike is **after the collapse
trough in 5**, **before the peak in 2**, and inside the (often ~500k-step) decline
window in 9 — where it carries little information. Concretely, `sminm10` seed 3:
eval peaks 491k, troughs 737k, `kl_max` at **778k** — *after it was already dead*.
And in that config the seed with the **largest** spike (seed 2, 0.150) **survived at
859**, while the crashing seed 3 had the **smallest** (0.108). The spike is not the
cause.

### 4.2 Refuted: "stochasticity protects training; only deterministic eval collapses"

**[verified]** `training/mean_reward` is the stochastic policy. In **6 of 9**
collapsing runs it falls too, and in the two Adam natural cases it **changes sign**
(+0.0128 → −0.0111; +0.0166 → −0.0102). The policy genuinely degrades.

### 4.3 Current best signature (correlational only)

Gradient norms carry **no** collapse signal under the natural target — crashing seeds
have equal or *lower* norms than survivors (0.204→**0.199** while crashing; healthy
0.264). What does track, across all three families:

| | `\|μ\|` growth | saturation |
|---|---|---|
| crashing (7 runs) | **1.2–3.1×** | rises |
| healthy (6 runs) | 1.11–1.27× | ~flat |

**This is a signature, not a cause.** Saturation *level* does not discriminate —
healthy `sminm10` seed 1 runs at 0.119 saturation and scores 911, *higher* than
crashing seed 3's 0.099. Only the *growth* separates them, and `|μ|` growth could be
cause, symptom, or bystander.

---

## 5. What is not established

1. **Causality of `|μ|` growth.** Needs the frozen-checkpoint intervention:
   evaluate a collapsed checkpoint under `tanh(clip(μ,−b,b))`. If return recovers,
   μ-magnitude is functionally responsible; if not, it is downstream.
2. **What makes μ grow.** Untested: gradient *direction* (cosine similarity across
   updates — moderate but aligned gradients would produce exactly the observed slow
   monotone drift), Adam's preconditioned update norm `‖m̂/(√v̂+ε)‖` vs `‖g‖`,
   per-layer/per-head decomposition.
3. **Whether the winner survives 5M.** §3.8 says probably not as-is (SGD lr=0.03
   collapses 3/3 at 5M).
4. **Statistical power.** Every cell is n=3.

### Blocker
All sweep scripts run `--no-save`, so **no checkpoints exist**. Every decisive
intervention in (1) and (2) requires periodic checkpoints *plus* optimizer moments,
RNG state, and obs/reward normaliser state. That is the prerequisite work.

---

## 6. Reproduce

```bash
python scripts/run_pcpg_knob_fill.py --seeds 1 2 3 --skip-complete
python scripts/run_gradclip_probe.py --stage auto --seeds 1 2 3
python scripts/analyze_pcpg_logs.py --results-dir results/<dir>
python scripts/plot_collapse_anatomy.py --results-dir results/<dir>
bash scripts/run_overnight_batch.sh            # both, unattended, auto-stops the pod
```

Each `results/<dir>/<config>/` holds `config.yaml` (exact resolved config),
`meta.json` (git commit, timestamp), `seed_N.log` (per-update diagnostics).

**Naming gotcha:** `lr003` = 0.03, `lr0003` = 0.003.

## 7. Folder map

| folder | configs | runs | steps |
|---|---|---|---|
| `trust_region_kl` | 16 | 48 | 1M |
| `trust_region_kl_clip` | 9 | 26 | 1M |
| `trust_region_kl_stdglobal` | 8 | 24 | 1M |
| `trust_region_kl_tclip` | 6 | 18 | 1M |
| `trust_region_kl_natural` | 4 | 12 | 1M |
| `knob_fill_smin_vlr` | 4 | 12 | 1M |
| `gradclip_probe` | 4 | 10 | 1M |
| `benchmark_halfcheetah` | 7 | 21 | 1M |
| `benchmark_halfcheetah_pcpg_5m_27runs_20260721` | 9 | 27 | 5M |
| `benchmark_halfcheetah_capacity_5m` | 3 | 9 | 5M |
| `pcr_sota` | 8 | 24 | 5M |
| `overnight_halfcheetah_sota_sweep{,_01}` | 9 | **0** | — (empty; data in the 27-run folder) |

Every folder with logs has `per_run.csv`, `SUMMARY.md`, `learning_curve.png`,
`diagnostic_plots.png`, and per-config `collapse_anatomy_<config>.png`.
