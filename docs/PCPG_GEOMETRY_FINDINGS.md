# Does the PCPG update have natural-gradient geometry? — measurement summary

**Status:** internal, for circulation. **Branch:** `fixing-pcpg` @ `610c909`.

**Answer:** no — the reason decomposes into three independent gaps (§4c), only one
of which is about predictive coding at all. Since first draft: **Gap 1 is fixed**
(inference now genuinely settles, at no measurable cost — §4d), and the ablation it
unblocked shows that **settling significantly slows learning** on the bandit (§4e).
That falsifies §4b's prediction *on that task* and shows §4b's "inference is inert"
result is **geometry-specific and does not transfer** — the MuJoCo/Gaussian case is
still untested and is the top outstanding experiment.

Scope note up front: §§1–4c are **Probe 1** — synthetic batch, weights at
initialisation (with σ-head bias shifts to place the regime), single seed, `n=256`,
CPU. That part answers *"what does this machinery compute"*, not *"what happens along
a real trajectory"*. §4e is the one real-training result here, and it is bandit-only.
Probe 2 (real checkpoints) is specified but unbuilt. See **Limitations**.

Environment: `jax 0.4.38`, `jpc 1.0.0`, `equinox 0.13.8`, `diffrax 0.7.2`.
Geometry: `jpc.make_mlp` 17 → 64 → 12, tanh, `ts=1.0`, seed 0.

---

## 1. What was already known, and what is new

`docs/PCPG_NATURAL_GRADIENT_PROBES.md` findings **F1–F5** are pre-existing (pilot,
`results/ng_probe_pilot/`). They **reproduce** on an independent machine:

| quantity | this run | pilot |
|---|---|---|
| floor / Euclidean `cos(Δ, d_NG*)`, t₁=20 → eq | 0.616 → **0.124** | 0.68 → 0.11 |
| floor / Euclidean `cos_F(Δ, d_NG*)`, t₁=20 → eq | 0.904 → **0.281** | 0.93 → 0.27 |
| mixed / Euclidean `cos(Δ, d_NG*)` eq | **0.202** | 0.20 |
| mixed / natural `cos(Δ, d_NG*)` eq | **0.618** | 0.60 |
| harness anchors at t₁=0 (`share_last`, block-cos) | 1.000000 | 1.000000 |

New in this pass: **F6** (state-independent-σ control), **F7** (the clamp mechanism
is misattributed), the **trust-region metric measurement** (§4), and a documented
**confound in experiment-log §3.4** (§5).

Raw logs: `results/ng_probe_rerun/`.

---

## 2. F6 — the fixed-σ control: with σ constant there is almost no geometry to get right

Added `--regimes fixed`: zero the `log_std` rows of the final weight so
`σ = exp(bias)` is state-independent, `bias = 0` → **σ ≡ 1**. This is PPO's
parameterisation and the one experiment-log §3.4 switches on via `state_indep_std`.

| regime | σ range | `cos(d_SGD, d_NG)` | `cos(d_OUT, d_NG)` |
|---|---|---|---|
| **fixed** | 1.000 – 1.000 | **0.941** | 0.933 |
| init | 0.419 – 1.649 | 0.925 | 0.850 |
| mixed | 0.135 – 1.162 | 0.622 | 0.723 |
| floor | 0.135 – 0.297 | **0.341** | 0.500 |

With σ constant, `F_out = diag(1, 2)` is nearly isotropic, so the natural gradient
and the vanilla gradient point almost the same way. The claim "PC ≈ NG" is close to
**unfalsifiable** there, because *every* direction is ≈ NG. The heteroscedastic
spread is what *creates* the gap the claim is about, and it is widest at the floor.

**Consequence for how we write this up:** `fixed` is the right **control** and the
wrong **test case**. Cite it to calibrate how much geometry is at stake; never as
evidence that the update is natural.

F2/F3 are regime-independent: `fixed` still shows `shL ≥ 0.97` (final-layer delta
rule) and `cos_NG*` *falling* with t₁ (0.589 → 0.491).

---

## 3. F7 — the `log_std` clamp binds through the target, not through where σ sits

F5 attributed the fidelity loss to σ being pinned at `LOG_STD_MIN`. That mechanism
is wrong, or at best secondary.

In `fixed`, σ ≡ 1 and **0.0%** of raw `log_std` outputs are out of bounds — yet
target fidelity on the `log_std` half is still **0.6995** (Euclidean) / **0.7902**
(natural), not 1.0. The clip in `gaussian_pc_targets` applies to
`log_scale + log_scale_offset`, so a target leaves the window whenever the *offset*
is large relative to the window width (2.5), wherever σ sits. Measured at σ ≡ 1:

| `target_scale` | `log_std` targets clipped | median \|offset\| |
|---|---|---|
| 1.0 (probe default) | 27.3% | 0.43 |
| **10.0 (bench `ts10`)** | **80.9%** | 4.27 |

**Every bench config is `ts10`** (`..._adam_tanh_ts10_bench_mt20_...`). So in
production roughly **four in five `log_std` targets are truncated** — in the
configuration that was supposed to be the clean one. F5's floor-regime fidelity
(0.37–0.39) is therefore two effects compounding, not one.

**Consequence:** any statement about what the `log_std` channel encodes must be
conditioned on `target_scale`, not only on σ.

---

## 4. Why Innocenti's trust region does not deliver a natural gradient

The working expectation has been that PC-at-equilibrium supplies natural-gradient
geometry, on the strength of the adaptive-trust-region result. Two things are true,
and they are separate.

*Caveat: this characterises the result as the repo's own probe does — settled PC =
backprop on an `S⁻¹`-rescaled loss, exact for linear nets — not from a fresh read of
Innocenti et al. (2305.18188).*

### 4a. The theorem and "natural gradient" are different claims

```
S_i = I + Σ_l B_{l,i} B_{l,i}ᵀ ,   B_{l,i} = ∂out_i/∂z_{l,i}    (trust region)
F   = Jᵀ F_out J ,                 F_out = diag(1/σ², 2)        (Fisher)
```

`S` is built from the **network Jacobian** — a property of the architecture. `F` is
built from the **output distribution's** KL geometry — a property of the policy.
Different objects. Nothing in PC ever sees the policy distribution; PC receives a
regression target. So the distributional geometry has to be injected at the output,
which is exactly what `natural_target` does — consistent with F4's finding that it
is the only NG ingredient present.

### 4b. Worse: in this architecture `S` is inert

New measurement (`scripts/probe_trust_region_metric.py`) — the existing probe
reported `cos(Δ, d_EQ)` but never how far `d_EQ` is from plain backprop, which is
the entire content of the trust-region claim:

| regime | hidden layers | eig(S) | cond(S) | `cos(d_BP, d_EQ)` | `cos(d_EQ, d_NG)` |
|---|---|---|---|---|---|
| mixed | 1 (bench) | 1.04 – 1.6 | 1.5 | **0.9948** | 0.931 |
| mixed | 2 | 1.11 – 1.8 | 1.6 | **0.9936** | 0.921 |
| mixed | 4 | 1.18 – 1.9 | 1.6 | **0.9921** | 0.912 |
| mixed | 8 | 1.18 – 2.0 | 1.7 | **0.9853** | 0.897 |
| fixed | 1 | 1.00 – 1.5 | 1.5 | **0.9930** | 0.632 |
| fixed | 4 | 1.00 – 1.7 | 1.7 | **0.9903** | 0.278 |
| fixed | 8 | 1.00 – 1.8 | 1.8 | **0.9949** | 0.076 |

`cos(d_BP, d_EQ) ≈ 0.99` everywhere: **PC-at-equilibrium is directionally
backprop.** The reason is the *condition number*, not the magnitude — a
preconditioner only rotates a gradient if it is anisotropic, and `cond(S) ≈ 1.5`
is nearly isotropic. `S` rescales step length and leaves direction essentially
untouched.

Two obvious escapes, both closed:

- **Depth does not rescue it.** The sum runs over hidden layers, so growth was
  expected — but `cond` creeps only 1.5 → 1.8 from 1 to 8 hidden layers.
- **Not a tanh-linearisation artifact.** In a **linear** network, where the
  equilibrium result is exact rather than first-order, `cos(d_BP, d_EQ)` is still
  0.985–0.995 (`--act-fn linear`).

Also note `cos(d_SGD, d_NG) = 1.0000` in `fixed` at every depth — F6, sharply.

### 4c. Three independent gaps

1. **Production never reaches equilibrium.** F1: the energy is batch-normalised, so
   settling needs `t₁/N ≈ 10–40`. Bench is `N=2048, max_t1=20` → `t₁/N ≈ 0.01`, and
   jpc hard-stops at `t₁=4096`. The theorem is a statement about equilibrium; we run
   ~3 orders short — hence the final-layer delta rule (F2).
2. **At equilibrium `S` is inert** — §4b.
3. **Even an active `S` is not the Fisher** — §4a, and the `cos(d_EQ, d_NG)`
   ceiling, which collapses to 0.076 at depth 8 in the control.

Gap 1 is about our configuration. Gap 2 is about this architecture. Gap 3 is
conceptual and does not go away with tuning.

---

## 4d. Gap 1 is now fixed — and the fix has a price worth knowing

**Diagnosis.** Three compounding causes, all in `jpc`:

1. The energy is batch-normalised (`F = 1/2N Σᵢ …`), so `dz/dt = −∂F/∂z` carries a
   `1/N`. Because each sample's activities are *independent* variables, that factor
   is a pure rescaling of the inference **clock** — it does not move the fixed
   point. Confirmed: the residual is a function of `t₁/N` **alone**, hitting 0.39 of
   its initial value at `t₁/N ≈ 0.6` for every N tested.
2. `max_t1` cannot buy the time back: jpc hard-codes
   `timeout_reached = t >= 4096` in `steady_state_event_with_timeout`.
3. That event's steady-state test is `rms_norm(y) < atol + rtol·rms_norm(y)`, i.e.
   `rms_norm(activities) < ~1e-3` — a test on the **activities**, not on `dz/dt`.
   Activities are O(1), so it never fires; inference just runs the clock out.

**Fix** (`src/pc_algorithms/inference.py`, flag `train.inference_rate_correction`,
default off): integrate the *same* ODE at per-sample rate by scaling the inference
vector field by N, while the weight step keeps jpc's batch-normalised
`compute_pc_param_grads` so gradients remain a mean and learning rates carry over.

**Validation** (`scripts/probe_inference_settling.py`):

| check | result |
|---|---|
| same fixed point as jpc integrated to `t₁ = 40N` | `cos = 1.000000`, max rel. diff ~6e-4 |
| relative residual at `t₁ = 20`, corrected | 4.5e-3 – 4.9e-3, **independent of N** |
| relative residual at `t₁ = 20`, jpc | 0.68 (N=64) → 0.95 (N=512) — *worsens* with N |

So the existing `max_t1 = 20` default is already sufficient once the rate is right.
End-to-end smoke test passes through YAML on the bandit (`TRAINING END`).

### How much inference we ran vs how much is needed

`τ = t₁/N` is per-sample inference time. The residual is a function of `τ` **alone**
(N=64 and N=256 agree to 3 decimals), so one curve covers every batch size:

| τ | residual / initial | |
|---|---|---|
| **0.00977** | **0.988** | ← what production ran (`max_t1=20`, N=2048) |
| 0.03 | 0.962 | |
| 0.1 | 0.873 | |
| 0.3 | 0.645 | |
| 1.0 | 0.218 | |
| 3.0 | 0.026 | 90% settled |
| **10.0** | **0.002** | ← converged; τ=30 and τ=100 do not improve on it |

So production removed **1.2% of the residual** and needed **τ ≈ 10** — about a
**1000× shortfall** in inference time (`t₁ = 20` run, `t₁ ≈ 20,480` required at
N=2048 *without* the rate correction).

With the correction, `max_t1` *is* τ, so the existing `max_t1 = 20` already
over-settles (τ=20 vs the τ=10 needed). Measured at the real bench N=2048:
residual 0.988 → **0.0037**. `max_t1 = 10` would also suffice.

**Cost: ~9.5× per PC step at bench scale.** Measured jitted at N=2048, mixed regime:
0.045 s/update unsettled vs **0.428 s settled**. The hardware-independent driver is the
adaptive solver's step count per inference solve, which carries over to GPU:

| arm | τ | solver steps |
|---|---|---|
| jpc, `max_t1=20` (what all committed runs did) | 0.0098 | **5** |
| corrected, `max_t1=20` | 20 | **62** |
| corrected, `max_t1=10` | 10 | **39** |

`max_t1=10` already converges (table above) and costs ~1.6× less than 20, so prefer it.

> ⚠ **Two retractions, both from measuring in the wrong regime.** An earlier draft
> claimed ~80×; that was a missing `@eqx.filter_jit` on `make_pc_step_at_rate`
> re-tracing the solve on every call (`jpc.make_pc_step` is itself `filter_jit`'d) —
> genuinely a bug, genuinely fixed. But the *replacement* claim that settling is then
> "effectively free" was also wrong, for two separate reasons: the bandit figure
> (0.199 s vs 0.221 s) is real but comes from a tiny net where inference is not the
> bottleneck, and the bench-scale figure behind it was a **non-jitted**
> `settle_activities` call whose wall-clock was tracing-dominated, hiding a 5 → 62
> step difference. The jitted numbers above supersede both.
>
> Corroborating evidence that should have been noticed earlier: the committed
> `mt10/20/40/80` runs all took the *same* wall-clock (403/407/406/412 s). That is
> itself a symptom of never settling — τ stayed in [0.005, 0.039], where the solver
> coasts in ~5 steps regardless of `max_t1`.

Because settling is *not* free, the fixed-point alternative (solve `∂F/∂z = 0`
directly instead of integrating to it — a linear solve when `Π_L` is frozen at the
feedforward σ) is worth building if the ablation shows settling matters.

---

## 4e. The settled-vs-unsettled ablation — settling DOES change learning

Run (`scripts/run_settled_ablation.py`, `results/settled_ablation/`): bandit,
favor_suboptimal (π₀=0.018), 60k steps, seeds 1–5, both PC algorithms, matched
seeds, 20/20 runs completed. Arms are `train.inference_rate_correction` off/on,
i.e. residual **0.90 → 0.003** at the bandit's N=256.

**Final π(opt) shows nothing — but it has no power.** Both arms hit 5/5 success at
π ≈ 0.98; the task is saturated at 60k steps.

| algo | unsettled | settled | paired Δ | verdict |
|---|---|---|---|---|
| pc_reinforce | 0.989 ± 0.003 | 0.987 ± 0.003 | −0.002 ± 0.002 | n.s. (t(4)=−1.00) |
| pc_actor_critic | 0.992 ± 0.003 | 0.977 ± 0.008 | −0.015 ± 0.007 | n.s. (t(4)=−2.30) |

**Steps-to-π≥0.9 uses the whole curve, and it is unambiguous — settling is slower:**

| algo | paired Δ steps | t(4) | 95% CI |
|---|---|---|---|
| pc_reinforce | **+3661 ± 407** | +9.00 | [+2532, +4790] |
| pc_actor_critic | **+5085 ± 1702** | +2.99 | [+361, +9809] |

All 5 pc_reinforce seeds slower; 4/5 pc_actor_critic slower, 1 tie. So **the §4b
prediction ("settling changes nothing") is falsified on this task** — but not for the
reason it would seem.

**Why, and why §4b is not overturned.** Measuring the update directly in the
*bandit's* geometry (discrete softmax head, 1→32→2), settled vs unsettled over 5
seeds: `cos = 0.67–0.89` (mean ≈ 0.78) and `‖Δ‖` ratio 1.05–1.65 (mean ≈ 1.31). So
inference here **rotates the update substantially** — unlike the Gaussian
MuJoCo-like geometry of §4b (17→64→12), where `cos(d_BP, d_EQ) ≈ 0.99`. And the
slowdown is not a step-size artifact: settled steps are *larger* yet converge later,
so a changed direction, not a changed magnitude, is doing the work.

**The real lesson: §4b's "inference is inert" result is geometry-specific and does
not transfer.** It was measured on one architecture; the bandit's is different and
behaves differently. The MuJoCo/Gaussian claim therefore still needs its own
settled-vs-unsettled training run — which is GPU-blocked here, and is now the
highest-priority outstanding experiment.

**Also actionable:** every committed bandit result was produced at ~10% settled, and
every MuJoCo result at ~1.2%. Settling measurably changes the bandit ones (speed, not
final performance). Whether it changes the MuJoCo conclusions is unknown.

---

## 5. Methodological: experiment-log §3.4 cannot support its stated conclusion

§3.4 (`trust_region_kl_stdglobal`) concludes that PCPG "performs dramatically worse
with a state-independent σ" (returns −451 to −647). Two uncontrolled factors each
explain that on their own:

1. **No trust region is enforced anywhere.** Despite the `trust_region_kl_*` family
   name, `policy_kl` is computed only as a diagnostic
   (`pc_actor_critic.py:477,495-496`). There is no KL rejection, backtracking, or
   step rescaling in `src/pc_algorithms/` — `grep target_kl|kl_limit|reject|
   backtrack|line_search` returns nothing. These configs set neither `target_clip`
   nor `max_grad_norm`, and Adam renormalises `max_grad_norm` away regardless. A
   realised `kl_max` of **170** against TRPO's `target_kl` of **0.01** is the
   *absence* of a trust region, not a loose one. The only bounded cells (SGD,
   `kl_max` 0.6–1.0) never learned (best ≤ 73) — so **no cell had both a bounded
   step and learning**.
2. **The `log_std` channel was ~81% truncated** (F7): these are `ts10` configs, so
   the `state_indep_std` arm did not cleanly test the parameterisation it names.

What §3.4 *does* support: PPO's σ parameterisation does not by itself rescue PCPG.
A clean test needs a bounded step at matched `target_scale`.

---

## 6. Limitations

- **Initialisation geometry only.** Weights are at init plus σ-head bias shifts.
  Trained checkpoints may sit in better-conditioned geometry where `S` is
  anisotropic — this is the single most likely way §4b softens, and it is exactly
  what Probe 2 exists to settle.
- **Single seed, `n=256`, synthetic batch** (distributionally matched: obs ~ N(0,1)
  as after Welford normalisation, `z` on-policy, advantages standardised).
- **One architecture family** (`jpc.make_mlp`, tanh/linear, width 64). The depth
  sweep varies architecture but stays at init.
- `cos(d_EQ, d_NG)` uses the best-fit damping λ per cell; λ is part of the
  measurement, not a fixed constant.
- **Side finding, still open:** `jpc.make_mlp(depth=2)` builds **one** hidden layer,
  not the "[64,64]" the bench config comment claims. Baseline-comparability claims
  against PPO/TRPO `[256,256]` need re-checking.

---

## 7. Recommended next steps, in priority order

1. **Run the settled-vs-unsettled ablation** — it is now cheap (§4d: no measurable
   cost) and it is the single highest-value experiment available. Does
   `inference_rate_correction=true` change learning at all? Given §4b
   (`cos(d_BP, d_EQ) ≈ 0.99`) the prediction is that it does **not**. Either outcome
   is informative: no change confirms that PC-at-equilibrium is backprop in this
   architecture; a change means `d_EQ` is missing something and §4b needs revisiting.
   Every prior PCPG result was produced at ~1% settled, so this also tells us whether
   any of them need re-running.
2. **Enforce a trust region** (KL cap or `target_clip`). Independent blocker: no σ or
   geometry conclusion has a valid arm until it exists (§5).
3. **Decouple `target_scale` from the clamp** so the `log_std` channel encodes what
   theory assumes — lower ts on that channel, widen the window, or use relative
   clipping (§3).
4. **Build Probe 2** (checkpoint dump hook). The one route by which §4b could change.
5. **Stop treating inference as the source of geometry.** `natural_target=True` is
   the only lever that injects `F_out⁻¹`; that is where distributional geometry has
   to enter.
6. **If we want to test the trust-region result on its own terms**, we need `S`
   anisotropic — larger `‖B_l‖` via init gain or width — and should validate on a
   linear net first, where it is exact. As configured, the experiment cannot
   distinguish the theorem from backprop.

---

## Appendix — reproducing

```bash
pip install "jax==0.4.38" "jaxlib==0.4.38" "flax==0.10.2" "optax==0.2.4" \
    git+https://github.com/thebuckleylab/jpc

python scripts/probe_natural_gradient.py                     # 4 regimes incl. fixed
python scripts/probe_natural_gradient.py --regimes fixed     # the control alone
python scripts/probe_trust_region_metric.py                  # S spectrum + cos(d_BP,d_EQ)
python scripts/probe_trust_region_metric.py --act-fn linear  # theorem's exact case
```

Both probes are CPU-only and need no env. Logs in `results/ng_probe_rerun/`.

**Unrelated status:** the MuJoCo baseline grid (seeds 4–5 × 4 envs × 2 algos = 16
cells, the outstanding `BENCHMARK_PROTOCOL.md` §7 requirement) has a Modal runner on
branch `claude/pcpg-fix-modal-experiments-t6tt73` but **has not been run** — Modal's
client is gRPC-only and the sandbox that produced this writeup cannot reach it.
