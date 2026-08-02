# PCPG natural-gradient probes — does the PC update implement F⁻¹∇L?

**Question.** The project's working claim is that the PC weight update
approximates a *natural gradient* at inference convergence (see
`gaussian_policy.py:62-66`, `likelihood_energy.py:3-14`, experiment log §3.5).
This document designs experiments that measure that claim directly instead of
inferring it from training curves — for the **non-clipped** configuration
(`target_clip: null`, `max_grad_norm: null`, raw pre-optimizer update — the
direction SGD realises), in **both** target families:

| family | flag | mean-target offset |
|---|---|---|
| Euclidean — *with* variance division | `natural_target: false` | `ts·A·(z−μ)/σ²` |
| natural — *without* variance division | `natural_target: true` | `ts·A·(z−μ)` |

Note "non-clipped" leaves one clamp in place that cannot be turned off: the
`log_std` clamp to `[LOG_STD_MIN, LOG_STD_MAX]` inside `gaussian_pc_targets`
(`gaussian_policy.py:86-87`). It is part of the production target, so the
probes *measure* its distortion rather than remove it (the "fidelity" metric
below). It turns out to matter a lot (finding F5).

## 1. Making the claim falsifiable

For the Gaussian head, the policy-gradient surrogate on a frozen batch
`{(s_i, z_i, A_i)}` is

```
L(θ) = -(ts/N) Σ_i A_i · log N(z_i; μ_θ(s_i), σ_θ(s_i)²)
```

whose exact parameter-space Fisher factors through the network Jacobian
`J_i = ∂out_i/∂θ`:

```
F = (1/N) Σ_i J_iᵀ F_out,i J_i ,    F_out = diag(1/σ², 2)   (μ / log_std halves)
```

So the natural gradient `F⁻¹∇L` needs **two ingredients**, and the codebase's
two knobs supply at most one each:

* **output metric `F_out⁻¹`** — supplied (or not) by the **target family**:
  the natural target pre-multiplies the score by `F_out⁻¹ = [σ², ½]`; the
  Euclidean target keeps the raw `1/σ²` score;
* **network factor** — could only come from **inference**: how the settled
  activities reshape the mapping of output errors into parameters.

"Our PCPG has a natural gradient" therefore decomposes into two measurable
sub-claims: (a) the target encodes `F_out⁻¹` (checkable at the update level),
and (b) inference contributes the network factor. Each probe below measures
the actual update direction `Δ(t₁) = −∂E/∂θ` at settled activities against
reference directions computed on the same frozen `(θ, batch)`:

| reference | definition | hypothesis it represents |
|---|---|---|
| `d_SGD` | `−∇L` | vanilla policy gradient (Euclidean) |
| `d_OUT` | `−Jᵀ P u`, `P=[σ², ½]`, `u=∂L/∂out` | output-space natural only — exactly what the natural target encodes |
| `d_NG(λ)` | `−(F+λI)⁻¹∇L` | **the natural gradient** (both factors); λ swept, best-fit λ is part of the measurement |
| `d_BP` | `−∇_θ E_composed` (backprop on the target energy) | Millidge-style "PC ≈ BP"; equals the intended gradient up to the log_std-clamp truncation |
| `d_EQ` | `−∇_θ[(1/2N)Σ r_iᵀS_i⁻¹r_i]`, `S_i = I + Σ_l B_{l,i}B_{l,i}ᵀ`, `B_{l,i}=∂out_i/∂z_{l,i}` (S stop-gradiented) | the *equilibrium prediction*: settled PC = BP on an output-rescaled loss — exact for linear nets (Innocenti et al., 2305.18188), first-order for tanh |

`F` is the exact (deterministic, GGN-form) Fisher applied matrix-free via
jvp/vjp + CG — no action-sampling noise. `S_i` is built exactly per sample
(output dim is small).

**Metrics.** Euclidean cosine `cos(Δ, d)` globally and per layer; the
**Fisher-metric cosine** `cos_F(Δ, d_NG) = ⟨Δ, F d_NG⟩ / √(⟨Δ,FΔ⟩⟨d_NG,F d_NG⟩)`
(parameterisation-invariant: = 1 iff Δ buys the most L-improvement per unit
KL); norm `|Δ|`; final-layer norm share `shL`; inference residual
`‖∂E/∂z‖` and activity shift; target fidelity `cos(u_eff, u_intended)` per
output half, where `u_eff = (out−target)/N` is what the energy actually
descends.

**Built-in anchors** (harness validation, must hit 1.0 to float precision):
at `t₁=0` every hidden error is identically zero, so the PC update is
*final-layer-only* and its final-layer block equals `d_BP`'s exactly.
Verified: `share_last = 1.000000`, block-cos `= 1.000000` in all runs.

## 2. Probe 1 — synthetic geometry probe (implemented, validated)

`scripts/probe_natural_gradient.py` — no env, no training, CPU, ~10 min.

Freezes a bench-geometry policy (`jpc.make_mlp`: 17 → 64 → 12, tanh, ts=1.0 —
the `sgd_tanh_ts10_bench_lr003_mt20` winner geometry) and an on-policy
synthetic batch (obs ~ N(0,1) as after Welford normalisation, `z` sampled
from the policy, advantages standardised). Builds targets with the production
`gaussian_pc_targets` (both families, `target_clip=None`), runs jpc's own
`init_activities_with_ffwd → solve_inference → compute_pc_param_grads` chain
(Heun + PID(1e-3), exactly `make_pc_step` minus the optimizer) over a sweep of
integration times `t₁`, and prints all metrics against all references.

Three σ regimes position the geometry (the independent variable):
`init` (σ≈1, output metrics nearly degenerate), `mixed` (per-dim log_std bias
U[−1.8, 0.3] — the late-training mix of floored and healthy dims), `floor`
(bias −1.95, σ≈0.14 — the `1/σ²`≈49× amplifier regime).

```
python scripts/probe_natural_gradient.py                    # full pilot
python scripts/probe_natural_gradient.py --seeds…           # see Modal below
```

### Pilot results (seed 0, n=256; full logs in `results/ng_probe_pilot/`)

**F1 — the inference ODE is batch-normalised, so production inference is
~1% settled.** jpc's energy is `F = (1/2N)Σᵢ…`, so `dz/dt = −∂F/∂z` runs N×
slower than its per-sample dynamics: the residual decays on the `t₁/N` scale
(settled by `t₁/N ≈ 5`, confirmed at n∈{32,128,256}). The bench operating
point is minibatch **N=2048 with `max_t1=20` → t₁/N ≈ 0.01**; and jpc
hard-stops integration at `t₁=4096` (`steady_state_event_with_timeout`), so a
bench minibatch **cannot be equilibrated in production at all** (would need
t₁ ≈ 10-80k). This also explains the log's "mt80 adds seed variance without
benefit": 20/2048 and 80/2048 are both ≈ 0.

**F2 — at the operating point the PCPG update is a final-layer delta rule.**
At `t₁=0` the update is exactly final-layer-only (anchor), and the final-layer
share is still ≥ 0.99 at `t₁/N ≈ 0.08` (8× the bench point). Under SGD the
hidden layer effectively does not train; under Adam the tiny hidden gradients
get per-coordinate renormalised — a mechanism candidate for the log's
SGD-vs-Adam asymmetry, measurable online via Probe 3. (Even at equilibrium the
final-layer share stays 0.93-1.00 in this geometry.)

**F3 — inference does not move the update toward the natural gradient; at
equilibrium it moves it away, catastrophically so for the Euclidean family.**
Selected numbers (best-λ values; `t₁=20` ≈ operating point, `t₁=4096` ≈
equilibrium):

| regime, family | cos(Δ,d_NG*) t₁=20 → eq | cos_F(Δ,d_NG*) t₁=20 → eq | cos(Δ,d_BP) eq | \|Δ\| growth | z-shift eq |
|---|---|---|---|---|---|
| mixed, Euclidean | 0.65 → **0.20** | 0.92 → **0.57** | 0.24 | ×4.8 | 0.42 |
| mixed, natural | 0.68 → 0.60 | 0.72 → 0.71 | 0.79 | ×1.1 | 0.14 |
| floor, Euclidean | 0.68 → **0.11** | 0.93 → **0.27** | 0.12 | ×3.9 | 0.59 |
| floor, natural | 0.13 → 0.14 | 0.61 → 0.61 | 0.97 | ×0.9 | 0.06 |

The Euclidean update at equilibrium tracks *nothing* gradient-like — not
`d_BP`, not `d_EQ` (its output layer decorrelates: per-layer cosEQ L1 ≈ 0.11-0.24
while the hidden layer still tracks, cosEQ L0 ≈ 0.74-0.84), not `d_NG`. The
huge targets (`1/σ²` amplifier) push activities 40-60% away from feedforward —
far outside the linearised regime the equilibrium theory (and the NG claim)
lives in. The natural family stays in that regime (z-shift 0.06-0.14) and
matches `d_EQ` well (floor: cos_EQ = 0.98) — the equilibrium theory *works*
there — but `d_EQ` itself is not the natural gradient (ceilings
`cos(d_EQ, d_NG)` ≈ 0.74-0.75 mixed/init, **0.14 floor**).

**F4 — the natural-gradient content that exists comes from the target, not
from PC.** In the F-metric, the *unsettled* (t₁≈0) update is already the most
natural-gradient-aligned thing the algorithm ever produces (Euclidean mixed/floor:
cos_F ≈ 0.96-0.98 at t₁=0 — a well-aligned but 49×-oversized direction, which is
exactly the "right direction, catastrophic magnitude" collapse story; natural:
cos_F ≈ 0.61-0.72 stable in t). Settling only degrades it. So at *any* t₁, the
`natural_target` flag's `F_out⁻¹` is the only NG ingredient present.

**F5 — the log_std clamp corrupts what the natural target encodes exactly in
the regime that matters.** Target fidelity (log_std half): 0.72-0.82 in
init/mixed, **0.47-0.48 at the floor**, and at the floor
`cos(d_BP, d_OUT) = 0.19` — i.e. with σ pinned at the clamp, the energy no
longer descends anything close to the intended Fisher-preconditioned score.
(Mechanism: for pinned coords the raw output sits below `LOG_STD_MIN`, and the
clamped target yields a constant push back toward the clamp boundary
regardless of the score's sign or size.) Any theory-level claim about the
natural target silently assumes this clamp is inactive.

**Side finding** — `jpc.make_mlp(depth=2)` (the bench `Config.depth = 2`)
builds **one** hidden layer (obs→64→out), not the "[64,64]" the bench config
comment claims (`configs/mujoco_halfcheetah_pc_actor_critic_bench.yaml:1`).
Worth fixing the comment and re-checking baseline-comparability claims.

### Added regime: `fixed` — the state-independent-σ control (seed 0, n=256)

`--regimes fixed` zeroes the `log_std` rows of the final weight so `σ = exp(bias)`
is constant across states, with `bias = 0` → **σ ≡ 1**, mid-window so the clamp
cannot bind on the raw output. This is PPO's parameterisation and the one
experiment-log §3.4 switches on via `state_indep_std`. It is the reference point
the three heteroscedastic regimes are read against. Full log:
`results/ng_probe_rerun/probe_natural_gradient_4regimes.txt`.

**F6 — with σ state-independent there is almost no natural-gradient geometry to
get right.** The separation between the references collapses:

| regime | σ range | `cos(d_SGD, d_NG)` | `cos(d_OUT, d_NG)` |
|---|---|---|---|
| **fixed** | 1.000 – 1.000 | **0.941** | 0.933 |
| init | 0.419 – 1.649 | 0.925 | 0.850 |
| mixed | 0.135 – 1.162 | 0.622 | 0.723 |
| floor | 0.135 – 0.297 | **0.341** | 0.500 |

With σ constant, `F_out = diag(1, 2)` is nearly isotropic, so the natural gradient
and the vanilla gradient point almost the same way — the claim "PC ≈ NG" is nearly
unfalsifiable there because *every* direction is ≈ NG. The heteroscedastic spread is
what *creates* the gap that the claim is about, and it is widest at the floor. So
`fixed` is the right **control** and the wrong **test case**: quote it to calibrate
how much geometry is at stake, never as evidence that the update is natural.

Note F2/F3 are regime-independent: `fixed` still shows `shL ≥ 0.97` (final-layer
delta rule) and `cos_NG*` *falling* with `t₁` (0.589 → 0.491), so inference
contributes no network factor here either.

**F7 — the `log_std` clamp binds through the *target*, not through where σ sits.
The mechanism in F5 is misattributed, and the clamp is essentially never inactive
at the bench operating point.** In `fixed`, σ ≡ 1 and *0.0%* of raw `log_std`
outputs are out of bounds — yet target fidelity on the `log_std` half is still
**0.6995** (Euclidean) / **0.7902** (natural), not 1.0. The clip in
`gaussian_pc_targets` applies to `log_scale + log_scale_offset`, so a target leaves
the window whenever the *offset* is large relative to the window width (2.5), no
matter how central σ is. Measured at σ ≡ 1, `log_std` targets clipped:

| `target_scale` | clipped | median \|offset\| |
|---|---|---|
| 1.0 (probe default) | 27.3% | 0.43 |
| **10.0 (bench `ts10`)** | **80.9%** | 4.27 |

Every bench config is `ts10` (`..._adam_tanh_ts10_bench_mt20_...`), so in production
roughly **four out of five `log_std` targets are truncated** — with a mid-window,
state-independent σ, i.e. in the configuration that was supposed to be the clean
one. F5's floor-regime fidelity (0.37–0.39) is then two effects compounding, not
one: σ pinned at the boundary *and* oversized offsets. Any claim about what the
`log_std` channel encodes has to be conditioned on `target_scale`, not just on σ.

### ⚠ Scope limit of the `d_EQ` reference — it cannot test the saddle-escape claim

Innocenti et al. (2305.18188, *Understanding Predictive Coding as an Adaptive
Trust-Region Method*) state the headline prediction as **PC escapes saddle points
faster than BP**, with the dynamics *interpolating* between BP's loss-gradient
direction and a TR direction, proved in a shallow linear model. Two consequences for
how we read `d_EQ`:

1. **BP is one endpoint of the predicted interpolation**, so `cos(Δ, d_BP) ≈ 1` is
   the theory's own degenerate case, not a refutation of it.
2. **`d_EQ` goes inert exactly at a saddle.** `S = I + Σ_l B_l B_lᵀ` with
   `B_l = ∂out/∂z_l` is built from products of weight matrices, so it vanishes as
   weights approach the origin saddle of a linear net. Measured (linear net, depth 5,
   all weight matrices scaled by ε):

   | ε | cond(S) | `cos(d_BP, d_EQ)` |
   |---|---|---|
   | 1.00 | 1.62 | 0.99097 |
   | 0.30 | 1.04 | 0.99995 |
   | 0.10 | 1.00 | 1.00000 |
   | 0.01 | 1.00 | 1.00000 |

   So the rescaling is *most* inert in the regime where the paper predicts the
   *largest* effect.

`d_EQ` therefore encodes a **local direction-rescaling** reading — BP on an
`S⁻¹`-rescaled loss at a fixed `(θ, batch)`. Saddle escape is a **landscape and
dynamics** claim: that inference moves the critical points, so PC's energy has no
saddle where BP's loss does. A linear reparameterisation of the gradient at one
point cannot express that. **F3/F4 should therefore be read as "inference supplies no
natural-gradient/output-metric content", which is what they measure, and NOT as
evidence against the trust-region result.** Testing that needs a saddle-escape
experiment (initialise at/near a saddle, compare escape time under PC vs BP with
plain GD), which is not implemented.

Two further conditions we do not currently satisfy, worth stating before any claim
about the theorem: the property is cited in `PCPG_TUNING_METHODOLOGY.md:137` as
holding for **plain gradient descent**, whereas every headline bench cell is
**Adam** — and since the measured effect of `S` is on step *magnitude* rather than
direction, Adam's per-coordinate renormalisation removes it. And equilibrium is
required, which production did not reach until `inference_rate_correction`
(see `PCPG_GEOMETRY_FINDINGS.md` §4d).

*Sourcing note: the paper's full text was not retrievable from this environment
(arxiv/openreview/semanticscholar all blocked at the egress proxy). The above uses
the title and abstract plus the repo's own citations; the exact theorem conditions
have not been checked against the text.*

### Probe-1 caveats (why Probe 2 exists)

Weights are at initialisation (with σ-head bias shifts); trained checkpoints
may sit in better-conditioned geometry. The batch is synthetic (though
distributionally matched). Single seeds are reported; the Modal runner fans
out seeds. Probe 1 answers "what does this machinery compute"; Probe 2
answers "on the real trajectory".

## 3. Probe 2 — real-checkpoint probe (specified, needs the dump hook)

Same measurement, fed with production artifacts. Requires a small
training-loop addition (default-off, zero behavior change when off):

* **Dump hook** in `pc_actor_critic.py` — a `Config.probe_dump_every = None`
  knob; when set, every K training steps serialise
  `policy_model` (`eqx.tree_serialise_leaves`, as the existing save block
  `pc_actor_critic.py:524-531` does) plus the probe slice already assembled at
  `pc_actor_critic.py:332-343` (`observations`, `pre_tanh_flat`, `advantages`
  — post-normalisation, i.e. exactly what the targets see) to
  `results/ng_probe/<run_name>/step_<k>.{eqx,npz}`. Map the key in
  `scripts/run_train.py` KEY_MAP.
* **Probe loader** — `--checkpoint step_<k>.eqx --batch step_<k>.npz` flags on
  `probe_natural_gradient.py` (deserialise onto the `make_mlp` skeleton,
  replace the synthetic batch; everything downstream is unchanged).

**Run matrix** (per family; both non-clipped bench configs
`…sgd_tanh_ts10_bench_lr003_mt20` and `…_nat`):

| axis | values |
|---|---|
| checkpoints | init, 100k, 500k, pre-collapse (pick via `kl_max` from the run log), 1M |
| seeds | 3 (the §3.5 seeds) |
| minibatches | ≥3 per checkpoint |
| t₁ | {0, 20, 320, 1280, 4096} + t₁/N-matched points |
| λ | {1e-3 … 1e2} |
| batch size | 2048 (production) + 256 (equilibrium reachable) |

Key questions it settles: do the F2/F3/F4 patterns hold on trained weights?
Does natural-gradient alignment *drift over training* (e.g. collapse of
cos_F right before the KL spike — connect to the collapse-anatomy plots)?
How much does the clamp distortion (F5) bind late in training, when
`frac_std_at_min` is high?

## 4. Probe 3 — online training diagnostics (specified)

Cheap per-update numbers logged into the existing `diag/` stream during real
runs, tying the probe to the regime where collapse actually happens:

* `diag/ng_share_last` — final-layer share of the PC gradient. **Free**: jpc
  already returns per-layer norms (`policy_result["model_grad_norms"]`,
  `pc_actor_critic.py:428`).
* `diag/ng_cos_bp` — cos(Δ, d_BP) with d_BP = one extra backprop of
  `0.5·mean‖targets − f(obs)‖²` on the minibatch. One fwd+bwd per update.
* `diag/ng_cos_nat`, `diag/ng_cosF_nat` — cos and cos_F against `d_NG(λ)`
  via CG every K=50 updates (~40 extra fwd+bwd per measurement).
* `diag/ng_kl_sigma_slope` — regression slope of per-state update-KL against
  `log σ(s)` on the existing probe slice (`pc_actor_critic.py:466-500` already
  computes both pieces). Natural-gradient updates predict slope ≈ 0; the
  `1/σ²` amplifier predicts strongly negative.

Prediction to test against §3.5: under Adam the *realised* step direction
(post-optimizer) should lose whatever cos_F the raw update had; under SGD it
is preserved by construction. That is the "stability = optimizer-target
interaction" mechanism made measurable.

## 5. Probe 4 — reparameterisation invariance (optional corroboration)

Fisher-free cross-check that needs no CG: natural-gradient updates induce the
same *distributional* change ΔKL(s) under any smooth reparameterisation of the
head. Concretely: scale the σ-head rows of the final layer by α and redefine
`log σ = raw/α` (same policy, different parameters); re-run the PC update;
compare induced per-state (Δμ(s), Δlogσ(s)). Invariance ⇒ natural;
α-dependence following the vanilla-gradient formula ⇒ Euclidean. Useful as an
independent validation of the CG machinery, and as a teaching artifact.

## 6. Decision rules

* **"PCPG-as-run has a natural gradient"** requires, at the production
  operating point (t₁/N ≈ 0.01, natural family, non-clipped), across
  checkpoints × seeds × minibatches: `cos_F(Δ, d_NG(λ)) ≥ 0.9` for some λ in
  the sweep, and Euclidean-family cos_F materially lower (else the flag is
  irrelevant and something else explains it).
* **"PC contributes the network factor"** requires alignment to *rise* toward
  `d_EQ`-then-`d_NG` as measured residual falls (dose-response in t₁), with
  the `d_EQ→d_NG` ceiling high. Pilot: falsified at init-geometry (F3/F4);
  Probe 2 decides on trained weights.
* **Anchor gate**: any run whose t₁=0 anchors miss 1.0 is a harness bug, not
  a finding.

Pilot-level verdict (Probe 1, init geometry): **no natural gradient beyond
the target preconditioning** — the update at the operating point is a
final-layer delta rule whose only NG ingredient is the natural target's
`F_out⁻¹`; running inference longer would not fix this and at equilibrium
actively un-does it (Euclidean) while the log_std clamp corrupts it at the
floor (natural). The §3.5 stability result is *consistent with the probe*:
the natural target tames the last-layer step in the KL metric (F4) — no
appeal to "PC = natural gradient" needed.

## 7. Validity notes

* **Pre-optimizer.** All measurements are on the raw gradient — the direction
  SGD realises. Adam re-preconditions it (`pc_actor_critic.py:77-79`); Probe 3
  measures that separately. Any claim about "the PC update" that survives only
  under one optimizer is about the optimizer.
* **Minibatch Fisher.** F is computed on the same batch as Δ (as TRPO does);
  this is the right comparison for direction geometry, not a population claim.
* **Damping.** `d_NG(λ)` direction depends on λ; we sweep λ and report the
  best fit plus the full line — a claim that holds only at one λ is reported
  as such.
* **Clamp flat-spots.** The Fisher's log_std rows are zeroed where the raw
  output is outside `[LOG_STD_MIN, LOG_STD_MAX]` (clip has zero gradient);
  `d_OUT` uses the production (unmasked) `[σ², ½]` so it represents exactly
  what `gaussian_pc_targets` encodes.
* **x64.** The probe runs in float64 with CG tol 1e-12; CG residuals are
  checked. Cosines are scale-free, so jpc's energy normalisation does not
  affect alignments (it *does* set the inference timescale — F1).

## 8. Running on Modal

`scripts/run_probe_modal.py` packages Probe 1 (and, once the dump hook lands,
Probe 2 — point it at a volume with checkpoints):

```
pip install modal
modal token new                      # once, or set MODAL_TOKEN_ID/SECRET
modal run scripts/run_probe_modal.py                              # pilot
modal run scripts/run_probe_modal.py --seeds "0 1 2 3 4"          # seed sweep
modal run scripts/run_probe_modal.py --seeds "0 1 2" \
    --probe-args "--n 2048 --t1 0 20 4096"                        # bench-size N
```

Logs stream to the terminal and persist in the `pcpg-probe-results` Modal
volume (`modal volume ls pcpg-probe-results`). CPU-only (`cpu=8`) is
appropriate at bench geometry; scale the decorator only if the net grows.
