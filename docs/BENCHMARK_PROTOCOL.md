# Benchmark Protocol (FROZEN) — PCPG continuous-control evaluation

**Status:** APPROVED (advisor sign-off, Marco B., see §0).
**Frozen on:** _<date>_  ·  **Git commit:** _<hash>_  ·  **Owner:** _<you>_

Once signed off, nothing in §1–§8 changes for the paper's main results. Any change
after freeze requires a new commit hash here and a re-run of *all* affected cells
(not just the new method). This is the calibration target: the harness must
reproduce standard baseline behavior before any PCPG claim is trusted.

---

## 0. Decisions (advisor-signed)

| # | Decision | Resolution | Sign-off |
|---|---|---|---|
| 1 | Need Gymnasium MuJoCo results for parity, or is Brax/MJX-only OK? | **Brax/MJX-only.** Old TRPO/PPO papers used older gym/MuJoCo → no direct comparison exists regardless. | ☑ |
| 2 | Authoritative env backend | **MJX-backed Brax** (named explicitly, see §1) | ☑ |
| 3 | Primary scientific claim | **Return-vs-env-steps learning curve** is the headline; **final return at a fixed budget** is the secondary metric. Wall-clock/sample-eff not central. | ☑ |
| 4 | One-time external calibration on Gymnasium (SB3/CleanRL)? | **Optional / deprioritized.** Not required for the claim (no direct cross-version comparison anyway). Keep as a private sanity check only if PPO/TRPO look off. | ☑ |
| 5 | Freeze the whole protocol before large sweeps? | **Yes** ("sounds like a plan"). | ☑ |

---

## 1. Environment suite (authoritative)

- **Backend:** Brax `backend="mjx"` (MuJoCo physics compiled to XLA). This is what the
  repo is built on (`src/env/mujoco.py`). We report it as **"Brax/MJX"**, never as
  bare "MuJoCo" — rewards, termination, horizon, and action scaling are **not**
  identical to Gymnasium `*-v5` and we will not claim they are.
- **Gating task:** `halfcheetah` — a representative env. **Get all algorithms
  working here first** (esp. TRPO); only then expand the suite.
- **Expansion (once HalfCheetah works):** `hopper`, `walker2d`, `ant`.
  **Stretch (compute permitting, not currently run):** `humanoid`.
- **Gymnasium parity is explicitly out of scope** (decision #1): we do not claim
  numerical equivalence to gym/MuJoCo and do not need v5 results.
- **Episode length:** 1000 (`episode_length=1000`, `action_repeat=1`).
- **Action space:** continuous, `tanh`-squashed to [-1, 1] (`NormalTanhDistribution`).
- Pinned versions (must match for every run): `jax==0.4.38 jaxlib==0.4.38
  brax==0.11.0 mujoco==3.2.7 mujoco-mjx==3.2.7`, Python 3.11. Record CUDA/GPU per run.

## 2. Algorithms

REINFORCE, PPO, TRPO, **PCPG (ours)**. All share the env, network, normalization,
eval, and seed protocol below. Only the update rule differs.

## 3. Network & optimizer (identical across algorithms)

- Separate actor/critic MLPs, **CleanRL/Engstrom "SOTA" spec** (`sota_init=true`):
  orthogonal init (√2 hidden / 0.01 actor-head / 1.0 critic-head), **tanh**,
  **state-independent `log_std`** init 0 (σ=1), `σ=exp(log_std)`.
- Hidden sizes: **[64,64]** default; **[256,256]** for HalfCheetah (capacity-limited
  at 64). Same sizes used for *every* algorithm on a given task.
- Optimizer: Adam, lr `3e-4`, `eps=1e-5`, global-grad-norm clip `0.5`.
  (TRPO: Adam trains the *value* net only; policy uses the natural-gradient step.)

## 4. Training budget (same env-step budget for all algorithms)

| Task | Env steps |
|---|---|
| hopper, walker2d | 3M |
| halfcheetah, ant | 5M |
| humanoid (stretch) | 10M |

- Compared at **equal environment interactions**, never equal update counts.
- **Per-algorithm rollout config is locked here** (these are correctness/regime
  requirements, not tuning knobs):
  - **PPO:** 256 envs, rollout 16, 32 minibatches × 4 epochs.
  - **TRPO:** 256 envs, **full-batch Fisher** (`num_minibatches=1`), **long rollout**
    (≥ 256, *not* 16 — short horizon was the cause of non-convergence), 1 natural-
    gradient step/rollout, `target_kl=0.01`, cg_iters 10, damping 0.1, line-search
    10/shrink 0.8. Budget skews high because TRPO does ~1 update/rollout.
  - **REINFORCE:** `num_envs=1`, `num_minibatches=1` (rollout requires single env).
    Documented weak floor.
- Log per run: env steps, wall-clock, #parallel envs, batch size, rollout length,
  #updates, gradient steps/update.

## 5. Evaluation

- Every **N=** _<fill, e.g. 100k>_ env steps, freeze policy and run **K=10** episodes.
- **Deterministic** policy mean (`a = tanh(μ)`), no exploration noise.
- **Separate eval env + eval seeds**; eval **never** updates train normalization stats.
- Returns reported in **raw reward units** (train-time reward normalization is
  train-only and must not leak into eval).
- Base all claims on **evaluation** return. Training return = debugging only.
- **No "best-over-evals" / max-cherry-picking.** Report final + aggregate (see §7).

## 6. Normalization (state explicitly per run)

- **Observation:** running Welford mean/std, **train-only**, frozen at eval — ON.
- **Reward:** return-std scaling, **train-only** — ON for MuJoCo.
- **Advantage:** per-minibatch normalization — ON.
- GAE: γ=0.99, λ=0.95. (Note: repo returns the Brax λ-return advantage, not the
  literal GAE-paper sum — documented, consistent across all algos.)

## 7. Seeds & statistics

- **5 seeds** per (task, algorithm) as the baseline across the board. Compute CIs;
  **add more seeds only if the CIs are too wide to separate methods** (advisor call).
- **Identical seed set across algorithms.**
- **Headline figure (decision #3): return vs. env-steps learning curve**, mean line
  with a 95% CI band, per task — this is the primary result.
- **Secondary: final return at the fixed budget**, reported with uncertainty across
  seeds (mean + 95% bootstrap CI). Use **IQM / probability-of-improvement** when
  *comparing PCPG to baselines*; for single-algorithm sanity, mean + CI is fine.
- No bare mean±std as the headline; no "best-over-evals".

## 8. Diagnostics (logged for every run, regardless of headline claim)

KL per update, entropy, gradient norm, value explained-variance, advantage spread,
episode length, termination frequency, policy σ. For TRPO also: line-search success
rate, realized KL vs `target_kl`, surrogate improvement. For PPO: clip fraction,
approx_kl.

## 9. Logging artifacts (per seed)

One raw CSV/JSON per seed · full config snapshot · git commit hash · dependency
lockfile · hardware metadata. Stored under `results/<suite>/<task>_<algo>_seed<k>/`.

## 10. Pre-sweep gate (must pass before large runs)

- [ ] **TRPO converges on HalfCheetah** (the gating task) — the current blocker.
      Likely fix: longer rollouts + bigger batch + higher step budget (it's
      update-starved, not buggy: line search succeeds, KL ≈ 0.006 < target,
      surrogate improves each step).
- [ ] TRPO line-search success ≈ 1.0 and realized KL ≈ target on a smoke run.
- [ ] Unit tests pass: `compute_gae` vs hand-computed trajectory; TRPO Fisher-vector
      product vs explicit dense Fisher on a tiny policy; line-search KL-respect.
- [ ] PPO reaches plausible regime on HalfCheetah (sanity, not SOTA). ✅ (~2.6k final)
- [ ] (Optional, decision #4) external SB3/CleanRL PPO check — only if results look off.
- [ ] Protocol §1–§8 confirmed (§0 table all ☑) and commit hash recorded above.

REINFORCE is **not** a gate: it is expected to underperform on these envs (advisor
confirmed) and serves only as the weak floor.

---

### Validity scope (state in the paper)
> Our harness reproduces standard PPO/TRPO/REINFORCE behavior on Brax/MJX
> continuous control; all methods use identical environment-step budgets, network
> architectures, normalization, evaluation procedures, and seed sets; results are
> reported with uncertainty-aware aggregate metrics (IQM, 95% bootstrap CIs).
> Environments are Brax/MJX and are **not** claimed to be numerically identical to
> Gymnasium MuJoCo `v5`.
