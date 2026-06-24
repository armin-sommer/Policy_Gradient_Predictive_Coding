# Benchmark Protocol (FROZEN) — PCPG continuous-control evaluation

**Status:** DRAFT pending advisor sign-off on the 5 open decisions below.
**Frozen on:** _<date>_  ·  **Git commit:** _<hash>_  ·  **Owner:** _<you>_

Once signed off, nothing in §1–§8 changes for the paper's main results. Any change
after freeze requires a new commit hash here and a re-run of *all* affected cells
(not just the new method). This is the calibration target: the harness must
reproduce standard baseline behavior before any PCPG claim is trusted.

---

## 0. Open decisions (advisor sign-off required before freeze)

| # | Decision | Proposed default (my lean) | Sign-off |
|---|---|---|---|
| 1 | Need Gymnasium MuJoCo results for parity, or is Brax/MJX-only OK? | Brax/MJX-only for main results | ☐ |
| 2 | Authoritative env backend | **MJX-backed Brax** (named explicitly, see §1) | ☐ |
| 3 | Primary scientific claim | Final return + seed stability (sample-eff. secondary) | ☐ |
| 4 | One-time external calibration on Gymnasium (SB3/CleanRL)? | Yes — calibration only, not in the JAX loop | ☐ |
| 5 | Freeze the whole protocol before large sweeps? | Yes | ☐ |

---

## 1. Environment suite (authoritative)

- **Backend:** Brax `backend="mjx"` (MuJoCo physics compiled to XLA). This is what the
  repo is built on (`src/env/mujoco.py`). We report it as **"Brax/MJX"**, never as
  bare "MuJoCo" — rewards, termination, horizon, and action scaling are **not**
  identical to Gymnasium `*-v5` and we will not claim they are.
- **Tasks (core):** `halfcheetah`, `hopper`, `walker2d`, `ant`.
  **Stretch (compute permitting):** `humanoid`.
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

- **10 seeds** per (task, algorithm); **5 minimum** for expensive tasks.
- **Identical seed set across algorithms.**
- Report **IQM + 95% stratified bootstrap CIs** and **probability of improvement**
  (rliable / Agarwal et al. 2021). No bare mean±std as the headline.
- Figures: per-task learning curves (mean ± CI band), aggregate performance profile,
  final-score table, sample-efficiency-to-threshold. Optional: wall-clock plot.

## 8. Diagnostics (logged for every run, regardless of headline claim)

KL per update, entropy, gradient norm, value explained-variance, advantage spread,
episode length, termination frequency, policy σ. For TRPO also: line-search success
rate, realized KL vs `target_kl`, surrogate improvement. For PPO: clip fraction,
approx_kl.

## 9. Logging artifacts (per seed)

One raw CSV/JSON per seed · full config snapshot · git commit hash · dependency
lockfile · hardware metadata. Stored under `results/<suite>/<task>_<algo>_seed<k>/`.

## 10. Pre-sweep gate (must pass before large runs)

- [ ] Unit tests pass: `compute_gae` vs hand-computed trajectory; TRPO Fisher-vector
      product vs explicit dense Fisher on a tiny policy; line-search KL-respect.
- [ ] PPO reaches plausible regime on all core tasks (sanity, not SOTA).
- [ ] (If decision #4 = yes) external SB3/CleanRL PPO on Gymnasium confirms our
      PPO/TRPO are order-of-magnitude correct.
- [ ] TRPO line-search success ≈ 1.0 and realized KL ≈ target on a smoke run.
- [ ] Protocol §1–§8 signed off (§0 table all ☑) and commit hash recorded above.

---

### Validity scope (state in the paper)
> Our harness reproduces standard PPO/TRPO/REINFORCE behavior on Brax/MJX
> continuous control; all methods use identical environment-step budgets, network
> architectures, normalization, evaluation procedures, and seed sets; results are
> reported with uncertainty-aware aggregate metrics (IQM, 95% bootstrap CIs).
> Environments are Brax/MJX and are **not** claimed to be numerically identical to
> Gymnasium MuJoCo `v5`.
