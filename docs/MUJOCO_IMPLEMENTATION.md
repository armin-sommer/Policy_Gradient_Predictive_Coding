# MuJoCo Backprop Benchmark — Full Implementation Reference

PPO / TRPO / REINFORCE continuous-control baselines on **Brax + MJX**, in JAX/Flax.
This document covers the math, architecture, initialization, and every implementation
detail, mapped to the actual code. Files:

```
src/env/mujoco.py              MujocoVecEnv / MujocoEvalEnv (Brax wrapper, obs/reward norm)
src/networks/networks.py       MLP + SOTA (orthogonal/tanh/state-indep log_std) builders
src/networks/distributions.py  NormalTanhDistribution (Gaussian + tanh squashing)
src/backprop_algorithms/common.py   make_networks, compute_gae, Transition, TrainingState
src/backprop_algorithms/ppo.py      PPO loss + training loop
src/backprop_algorithms/trpo.py     TRPO natural gradient + line search
configs/mujoco_*.yaml          per-task hyperparameters
scripts/run_train.py           YAML -> Config dispatch
scripts/run_mujoco_benchmark.py  multi-seed sweep runner
```

Notation: `s` state/observation, `a` action, `r` reward, `π_θ` policy with params θ,
`V_φ` value with params φ, `γ` discount, `λ` GAE parameter, `A_t` advantage.

---

## 0. The big picture / data flow

```
Brax MJX env (GPU)  --obs-->  normalize_obs  --s-->  π_θ (actor MLP)  --logits-->
   NormalTanh dist  --raw u-->  a = tanh(u)  --step-->  Brax env  --r, done-->
   normalize_reward  -->  Transition buffer  -->  GAE (advantages, value targets)
   -->  {PPO clipped SGD | TRPO natural gradient}  -->  updated θ, φ
```

Everything runs in one JAX program on the GPU. The rollout is **eager** (Python loop,
NumPy boundary each step); the **update** is `jax.pmap`-compiled. Observations are
state vectors (≈4–27 floats), so the encoder is a plain **MLP — no CNN**.

---

## 1. Environment layer (`src/env/mujoco.py`)

### 1.1 Brax/MJX construction
```python
self._env = brax_envs.create(
    cfg.env_name, episode_length=cfg.episode_length, action_repeat=1,
    auto_reset=True, batch_size=cfg.num_envs, backend="mjx")
```
- **MJX backend** = MuJoCo's own physics compiled to XLA → real MuJoCo dynamics on GPU.
- `batch_size=num_envs` → `VmapWrapper` vmaps the whole sim over N envs; `reset`/`step`
  are `jax.jit`-compiled.
- `auto_reset=True` → on episode end the env resets that lane automatically.

### 1.2 Termination vs truncation (critical for correct advantages)
Brax exposes two signals:
- **`done`** — episode ended (either the robot fell **or** hit the time limit).
- **`info["truncation"]`** — 1 iff `done` was due to the **time limit** (not a real
  terminal state).

From these the algorithms reconstruct:
```python
discount     = 1 - done                       # stored per transition
truncation   = info["truncation"]
termination  = (1 - discount) * (1 - truncation)   # = real terminal (fell), not time-limit
```
**Why it matters:** at a *truncation* you must **bootstrap** `V(s_{t+1})` (the episode
was artificially cut); at a *termination* you must **not** (the future return is genuinely
0). GAE below uses `termination` to gate bootstrapping and `truncation` to mask deltas.

### 1.3 Observation normalization (Welford running mean/std)
TRPO's natural gradient is scale-sensitive; unnormalized obs make the Fisher
ill-conditioned. We maintain a running mean μ and variance σ² over all observations seen:

Parallel (batch) Welford update for a batch of n obs with batch mean m_b, var v_b:
```
δ      = m_b − μ
μ      ← μ + δ · n / (count + n)
M2     ← σ²·count + v_b·n + δ²·count·n/(count+n)
σ²     ← M2 / (count + n)
count  ← count + n
ŝ      = clip( (s − μ) / sqrt(σ² + 1e-8), −10, 10 )
```
**Implementation rule (important):** `normalize_obs(obs, update=True)` mutates the stats
on every call. The rollout therefore normalizes each raw obs **exactly once** and reuses
the result for both the policy input and the stored `Transition.observation`. If you
normalize the same obs twice (stats move between calls), the stored obs ≠ the obs the
policy acted on → the on-policy ratio ρ ≠ 1 → TRPO's line search rejects steps.
**Eval uses `update=False`** so evaluation never shifts the training statistics.

### 1.4 Reward normalization (return-based, SOTA, training-only)
Standard CleanRL/SB3 reward scaling — divide rewards by the running std of the
**discounted return** (no mean subtraction):
```
ret_e   ← γ · ret_e + r_e              # per-env running discounted return
σ²_ret  ← Welford update with ret       # running variance of returns
r̂       = clip( r / sqrt(σ²_ret + 1e-8), −10, 10 )
ret_e   ← 0  where done                 # reset at episode boundaries
```
Applied **only in the training rollout** (`Config.normalize_rewards`), **never on eval**
— `eval/mean_score` must stay in raw reward units to match the literature scale. This is
why training `value_loss` is ~O(0.1) instead of O(60): values live in normalized units.

---

## 2. Network architecture (`src/networks/networks.py`)

### 2.1 Topology
**Separate (unshared) actor and critic MLPs** — the standard for state-based MuJoCo
(CleanRL/SB3/Spinning Up). Sharing a trunk lets the value loss gradient destabilize the
policy; with small MLPs there's no compute saving from sharing.

### 2.2 The SOTA modules (CleanRL/Engstrom spec)
```python
class SOTAPolicyMLP(nn.Module):              # actor
    hidden_layer_sizes; action_size; activation = tanh
    def __call__(x):
        for h in hidden: x = tanh(Dense(h, kernel_init=orthogonal(√2), bias=0)(x))
        mean    = Dense(action_size, kernel_init=orthogonal(0.01), bias=0)(x)
        log_std = self.param('log_std', zeros, (action_size,))   # STATE-INDEPENDENT
        return concat([mean, broadcast(log_std)], axis=-1)       # -> param_size 2·act_dim

class SOTAValueMLP(nn.Module):               # critic
    def __call__(x):
        for h in hidden: x = tanh(Dense(h, kernel_init=orthogonal(√2), bias=0)(x))
        return squeeze(Dense(1, kernel_init=orthogonal(1.0), bias=0)(x))
```
- **Default size:** 2 hidden × 64, **tanh**. HalfCheetah uses **2×256** (it needs
  capacity; 2×64 caps PPO ~1800).
- The policy outputs `[mean, log_std]` so `param_size = 2·act_dim` and the downstream
  `NormalTanhDistribution` is unchanged. The `log_std` is a **free parameter**, not a
  function of the state.

### 2.3 The default (non-SOTA / Brax-style) path still exists
`make_policy_network` / `make_value_network` build `lecun_uniform` + ReLU MLPs with a
state-dependent scale head. Selected when `sota_init=False`. The CNN path
(`make_cnn_*`, NatureCNN) is for Procgen pixels and ignores `sota_init`.

---

## 3. Initialization — the math and the why (the crux)

| Layer | Scheme | Gain | Bias |
|---|---|---|---|
| Hidden | Orthogonal | **√2** (for tanh) | 0 |
| **Actor output (mean)** | Orthogonal | **0.01** | 0 |
| Critic output | Orthogonal | **1.0** | 0 |
| `log_std` | constant | — | **0** → σ = exp(0) = 1 |

### 3.1 Orthogonal initialization — what it is
A weight matrix `W ∈ R^{out×in}` is initialized so its rows (or columns) are
**orthonormal**, then scaled by `gain`: effectively `W = gain · Q` where `Q` has
`QᵀQ = I`. Consequence: `‖Wx‖ = gain·‖x‖` along the principal directions — the layer
**preserves the norm** of activations (forward) and gradients (backward). This prevents
the vanishing/exploding-gradient pathology across the many minibatch updates PPO/TRPO do
per rollout. (Contrast: Glorot/Lecun scale by `1/√fan` for variance, not norm
preservation.)

### 3.2 Why gain √2 for tanh hidden layers
tanh has slope 1 at 0 but compresses for large inputs; the `√2` gain (the same constant
as He init) compensates for the average attenuation so that activation variance is
roughly preserved layer-to-layer at init.

### 3.3 Why gain 0.01 on the actor output (the single most important detail)
With output weights ~0.01 and bias 0, the **initial mean action ≈ 0** for any state.
Combined with `log_std = 0` (σ = 1), the initial policy is a **centered, symmetric
Gaussian** of moderate spread. Physically: the robot **stands roughly still and explores
small symmetric torques** to discover the dynamics, rather than immediately slamming
joints to the limits and self-terminating. Engstrom et al. (2020) showed this single
choice accounts for a large fraction of PPO/TRPO's apparent robustness.

### 3.4 Why gain 1.0 on the critic output
The value head is a regression onto returns of arbitrary scale, so it gets a neutral
gain of 1.0 (no shrinkage).

### 3.5 State-independent `log_std` init 0
The action std is a **single learnable vector** `log σ` (one per action dim), independent
of the state, initialized to 0 → σ = 1. The policy **mean** is state-dependent (the MLP),
the **spread** is a global parameter that anneals as training sharpens. This is the
SB3/Spinning Up convention and is more stable than a state-dependent std head for MuJoCo.
(See §4.2 for `exp` vs `softplus`.)

Verify with `python scripts/verify_sota_init.py` — it asserts actor-output std ≈ 0.01,
`log_std == 0`, initial action ≈ 0, initial σ ≈ 1.

---

## 4. Action distribution (`src/networks/distributions.py`)

### 4.1 Gaussian + tanh squashing
Actions must lie in `[−1, 1]` (Brax action bounds). The policy is a diagonal Gaussian on a
**pre-tanh** latent `u`, squashed through tanh:
```
u ~ N(μ(s), σ)              # raw_action, what we store
a = tanh(u)                 # postprocessed action sent to the env
```
We operate on `u` (not `a`) for log-probs to avoid tanh saturation making `log_prob`
numerically impossible.

### 4.2 The std parametrization
```python
loc, scale = split(params, 2)        # params = [mean, log_std]
scale = exp(log_std)                 # SOTA path (exp_std=True): log_std=0 -> σ=1
      = softplus(scale) + min_std    # default path
```
`exp` is used with the SOTA spec so the init `log_std = 0` gives exactly σ = 1.

### 4.3 Log-probability with the tanh Jacobian correction
Because `a = tanh(u)`, the change-of-variables adds the log-determinant of the tanh
Jacobian:
```
log π(a | s) = log N(u; μ, σ)  −  Σ_i log(1 − tanh(u_i)²)
```
(the second term = `−Σ log(1 − a²)`, the `TanhBijector` log-det). This is the
`parametric_action_distribution.log_prob(logits, raw_action)` call. The stored
`behaviour log_prob` is computed identically at rollout time, so on-policy ρ = 1.

### 4.4 Entropy, mode, KL
- **Entropy** `H = Σ_i [½ log(2πe σ_i²)]` (plus the tanh correction, estimated with a
  sample) — note for σ < 1 the **differential entropy is negative**; this is normal as the
  policy sharpens on an easy task and is **not** an error.
- **Mode** (deterministic eval): `a = tanh(μ)`.
- **KL** between two diagonal Gaussians (used by TRPO):
  `KL(π_old‖π) = Σ_i [ log(σ_i/σ_{old,i}) + (σ_{old,i}² + (μ_{old,i} − μ_i)²)/(2σ_i²) − ½ ]`.

---

## 5. Advantage estimation — GAE (`common.py: compute_gae`)

Generalized Advantage Estimation with explicit termination/truncation handling.

### 5.1 The math
TD residual (δ), with bootstrapping gated by termination and masked by truncation:
```
δ_t   = r_t + γ (1 − term_t) V(s_{t+1}) − V(s_t),   then  δ_t ·= (1 − trunc_t)
A_t   = δ_t + γ λ (1 − term_t)(1 − trunc_t) A_{t+1}        (computed by a reverse scan)
V^tgt_t = A_t + V(s_t)         (the TD(λ) value target, called `vs` in code)
```
- At a **termination** (`term=1`): no bootstrap (`1−term=0`) → `δ = r − V`.
- At a **truncation** (`trunc=1`): the delta is masked to 0 for that step's contribution
  to the recursion (the cut is artificial; the last `V(s_{t+1})` bootstrap still applies
  via the un-masked delta term).
- `bootstrap_value = V(next_observation[-1])` provides `V(s_{T+1})`.

### 5.2 Two outputs
- **`advantages` A_t** → policy objective (PPO surrogate / TRPO surrogate).
- **`vs` (value targets)** → critic regression target.
Both are `stop_gradient`-ed.

### 5.3 Advantage normalization
Per update batch: `A ← (A − mean(A)) / (std(A) + 1e-8)`. Reduces gradient variance; note
this makes the raw surrogate value ≈ 0 at θ_old (since mean A ≈ 0), which is expected.

---

## 6. PPO (`src/backprop_algorithms/ppo.py`)

### 6.1 Objective (the math)
Probability ratio and clipped surrogate:
```
ρ_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) = exp(logπ_new − logπ_old)
L^CLIP(θ) = E_t[ min( ρ_t A_t,  clip(ρ_t, 1−ε, 1+ε) A_t ) ]
```
Total loss minimized:
```
L = −L^CLIP  +  c_v · ½ E[(V^tgt − V_φ)²]  −  c_e · H
```
with `ε = clip_coef (0.2)`, `c_v = vf_coef (0.5)`, `c_e = ent_coef (0.0)`.
Diagnostic: `approx_kl = E[(ρ−1) − log ρ]` (≥0, second-order estimate of KL).

### 6.2 Update structure (why PPO is sample-efficient)
Per rollout of `B` transitions: shuffle, split into `num_minibatches (32)` minibatches,
do `update_epochs (4)` passes → **4 × 32 = 128 Adam steps per rollout**. Over 1M steps
(≈244 rollouts) PPO makes ≈ 31,000 parameter updates. This is the structural reason PPO
converges far faster than TRPO (one update/rollout) at equal environment budget.

### 6.3 Optimizer
`optax.chain(clip_by_global_norm(0.5), adam(lr, eps=1e-5))`, lr 3e-4. `eps=1e-5` is the
PPO-MuJoCo convention (vs Adam default 1e-8).

---

## 7. TRPO (`src/backprop_algorithms/trpo.py`)

### 7.1 The constrained problem (the math)
Maximize the surrogate subject to a hard **trust region** on the policy change:
```
maximize_θ   L(θ) = E_t[ ρ_t(θ) A_t ]
subject to   E_t[ KL( π_θ_old ‖ π_θ ) ] ≤ δ        (δ = target_kl = 0.01)
```

### 7.2 Solution via natural gradient + conjugate gradient
Linearize the objective and quadratically approximate the KL around θ_old:
```
L(θ) ≈ gᵀ(θ−θ_old),     KL ≈ ½ (θ−θ_old)ᵀ F (θ−θ_old)
```
- `g = ∇_θ L|_{θ_old}` — the policy gradient (`policy_objective_grad`).
- `F` — the **Fisher Information Matrix** = Hessian of the KL = `E[∇log π ∇log πᵀ]`.

The constrained optimum points along the **natural gradient** `F⁻¹g`, solved without
forming F via **conjugate gradient** using only **Fisher-vector products** `Fv`:
```
Fv = ∇_θ( (∇_θ KL)·v )                       # Hessian-vector product of KL
   + cg_damping · v                          # (0.1) for numerical conditioning
x  = CG(F, g, max_iter=cg_max_iterations=10) # search direction ≈ F⁻¹g
```
Code mapping: `jacobian_vector_product` = `(∇KL)·v`; `hessian_vector_product` =
`∇((∇KL)·v) + λ_damp v = Fv`; `jax.scipy.sparse.linalg.cg` solves `Fx=g`.

### 7.3 Step size from the trust region
The largest step along `x` that satisfies the quadratic KL constraint:
```
β = sqrt( 2δ / (xᵀ F x) )       # max step size; xᵀFx via one more Fisher-vector product
Δθ = β · x
```

### 7.4 Backtracking line search (exact constraint enforcement)
The quadratic KL is only approximate, so backtrack until both the **true** KL is within
the region **and** the surrogate actually improved:
```
for k = 0,1,…,line_search_max_iter(10):
    θ_try = θ_old + (shrink^k · β) x          # shrink = 0.8
    accept if  KL(π_old‖π_try) ≤ δ  AND  L(θ_try) ≥ L(θ_old)
θ = θ_try if accepted else θ_old              # reject → no move this rollout
```
`line_search_success` logs 1 on accept. (A run-long string of 0s = the ρ≠1 obs-norm bug,
not a TRPO failure — see §1.3.)

### 7.5 Value network
Trained **separately** by Adam on the GAE targets (`compute_value_loss`,
`update_epochs` minibatch passes) — only the **policy** uses the natural gradient.

### 7.6 Runner overrides (full-batch Fisher)
`run_mujoco_benchmark.py` sets TRPO to `num_minibatches=1, batch_size=256, eval_every=20`
so the Fisher is estimated on the full rollout (one natural-gradient step per rollout, as
TRPO requires). **Consequence:** ~1 policy update/rollout vs PPO's 128 → TRPO is
**update-starved** at 1M steps and needs **5–10M** to converge on locomotion.

---

## 8. REINFORCE (the baseline)
Vanilla policy gradient `∇_θ J = E[ ∇_θ log π_θ(a|s) · A_t ]` with the GAE baseline.
**Intentionally left un-SOTA** (the weak floor), forced to `num_envs=1` (its rollout
requires single-env), `num_minibatches=1`. Not in the SOTA sweep.

---

## 9. Hyperparameter reference

| Param | PPO | TRPO | Notes |
|---|---|---|---|
| net | 2×64 (HC 2×256) tanh | same | shared per task |
| init | orthogonal √2 / 0.01 / 1.0 | same | §3 |
| log_std | state-indep, init 0, exp | same | σ=1 at init |
| optimizer | Adam lr 3e-4 eps 1e-5 | Adam (value only) | grad-clip 0.5 |
| γ, λ | 0.99, 0.95 | 0.99, 0.95 | GAE |
| clip ε | 0.2 | — | |
| target_kl | — (diag only) | 0.01 | trust region |
| cg_damping / cg_iters | — | 0.1 / 10 | Fisher solve |
| line search | — | 10 iters, shrink 0.8 | |
| epochs × minibatches | 4 × 32 | 1 × 1 (full batch) | update count |
| num_envs | 256 | 256 | REINFORCE = 1 |
| obs norm / reward norm | yes / yes | yes / yes | train only |
| eval | deterministic (tanh μ), 10 eps, raw reward | same | |

---

## 10. Training loop & infra
- **Rollout:** Python loop over `unroll_length`, NumPy boundary each step; collect
  `Transition(obs, action, reward, discount, next_obs, extras={policy_extras, state_extras})`.
- **Update:** reshape to `(devices, minibatch, …)`, `jax.pmap(learn)` over devices,
  `jax.lax.pmean` grads across the `'i'` axis.
- **Eval:** every `eval_every` updates, run the **deterministic** policy (mode = tanh μ)
  for `num_eval_episodes (10)` on a separate `num_envs=1` env; report **raw** returns.
- **Logging:** `logging.info({metric dict})` → wandb handler (`run_train._WandbHandler`,
  logs at the env-step) + stdout/file. Keys: `training/*`, `eval/mean_score`,
  `final_eval/*` (same convention as the bandit benchmark).

---

## 11. Convergence behavior (empirical, 1M steps, SOTA)
- **PPO:** hopper ≈ 3260, walker2d ≈ 3550 → **converged** at literature level. halfcheetah
  ≈ 1768 (capacity-limited at 2×64 → fixed by 2×256) and still climbing.
- **TRPO:** plateaus low (hopper ~240, walker2d ~450; halfcheetah oscillates) — **not a
  bug** (245/245 line searches succeed). Cause: update-starvation (§7.6) + the "stay
  upright" local optimum (eval episode lengths stuck ~110/230 of 1000). Fix: 5–10M steps.

---

## 12. Reproduction

### Known-good install stack (RunPod, non-Blackwell GPU)
```
jax==0.4.38  jaxlib==0.4.38  brax==0.11.0  mujoco==3.2.7  mujoco-mjx==3.2.7
```
All must be pinned together (newer brax forces newer jaxlib, which breaks the repo's pmap).

### Run
```bash
python scripts/run_mujoco_benchmark.py --env halfcheetah --algos ppo trpo \
    --seeds 1 2 3 --total-steps 1000000 --wandb-project mujoco-pcpg
python scripts/summarize_mujoco.py        # -> summary_all.csv + per-env mean±SEM curves
python scripts/verify_sota_init.py        # asserts the §3 init
```

---

## 13. Key references
- Schulman et al. 2015, *Trust Region Policy Optimization* (TRPO).
- Schulman et al. 2017, *Proximal Policy Optimization* (PPO); 2016, *GAE*.
- Engstrom et al. 2020, *Implementation Matters in Deep Policy Gradients* (the init/normalization study).
- Huang et al., *The 37 Implementation Details of PPO*.
- Freeman et al. 2021, *Brax*; DeepMind *MuJoCo Playground* (MJX).
- Reference code: CleanRL `ppo_continuous_action.py`; base adapted from PolicyGradientsJax.
