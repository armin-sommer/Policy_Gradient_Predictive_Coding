# Procgen MVP — backprop algorithms only

Goal: get a **backprop** policy-gradient algorithm (PPO) training end-to-end on
Procgen with the CNN policy, and *verify* the Procgen-specific code paths that
the bandit study never exercised. **PC algorithms are explicitly out of scope.**

## Key finding: the code already exists

The backprop + CNN Procgen path is already implemented:

- `src/env/env.py` — `ProcgenVecEnv` / `ProcgenEvalEnv` (gym3 wrapper).
- `src/env/__init__.py` — `make_vec_env` dispatches non-`bandit` names to Procgen.
- `src/networks/networks.py` — `NatureCNN` + `make_cnn_policy_network` / `make_cnn_value_network`.
- `src/backprop_algorithms/{ppo,cleanba_ppo,trpo,reinforce}.py` — all have a `use_cnn` path.
- `scripts/run_train.py` — YAML → Config dispatch, sets `use_cnn = "cnn" in network`.

So this MVP is **verification-driven**, not a build: confirm the existing path is
correct, fix only what is demonstrably broken, and add a clean config + entrypoint.

## What the bandit already verified (the overlap — trust it)

Bandit runs exercise the backprop algos with `use_cnn=False`, MLP, 1-step episodes:
training loop, `pmap`, REINFORCE/PPO/TRPO update math, MLP networks, distributions,
the NumPy↔JAX boundary, and the eval drain protocol.

## What the bandit did NOT verify (the risk surface — verify these)

| # | Surface | Why the bandit can't catch it |
|---|---------|-------------------------------|
| 🔴 1 | `ProcgenVecEnv` done/reward alignment (gym3 auto-reset) | bandit `done=1` every step |
| 🔴 2 | Multi-step GAE accumulation + truncation/termination | bandit episodes are 1-step → GAE collapses to single-step |
| 🟡 3 | `NatureCNN` forward (uint8 (H,W,C) → logits/value) | bandit uses MLP (`use_cnn=False`) |
| 🟡 4 | `run_train.py` YAML plumbing + `use_cnn` from network name | bandit uses `run_bandit_comparison.py` |

## Two-environment reality

- **This Mac (dev):** Procgen has **no installable wheel** (`procgen-mirror`,
  macOS arm64 / py3.11). Can verify only the pure-JAX surfaces (GAE 🔴2, CNN 🟡3).
- **Linux GPU box (target):** per `Dockerfile` (CUDA 12.4, py3.11, RunPod). Runs the
  env probe (🔴1), the YAML plumbing (🟡4), and the actual training.

## Plan (each step has a verifiable success criterion)

1. **GAE multi-step (🔴2)** — pure JAX, runs on Mac.
   `scripts/verify_procgen_backprop.py --check gae`.
   *Verify:* advantages equal the independent textbook identity
   `A_t = (discounted return-to-go, with bootstrap) − V_t` for λ=1, both a
   terminating episode and a continuing/bootstrapped rollout. **PASS = allclose.**

2. **CNN forward (🟡3)** — pure JAX, runs on Mac.
   `scripts/verify_procgen_backprop.py --check cnn`.
   *Verify:* `make_networks(use_cnn=True, observation_size=(64,64,3))` builds; a
   uint8 `(N,64,64,3)` batch yields policy logits `(N,15)` and value `(N,)`; the
   jitted inference fn samples actions in `[0,15)`. **PASS = shapes + range hold.**

3. **Procgen env probe (🔴1)** — needs Procgen, run on the box.
   `scripts/verify_procgen_backprop.py --check env` (auto-skips if Procgen absent).
   *Verify:* obs `(N,64,64,3)` uint8; `done ∈ {0,1}`; rewards finite; the terminal
   reward is attributed to the step where `done` flips (the gym3 off-by-one check);
   `ProcgenEvalEnv.evaluate()` returns one finite entry per completed episode.
   **PASS = all asserts hold.** *If it fails, the fix lives in `env.py:62-77`.*

4. **End-to-end smoke (🟡4 + integration)** — on the box.
   `python scripts/run_train.py --config configs/procgen_coinrun.yaml \
        --overrides train.total_steps=200000`.
   *Verify:* trains without crashing; `eval/mean_score` logged and trends upward.

## Known caveats / likely follow-ups

- **JAX version:** repo recommends JAX 0.4.38; the dev venv has 0.5.2. The backprop
  algos use `jax.pmap` heavily and the README warns pmap "breaks on newer JAX."
  If step 4 errors inside `pmap`/`device_put_replicated`, downgrade JAX on the box.
- **Network name is cosmetic:** `run_train.py` only checks for the substring
  `"cnn"`; the architecture is always `NatureCNN` regardless of `impala_cnn` in YAML.
- **`pip install -e .` fails** (hatchling direct-reference to the `jpc` git dep).
  Not needed for backprop — scripts put `src/` on the path themselves. (Out of scope.)

## Out of scope

PC algorithms (`pc_reinforce`, `pc_actor_critic`) — their image/conv front-end and
multi-step PC target are a separate effort.
