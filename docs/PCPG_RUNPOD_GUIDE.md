# Running Gaussian PCPG on RunPod (HalfCheetah)

Copy-paste guide for training **Gaussian PCPG** (`pc_actor_critic`, `pc_reinforce`)
on Brax/MJX HalfCheetah on a RunPod GPU, with results summarized to CSV + PNG and
pulled back to your Mac. It reuses the existing scripts — the only thing you run is
one wrapper.

For the *why* behind the tuning stages, see
[PCPG_TUNING_METHODOLOGY.md](PCPG_TUNING_METHODOLOGY.md). This file is the *how to
run it*. The backprop-baseline guide is [RUNPOD.md](../RUNPOD.md); the two rules
below are the same.

> **The two rules that avoid 90% of the pain:**
> 1. **GPU:** any **non-Blackwell** card (A100, H100, L40S, RTX 4090). **Never**
>    RTX 5090 / PRO 6000 / B200 / B300 — `jax 0.4.38` can't compile for them.
> 2. **Template:** a **CUDA 12.4** image (e.g. "PyTorch 2.4" / `cu124`). **Not**
>    CUDA 12.8 — it segfaults with the pinned JAX.

---

## 0. Prerequisite — push this branch

A pod clones from GitHub, so `feature/mujoco-halfcheetah-pcpg` must exist on
**origin** first (it currently only exists locally). From your Mac:

```bash
git push -u origin feature/mujoco-halfcheetah-pcpg
```

---

## 1. Deploy the pod

RunPod → **Pods → Deploy** → **A100 SXM** (or H100/L40S/4090, *non-Blackwell*) →
a **CUDA 12.4** template (**PyTorch 2.4** / `cu124`) → Deploy On-Demand → wait for
**Running** → **Connect → Start Web Terminal**. (Deploy fails with a pull-rate
limit? Just retry — it lands on a different node.)

---

## 2. Clone + the one command

```bash
cd /workspace
apt-get update && apt-get install -y libgl1 libglib2.0-0 tmux

git clone https://github.com/armin-sommer/Policy_Gradient_Predictive_Coding.git PCPG
cd PCPG
git checkout feature/mujoco-halfcheetah-pcpg

# setup -> verify GPU/env -> sweep -> summarize (CSV+PNG) -> print pull command
bash scripts/run_pcpg_runpod.sh
```

That single wrapper ([scripts/run_pcpg_runpod.sh](../scripts/run_pcpg_runpod.sh))
runs, in order: `setup_runpod_pcpg.sh` (installs JAX 0.4.38 / Brax / MJX / jpc and
the CUDA 12.4 pins — skipped on reruns), `verify_mujoco.py --check env`, the PCPG
sweep, and `summarize_mujoco.py`. Defaults are **halfcheetah, bench tier, seeds
1 2 3**.

**Arguments** — `run_pcpg_runpod.sh [ENV] [TIERS] [SEEDS] [extra sweep args…]`:

```bash
# quick smoke (does it run + not NaN?) — ~1 min after setup
bash scripts/run_pcpg_runpod.sh halfcheetah bench 1 --total-steps 50000 --no-save

# full bench + sota, 3 seeds each
bash scripts/run_pcpg_runpod.sh halfcheetah "bench sota" "1 2 3"
```

Everything after the third positional (e.g. `--total-steps`, `--no-save`,
`--algos pc_actor_critic`) is passed straight to the sweep.

---

## 3. What each piece does (for manual control)

You rarely need these individually, but the wrapper is just glue over them:

```bash
# a) one-time env setup (JAX/Brax/MJX/jpc + CUDA 12.4 pins)
bash scripts/setup_runpod_pcpg.sh

# b) sanity-check the GPU + a Brax env build BEFORE a long run
python scripts/verify_mujoco.py --check env --env halfcheetah   # -> PASS

# c) the sweep: algos x tiers x seeds, resumable, file logs
python scripts/run_mujoco_pcpg_sweep.py --env halfcheetah \
    --tiers bench sota --algos pc_actor_critic pc_reinforce \
    --seeds 1 2 3 --skip-complete
#   logs -> results/mujoco_pcpg_halfcheetah/halfcheetah_{tier}_{algo}_seed{N}.log
#   --skip-complete skips runs whose log already contains TRAINING END (resume)
#   --total-steps N overrides the config budget (for smoke/pilot runs)

# d) summarize -> CSV + PNG curves, saved next to the logs
python scripts/summarize_mujoco.py --results-dir results/mujoco_pcpg_halfcheetah
#   -> summary_all.csv, SUMMARY.md, halfcheetah_curve.png
#   curves are per (tier, algo) series, mean ± SEM over seeds
```

The four configs the sweep uses:

| tier | `pc_actor_critic` | `pc_reinforce` |
|---|---|---|
| bench (64-wide, 256 envs, 1M) | `configs/mujoco_halfcheetah_pc_actor_critic_bench.yaml` | `configs/mujoco_halfcheetah_pc_reinforce_bench.yaml` |
| sota (256-wide, 1024/512 envs, 5M) | `configs/mujoco_halfcheetah_pc_actor_critic_sota.yaml` | `configs/mujoco_halfcheetah_pc_reinforce_sota.yaml` |

---

## 4. Mapping to the tuning methodology

The [methodology](PCPG_TUNING_METHODOLOGY.md) is staged; here is how each early
stage is run *by hand* with the scripts above. **The statistical machinery (gates,
ASHA, AUC, bootstrap) is deliberately not automated yet** — the Appendix A
thresholds aren't chosen, so for now you read the logs directly.

- **Smoke (Stage 0A/1) — does it run and stay finite?**
  ```bash
  bash scripts/run_pcpg_runpod.sh halfcheetah bench 1 --total-steps 50000 --no-save
  grep -E "eval/mean_score|nan|inf" results/mujoco_pcpg_halfcheetah/*seed1.log | tail
  ```
  Green = no NaN/Inf in the log and `eval/mean_score` is moving.

- **Pilot (Stage 0B) — behavior in the realistic region.** Vary `target_scale` /
  `learning_rate` by copying a config (e.g. `cp …bench.yaml …bench_ts0p3.yaml` and
  edit `target_scale`), run a handful at a reduced budget, and inspect the
  diagnostics that the algorithms already log:
  ```bash
  python scripts/run_mujoco_pcpg_sweep.py --env halfcheetah --tiers bench \
      --seeds 1 2 --total-steps 500000
  # collapse signals to watch in the logs (see methodology §2):
  grep -E "diag/frac_std_at_min|diag/mu_target_mag_max|diag/value_explained_var" \
      results/mujoco_pcpg_halfcheetah/*.log
  ```

- **Full sweep** — once a config looks stable:
  ```bash
  bash scripts/run_pcpg_runpod.sh halfcheetah "bench sota" "1 2 3"
  ```

There is **no established PCPG return target** for HalfCheetah — that is what you
are measuring. Judge against the backprop baseline curves (HalfCheetah PPO reaches
~2,500 on the Brax scale; see [RUNPOD.md](../RUNPOD.md)), not an absolute number.

---

## 5. Long runs — detach with tmux

sota runs are 5M steps and take a while. Keep them alive after you disconnect:

```bash
tmux new -s pcpg
bash scripts/run_pcpg_runpod.sh halfcheetah "bench sota" "1 2 3"
# detach: Ctrl+B then D   -> safe to close the browser and walk away
# reattach anytime:  tmux attach -t pcpg
```

Monitoring is file-based (matches the saved workflow: PNGs on the pod, then pull).
Watch progress with `grep "eval/mean_score" results/mujoco_pcpg_halfcheetah/*.log`.
(W&B is optional and not wired into the PCPG sweep; it would need `wandb.mode` set
in the config.)

---

## 6. Pull results, then shut down

**Terminate wipes the pod — pull first.** `summarize_mujoco.py` has already written
the CSV + PNG into the results dir, so send the whole folder:

```bash
# on the POD (the wrapper prints this line for you):
runpodctl send results/mujoco_pcpg_halfcheetah
# on your MAC:
runpodctl receive <code>
```

Install runpodctl on the Mac once: `brew install runpod/runpodctl/runpodctl`.
Then **Pods → Terminate** to stop billing.

---

## 7. Quick reference

- **Repo / branch:** `armin-sommer/Policy_Gradient_Predictive_Coding` ·
  `feature/mujoco-halfcheetah-pcpg` (push it first — §0)
- **GPU:** A100 / H100 / L40S / RTX 4090 · **never** Blackwell · **CUDA 12.4** template
- **JAX:** pinned **0.4.38** — never `pip install -U jax`
- **One command:** `bash scripts/run_pcpg_runpod.sh [ENV] [TIERS] [SEEDS] [extra…]`
- **Results:** `results/mujoco_pcpg_halfcheetah/` (per-run logs + `summary_all.csv`
  + `halfcheetah_curve.png`)
- **Resume a killed sweep:** re-run the same command — `--skip-complete` skips
  finished runs.
- **Troubleshooting:** same table as [RUNPOD.md §7](../RUNPOD.md) (CUDA/GPU/JAX
  errors are identical).
