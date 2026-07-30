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

## 0. Prerequisite — push your work

A pod clones from GitHub, so anything you want to run must be pushed first. The
branch `feature/mujoco-halfcheetah-pcpg` **is on origin**; just make sure your
latest commits are too:

```bash
git push origin feature/mujoco-halfcheetah-pcpg
```

If you edited code locally and the pod behaves like the old version, this is why.

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
are measuring. Judge against the matched backprop baselines in this repo
(`results/mujoco/`, same [64,64] net / 256 envs / 1M steps / 3 seeds):

| | best per seed | mean |
|---|---|---|
| PPO | 1759 / 4360 / 2590 | 2903 |
| TRPO | 1276 / 1408 / 1330 | 1338 |
| best PCPG | 849 / 806 / 932 | 862 |

So PCPG currently sits ~3x below PPO and ~1.5x below TRPO at that budget.

---

## 4a. Set this once per pod session — it prevents `git pull` from breaking

`results/` is tracked in git (the experiment log embeds plots from it). If a pod
writes sweeps into the same paths, the next `git pull` fails with

    error: The following untracked working tree files would be overwritten by merge

because git wants to write files the pod already generated. Avoid it entirely by
sending pod output to a gitignored root:

```bash
export PCPG_RESULTS_ROOT=results_pod
```

All the §4b sweeps honour it (`results_pod/<experiment>/...`), and `results_pod/`
is in `.gitignore`, so the pod's tree never collides with the tracked one. Then
transfer and commit from your Mac:

```bash
# on the POD
runpodctl send results_pod/<experiment>
# on your MAC -- lands in the tracked results/ tree
cd results && runpodctl receive <code>
```

**If you already hit the collision**, the data is on GitHub and/or your Mac, so it
is safe to clear the pod's copy:

```bash
git checkout -- . && rm -rf results/<colliding-dir> && git pull
```

---

## 4b. The current experiment workflow (what the recent results used)

§2–§4 describe the original tier sweep. Everything in
[PCPG_EXPERIMENT_LOG.md](PCPG_EXPERIMENT_LOG.md) was produced with a different,
config-per-cell workflow — one directory per condition, resolved config + logs
stored together, then a shared analyzer.

**Run a comparison sweep** (each writes `results/<dir>/<config>/seed_N.log`
alongside the exact `config.yaml` and a `meta.json` with the git commit):

```bash
# the two open knobs: sigma floor (Adam mode) and critic LR (SGD mode)
python scripts/run_pcpg_knob_fill.py --seeds 1 2 3 --skip-complete

# gradient clipping, thresholds taken from the measured grad-norm tail
python scripts/run_gradclip_probe.py --stage auto --seeds 1 2 3

# target route vs likelihood-energy route (tests "does PC supply the geometry?")
python scripts/run_likelihood_vs_natural.py --seeds 1 2 3 --skip-complete
```

**Analyze any of them** — this is the analyzer that defines `final`, `best`,
`AUC`, `collapse` (see §1b of the experiment log for the exact definitions):

```bash
python scripts/analyze_pcpg_logs.py --results-dir results/<dir>
#   -> per_run.csv, SUMMARY.md, learning_curve.png, diagnostic_plots.png

python scripts/plot_collapse_anatomy.py --results-dir results/<dir>
#   -> collapse_anatomy_<config>.png per config: seeds overlaid across
#      eval return, train reward, value EV, KL, entropy, saturation
```

**Unattended overnight**, with auto-stop so it does not bill idle GPU:

```bash
tmux new -s overnight
export RUNPOD_API_KEY=<key>          # see §6
bash scripts/run_overnight_batch.sh  # gradclip + knob fill, ~2.5 h, then stops the pod
```

**Generate a new config cell** rather than hand-editing YAML:

```bash
python scripts/gen_benchmark_config.py --algo pc_actor_critic --tier bench \
    --opt sgd --act tanh --ts 1.0 --max-t1 20 --lr 0.03 --natural-target
# -> configs/benchmark/..._sgd_tanh_ts10_bench_lr003_mt20_nat.yaml
```

Naming gotcha: **`lr003` = 0.03**, `lr0003` = 0.003. The SGD baseline that learns
needs `--lr 0.03`.

**Checks that need no GPU** (useful before committing a long run):

```bash
PYTHONPATH=src python scripts/test_likelihood_energy.py      # likelihood step sanity
python scripts/probe_pc_sample_weighting.py                  # per-sample weighting probe
```

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

**Stop and Terminate are not the same thing:**

| action | GPU | `/workspace` | cost |
|---|---|---|---|
| **Stop** | released | **survives** | storage only (cents/day) |
| **Terminate** | released | **destroyed permanently** | none |

So **Stop is safe** — your results stay on the volume and are there when you
restart. Only Terminate requires pulling first.

```bash
# on the POD:
runpodctl send results/<dir>
# on your MAC:
runpodctl receive <code>
```

Install runpodctl on the Mac once: `brew install runpod/runpodctl/runpodctl`.

**Auto-stop.** `scripts/run_overnight_batch.sh` stops the pod itself when the runs
finish, so an overnight batch does not bill idle GPU until you notice. It needs an
API key in the shell you launch from:

```bash
export RUNPOD_API_KEY=<key from runpod.io/console/user/settings>
runpodctl get pod $RUNPOD_POD_ID     # must not say Unauthorized
```

The script preflights this and warns loudly at startup if it would fail.

---

## 7. Quick reference

- **Repo / branch:** `armin-sommer/Policy_Gradient_Predictive_Coding` ·
  `feature/mujoco-halfcheetah-pcpg` (push it first — §0)
- **GPU:** A100 / H100 / L40S / RTX 4090 · **never** Blackwell · **CUDA 12.4** template
- **JAX:** pinned **0.4.38** — never `pip install -U jax`
- **One command (tier sweep):** `bash scripts/run_pcpg_runpod.sh [ENV] [TIERS] [SEEDS] [extra…]`
- **Comparison sweeps (§4b):** `run_pcpg_knob_fill.py`, `run_gradclip_probe.py`,
  `run_likelihood_vs_natural.py`; analyze with `analyze_pcpg_logs.py` +
  `plot_collapse_anatomy.py`
- **Overnight, self-stopping:** `bash scripts/run_overnight_batch.sh` (needs
  `RUNPOD_API_KEY`)
- **Results:** `results/mujoco_pcpg_halfcheetah/` for the tier sweep;
  `results/<experiment>/<config>/seed_N.log` for the comparison sweeps
- **Stop ≠ Terminate:** Stop keeps `/workspace`; only Terminate needs a pull first
- **`export PCPG_RESULTS_ROOT=results_pod`** once per pod session (§4a) — keeps
  `git pull` from colliding with the tracked `results/` tree
- **Resume a killed sweep:** re-run the same command — `--skip-complete` skips
  finished runs.
- **Troubleshooting:** same table as [RUNPOD.md §7](../RUNPOD.md) (CUDA/GPU/JAX
  errors are identical).
