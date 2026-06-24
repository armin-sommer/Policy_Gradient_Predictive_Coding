# Running the MuJoCo benchmark on RunPod

End-to-end, copy-paste setup for the **MuJoCo backprop baselines** (PPO / TRPO /
REINFORCE via Brax + MJX) on a RunPod GPU. Follow top to bottom.

> **The two rules that avoid 90% of the pain:**
> 1. **GPU:** any **non-Blackwell** card (A100, H100, L40S, RTX 4090). **Never** RTX 5090, RTX PRO 6000, B200, B300 — `jax 0.4.38` can't compile for them.
> 2. **Template:** a **CUDA 12.4** image (e.g. "PyTorch 2.4" / `cu124`). **Not** CUDA 12.8 / `cu128` — it causes segfaults with the pinned JAX.

---

## 1. Deploy the pod

1. RunPod → **Pods → Deploy**.
2. **GPU:** **A100 SXM** (cheap, plenty) — or H100/L40S/RTX 4090. *Avoid Blackwell* (see rule 1).
3. **Template:** a **CUDA 12.4** one (**PyTorch 2.4** / `cu124`). *Avoid 12.8/cu128.*
4. Deploy On-Demand → wait for **Running** → **Connect → Start Web Terminal**.

If deploy fails with `toomanyrequests ... pull rate limit` → just **retry** (lands on a different node), or pick a **RunPod official** template, or change region.

---

## 2. One-time setup (paste this whole block)

```bash
cd /workspace
apt-get update && apt-get install -y libgl1 libglib2.0-0 tmux

git clone https://github.com/armin-sommer/Policy_Gradient_Predictive_Coding.git PCPG
cd PCPG
git checkout feature/mujoco

# JAX 0.4.38 (required — newer JAX breaks the algorithms' pmap), Brax/MJX, YAML.
# --ignore-installed blinker: the OS blinker has no RECORD file and aborts pip otherwise.
pip install --ignore-installed blinker \
    "jax[cuda12]==0.4.38" jaxlib==0.4.38 brax mujoco-mjx pyyaml

# jaxlib 0.4.38 is built for CUDA 12.4; pods pull CUDA 12.8/12.9 wheels, which
# segfault. Force the matching 12.4 libraries. (Breaks torch's CUDA — unused here.)
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12==12.4.5.8 nvidia-cuda-cupti-cu12==12.4.127 \
  nvidia-cuda-nvrtc-cu12==12.4.127 nvidia-cuda-runtime-cu12==12.4.127 \
  nvidia-cudnn-cu12==9.1.0.70 nvidia-cufft-cu12==11.2.1.3 \
  nvidia-cusolver-cu12==11.6.1.9 nvidia-cusparse-cu12==12.3.1.170 \
  nvidia-nccl-cu12==2.21.5 nvidia-nvjitlink-cu12==12.4.127
```

(Equivalent to `bash scripts/setup_runpod_mujoco.sh` if that file is on the branch.)

---

## 3. Verify the GPU + code (must pass before training)

```bash
python -c "import jax; print(jax.devices())"     # MUST print [CudaDevice(id=0)]
python scripts/verify_mujoco.py                   # dist / net / env  ->  PASS
```

- If `jax.devices()` shows **CPU** or `net` **segfaults** → wrong CUDA/GPU. Re-check rules 1 & 2; redeploy on a **CUDA 12.4** pod with a **non-Blackwell** GPU.

---

## 4. Run experiments

### Single run (smoke / debugging)
```bash
# PPO — fastest, converges ~2,500 on HalfCheetah by ~130k steps
python scripts/run_train.py --config configs/mujoco_halfcheetah.yaml \
    --overrides agent.algorithm=ppo train.total_steps=130000 2>&1 | tee quick_ppo.txt

# TRPO
python scripts/run_train.py --config configs/mujoco_halfcheetah.yaml \
    --overrides agent.algorithm=trpo train.total_steps=130000 2>&1 | tee quick_trpo.txt

# REINFORCE — needs num_envs=1 AND num_minibatches=1
python scripts/run_train.py --config configs/mujoco_halfcheetah.yaml \
    --overrides agent.algorithm=reinforce env.num_envs=1 train.num_minibatches=1 \
    train.total_steps=100000 2>&1 | tee quick_reinforce.txt
```
Watch the learning curve: `grep "eval/mean_score" quick_ppo.txt`

### Full sweep (all 3 algos × seeds, resumable, best-eval, CSV)
```bash
python scripts/run_mujoco_benchmark.py --seeds 1 2 3
# other tasks: --env hopper | walker2d | ant     (give non-HalfCheetah more steps)
```
Results land in `results/mujoco/` (per-run logs + `<env>_summary.csv`).

---

## 5. Long runs — detach (tmux) + live monitor (W&B)

The recommended way to run the sweep: **tmux** keeps it alive after you disconnect;
**W&B** lets you watch live from any device. W&B is optional — without it you still
get file/CSV logs.

```bash
# 1. log in to W&B once (persists on the pod); SKIP this if not using W&B
pip install wandb
wandb login <YOUR_API_KEY>          # key from https://wandb.ai/authorize

# 2. start a persistent session
tmux new -s bench

# 3. INSIDE tmux, launch the sweep (drop --wandb-project for file-only logging)
python scripts/run_mujoco_benchmark.py --seeds 1 2 3 --wandb-project mujoco-pcpg

# 4. detach: press Ctrl+B then D   -> safe to close the browser/SSH and walk away
```

- **Watch live** at `wandb.ai/<your-username>/mujoco-pcpg` — from any device, no reconnect needed.
- **Reattach** to the raw terminal anytime: `tmux attach -t bench`.
- **No W&B?** watch the files: `grep "eval/mean_score" results/mujoco/halfcheetah_ppo_seed1.log`.
- The pod keeps running (and billing) until you Stop/Terminate it.
- `wandb login` must run **before** launch; it persists for the pod, so logging in outside tmux works inside it.

---

## 6. Save results, then shut down

Download before terminating — **Terminate wipes the pod.**
```bash
# on the POD:
runpodctl send results/mujoco quick_ppo.txt outputs/checkpoints/*.params
# it prints a code; on your MAC:  runpodctl receive <code>
```
(Install runpodctl on the Mac once: `brew install runpod/runpodctl/runpodctl`.)

Then **Pods → Terminate** (stops billing). Next time, redeploy + repeat §2 (~10 min).

---

## 7. Troubleshooting (every error seen, with the fix)

| Symptom | Cause | Fix |
|---|---|---|
| `toomanyrequests ... pull rate limit` (deploy) | Docker Hub throttling the node | Retry deploy / official template / change region |
| `Cannot uninstall blinker ... no RECORD file` | OS-installed blinker | add `--ignore-installed blinker` to the pip install |
| `ptxas too old ... CC 12.0` / segfault in `net` | **Blackwell GPU** | redeploy on a **non-Blackwell** GPU (A100/H100/L40S/4090) |
| `INTERNAL: the library was not initialized` / `Failed to capture gpu graph` | CUDA 12.8/12.9 vs jaxlib 0.4.38 | use a **CUDA 12.4 template** + the §2 force-pin |
| `jax.devices()` shows CPU | GPU init failed (CUDA mismatch) | same as above — match CUDA 12.4 |
| `device_put_replicated is deprecated` | JAX was upgraded too far | **keep `jax==0.4.38`** — do NOT `pip install -U jax` |
| `cannot reshape array ... into shape (32, -1, ...)` (REINFORCE) | num_minibatches wrong for single-env | add `train.num_minibatches=1` (and `env.num_envs=1`) |
| `No module named 'jax'` | fresh pod, deps not installed | run §2 |
| `fatal: 'origin/feature/mujoco' does not appear to be a repository` | wrong git syntax | `git pull origin feature/mujoco` (space, not slash) |
| `ptxas ... 12.4.131 ... miscompile ... clamping` (warning) | known CUDA 12.4 warning | **harmless**, ignore |
| `Failed to import warp` (warning) | Brax optional Warp backend | **harmless** (we use the `mjx` backend) |

---

## 8. Quick reference

- **Repo / branch:** `armin-sommer/Policy_Gradient_Predictive_Coding`, branch `feature/mujoco`
- **GPU:** A100 / H100 / L40S / RTX 4090  ·  **never** 5090 / PRO 6000 / B200 / B300
- **Template:** CUDA **12.4** (PyTorch 2.4 / cu124)
- **JAX:** pinned **0.4.38** — never upgrade
- **HalfCheetah PPO target:** ~2,500–3,000 (Brax scale); converges by ~130k steps
- **Per-algo config:** PPO/TRPO use `num_envs=256, num_minibatches=32`; REINFORCE needs `num_envs=1, num_minibatches=1`
