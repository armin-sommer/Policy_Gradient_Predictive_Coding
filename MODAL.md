# Running the MuJoCo benchmark on Modal

Modal alternative to the manual RunPod flow in `RUNPOD.md`. Same pinned stack and
same frozen configs; the differences are that the image is built from
`requirements-gpu.txt` + `requirements-cuda124.txt` instead of a pasted shell
block, and the `(env, algo, seed)` grid runs **in parallel across containers**
instead of sequentially in tmux.

## Setup

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret> --profile=<workspace>
modal profile activate <workspace>
```

## Run

```bash
# 1. Gate: GPU visible + dist/net/env wiring passes. Always run this first —
#    if it fails, every training run fails the same way.
modal run modal_app.py::verify

# 2. The outstanding protocol cells (default grid: 4 envs x 2 algos x seeds 4,5).
modal run modal_app.py

# 3. Subsets.
modal run modal_app.py --envs ant,hopper --algos trpo --seeds 4,5
modal run modal_app.py --seeds 6,7 --wandb-project mujoco-pcpg

# 4. Pull logs into results/mujoco/ so they can be committed.
modal run modal_app.py::fetch
```

`modal run` streams every container's stdout, so you can watch all cells at once;
closing the terminal does not kill them (`modal app list` / `modal app logs` to
reattach).

### Knobs

Environment variables, read at launch time on the **local** side:

| Variable | Default | Notes |
|---|---|---|
| `PCPG_GPU` | `A100-40GB` | `H100`, `L40S`, `A10G` also fine. **Never** a Blackwell card (RTX 5090 / PRO 6000 / B200) — `jaxlib 0.4.38` cannot compile for compute capability 12.0. |
| `PCPG_MAX_CONTAINERS` | `8` | Cap on parallel GPUs; raise/lower to fit the workspace GPU quota. |

```bash
PCPG_GPU=H100 PCPG_MAX_CONTAINERS=4 modal run modal_app.py
```

## What runs

One container per `(env, algo, seed)` cell, using that cell's frozen SOTA config
`configs/mujoco_<env>_<algo>.yaml` — the same file `scripts/run_bench.sh` uses, so
Modal results are directly comparable to the committed seeds 1–3.

The default grid is **seeds 4 and 5** across `halfcheetah, hopper, walker2d, ant`
× `ppo, trpo` = 16 runs. That is what `docs/BENCHMARK_PROTOCOL.md` §7 (5 seeds per
task/algo) still needs; seeds 1–3 are already complete in `results/mujoco/`.

## Persistence

Logs and checkpoints go to the Modal Volume `pcpg-results`:

```
/results/mujoco/<env>_<algo>_sota_seed<seed>.log    # same names as run_bench.sh
/results/checkpoints/
```

The volume is committed every ~60s during a run, so a killed container still
leaves partial progress, and `modal volume ls pcpg-results /mujoco` works while
runs are live.

**Resume** works exactly like `run_bench.sh`: a log already containing
`TRAINING END` is treated as complete and skipped. Re-launching the same grid only
fills gaps. Note this is all-or-nothing per cell — `run_train.py` has no
mid-training checkpoint resume, so a cell killed at 80% restarts from 0. The
per-function timeout is Modal's 24h maximum.

## Known limitation: gRPC egress

Modal's client is **gRPC/HTTP-2 only**. Any network path that allows plain HTTPS
but not gRPC will fail at channel creation with:

```
Failed invoking function <create_channel_with_fallbacks>: Could not connect to the Modal server.
```

This fails in tens of milliseconds, *before* any auth exchange — so it is a
transport problem, not a bad token. `curl https://api.modal.com` returning 200
does **not** imply the client can connect. Notably, Claude Code's sandboxed web
sessions route egress through an HTTP CONNECT proxy that does not support gRPC, so
`modal run` cannot be launched from there; run it from a machine with direct
outbound access.

## Troubleshooting

Everything in `RUNPOD.md` §7 still applies (it is the same JAX/CUDA stack). Modal-specific:

| Symptom | Cause | Fix |
|---|---|---|
| `Could not connect to the Modal server` (instant) | gRPC blocked on the network path | see above — not a token problem |
| `verify` reports a CPU device | CUDA 12.4 pin didn't take | confirm `requirements-cuda124.txt` ran *after* `requirements-gpu.txt` with `--no-deps` |
| `ptxas too old` / segfault | Blackwell GPU | set `PCPG_GPU=A100-40GB` |
| runs queue instead of starting | workspace GPU quota | lower `PCPG_MAX_CONTAINERS` |
| `missing frozen config ...` | no `configs/mujoco_<env>_<algo>.yaml` | only the 8 committed cells have frozen configs |
