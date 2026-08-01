"""Run the MuJoCo SOTA benchmark (PPO/TRPO on Brax/MJX) on Modal GPUs.

Modal replaces the manual RunPod flow in RUNPOD.md: the image is built from the
pinned requirements files instead of a hand-pasted shell block, and the (env,
algo, seed) grid fans out across containers in parallel instead of running
sequentially in tmux. Logs and checkpoints land in a Modal Volume, so a killed
container never loses completed runs.

    modal run modal_app.py::verify                 # GPU + wiring gate (run first)
    modal run modal_app.py                         # the missing protocol cells
    modal run modal_app.py --seeds 4,5 --envs ant  # a subset
    modal run modal_app.py::fetch                  # pull logs into results/mujoco

Knobs are environment variables read at launch time (they are baked into the
function definition, so they must be set on the *local* `modal run`):

    PCPG_GPU=H100            GPU type (default A100-40GB). Never a Blackwell
                             card (RTX 5090 / PRO 6000 / B200) — jaxlib 0.4.38
                             cannot compile for compute capability 12.0.
    PCPG_MAX_CONTAINERS=4    Cap parallel GPUs (default 8) to stay inside the
                             workspace's GPU quota.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

# --- configuration ---------------------------------------------------------

# Non-Blackwell only (see docstring). A100-40GB fits num_envs=1024 comfortably.
GPU = os.environ.get("PCPG_GPU", "A100-40GB")
MAX_CONTAINERS = int(os.environ.get("PCPG_MAX_CONTAINERS", "8"))

# Modal's ceiling is 24h. A 30M-step TRPO run is the long pole; if a cell ever
# exceeds this it is killed and must be re-launched from scratch — run_train.py
# has no mid-training checkpoint resume, only the all-or-nothing skip below.
TIMEOUT = 24 * 60 * 60

REPO = "/workspace/PCPG"
RESULTS = "/results"

LOCAL_ROOT = Path(__file__).parent

# The grid. Protocol §7 wants 5 seeds per (task, algo); results/mujoco/ already
# has seeds 1-3 complete for all 8 cells, so 4 and 5 are what is outstanding.
DEFAULT_ENVS = ["halfcheetah", "hopper", "walker2d", "ant"]
DEFAULT_ALGOS = ["ppo", "trpo"]
DEFAULT_SEEDS = [4, 5]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    # libgl1/libglib2.0-0: MuJoCo links against these even headless.
    .apt_install("libgl1", "libglib2.0-0")
    .add_local_file("requirements-gpu.txt", "/deps/requirements-gpu.txt", copy=True)
    .add_local_file("requirements-cuda124.txt", "/deps/requirements-cuda124.txt", copy=True)
    .run_commands(
        "pip install --ignore-installed blinker -r /deps/requirements-gpu.txt",
        # Must come second and with --no-deps, or pip resolves CUDA back to 12.8.
        "pip install --force-reinstall --no-deps -r /deps/requirements-cuda124.txt",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    # Source last: editing an algorithm does not invalidate the dependency layer.
    .add_local_dir("src", f"{REPO}/src")
    .add_local_dir("scripts", f"{REPO}/scripts")
    .add_local_dir("configs", f"{REPO}/configs")
)

app = modal.App("pcpg-mujoco")

# Survives container death, so completed cells are never redone.
results_vol = modal.Volume.from_name("pcpg-results", create_if_missing=True)


def _log_path(env_name: str, algo: str, seed: int) -> Path:
    """Same naming as scripts/run_bench.sh, so logs interleave with existing ones."""
    return Path(RESULTS) / "mujoco" / f"{env_name}_{algo}_sota_seed{seed}.log"


# --- functions -------------------------------------------------------------


@app.function(image=image, gpu=GPU, volumes={RESULTS: results_vol}, timeout=900)
def verify():
    """Pre-sweep gate: the GPU is visible and dist/net/env wiring passes.

    Mirrors RUNPOD.md §3. If this fails, every training run will fail too, so
    always run it before launching the grid.
    """
    import jax

    devices = jax.devices()
    print(f"jax {jax.__version__}  devices={devices}")
    if devices[0].platform != "gpu":
        raise RuntimeError(f"expected a GPU, got {devices} — check the CUDA 12.4 pin")

    proc = subprocess.run(
        [sys.executable, f"{REPO}/scripts/verify_mujoco.py"],
        cwd=REPO, text=True, capture_output=True,
    )
    print(proc.stdout or "", proc.stderr or "", sep="\n")
    if proc.returncode != 0:
        raise RuntimeError(f"verify_mujoco.py failed (rc={proc.returncode})")
    return f"OK — {devices}"


@app.function(
    image=image,
    gpu=GPU,
    volumes={RESULTS: results_vol},
    timeout=TIMEOUT,
    max_containers=MAX_CONTAINERS,
)
def train_one(spec: dict):
    """Train one (env, algo, seed) cell using its frozen SOTA config.

    Resumable in the same sense as run_bench.sh: a log already containing
    "TRAINING END" is treated as complete and skipped, so re-launching the grid
    only fills the gaps.
    """
    env_name, algo, seed = spec["env"], spec["algo"], spec["seed"]
    tag = f"{env_name}/{algo}/seed{seed}"

    log_path = _log_path(env_name, algo, seed)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and "TRAINING END" in log_path.read_text(errors="replace"):
        print(f"== SKIP {tag} (already complete)")
        return {**spec, "status": "skipped", "score": _best_score(log_path.read_text(errors="replace"))}

    config = Path(REPO) / "configs" / f"mujoco_{env_name}_{algo}.yaml"
    if not config.exists():
        raise FileNotFoundError(f"missing frozen config {config} for {tag}")

    overrides = [f"seed={seed}"]
    if spec.get("total_steps"):
        overrides.append(f"train.total_steps={spec['total_steps']}")
    if spec.get("wandb_project"):
        overrides += [
            "wandb.mode=online",
            f"wandb.project={spec['wandb_project']}",
            f"wandb.group={env_name}_{algo}",
        ]

    cmd = [
        sys.executable, f"{REPO}/scripts/run_train.py",
        "--config", str(config),
        # Checkpoints must go to the volume: the mounted source tree is not a
        # durable (or reliably writable) place to write.
        "--checkpoint-dir", f"{RESULTS}/checkpoints",
        "--overrides", *overrides,
    ]

    print(f"== RUN {tag} -> {log_path}")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            cmd, cwd=REPO, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        last_commit = time.time()
        for line in proc.stdout:
            f.write(line)
            print(line, end="")
            # Periodic commit so `modal run ::fetch` can tail a live run and a
            # killed container still leaves partial progress behind.
            if time.time() - last_commit > 60:
                f.flush()
                results_vol.commit()
                last_commit = time.time()
        proc.wait()
    results_vol.commit()

    text = log_path.read_text(errors="replace")
    finished = "TRAINING END" in text
    dt = time.time() - t0
    status = "done" if (proc.returncode == 0 and finished) else "FAILED"
    print(f"== {status} {tag} in {dt/60:.1f} min (rc={proc.returncode})")
    return {**spec, "status": status, "score": _best_score(text), "minutes": round(dt / 60, 1)}


def _best_score(text: str):
    """Highest logged eval/mean_score, for a quick smoke read of a finished run.

    Protocol §5 forbids best-over-evals in reported results — this is a progress
    signal only. Headline numbers come from scripts/summarize_mujoco.py.
    """
    import re

    matches = re.findall(r"'eval/mean_score':\s*(?:np\.float64\()?(-?[\d.eE+]+)", text)
    return max(float(m) for m in matches) if matches else None


# --- entrypoints -----------------------------------------------------------


@app.local_entrypoint()
def main(
    envs: str = ",".join(DEFAULT_ENVS),
    algos: str = ",".join(DEFAULT_ALGOS),
    seeds: str = ",".join(str(s) for s in DEFAULT_SEEDS),
    total_steps: int = 0,
    wandb_project: str = "",
    skip_verify: bool = False,
):
    """Fan the (env, algo, seed) grid out across GPUs, one container per cell."""
    env_list = [e.strip() for e in envs.split(",") if e.strip()]
    algo_list = [a.strip() for a in algos.split(",") if a.strip()]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    specs = [
        {
            "env": e, "algo": a, "seed": s,
            "total_steps": total_steps or None,
            "wandb_project": wandb_project or None,
        }
        for e in env_list for a in algo_list for s in seed_list
    ]

    print(f"grid: {len(env_list)} envs x {len(algo_list)} algos x {len(seed_list)} seeds "
          f"= {len(specs)} runs on {GPU} (<= {MAX_CONTAINERS} in parallel)")

    if not skip_verify:
        print("gate: verifying GPU + wiring ...")
        print(verify.remote())

    results = list(train_one.map(specs))

    print("\n=== summary ===")
    for r in sorted(results, key=lambda r: (r["env"], r["algo"], r["seed"])):
        score = "n/a" if r["score"] is None else f"{r['score']:.1f}"
        print(f"  {r['env']:12} {r['algo']:5} seed{r['seed']}  {r['status']:8} best={score}")
    failed = [r for r in results if r["status"] == "FAILED"]
    print(f"\n{len(results) - len(failed)}/{len(results)} ok, {len(failed)} failed")
    print("pull logs with:  modal run modal_app.py::fetch")


@app.local_entrypoint()
def fetch(dest: str = "results/mujoco"):
    """Download the volume's logs into the local tree so they can be committed."""
    out = LOCAL_ROOT / dest
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for entry in results_vol.listdir("/mujoco"):
        name = Path(entry.path).name
        if not name.endswith(".log"):
            continue
        target = out / name
        with open(target, "wb") as f:
            for chunk in results_vol.read_file(entry.path):
                f.write(chunk)
        print(f"  {name}  ({target.stat().st_size/1024:.0f} KB)")
        n += 1
    print(f"wrote {n} logs to {out}")
