"""Run a matched HalfCheetah parameter-PCPG comparison across local GPUs.

The launcher assigns one process at a time to each GPU, writes independent logs
and status JSON files, and skips logs that already contain ``TRAINING END``.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Method:
    name: str
    config: str
    overrides: tuple[str, ...] = ()


METHODS = {
    "parameter_tr": Method(
        "parameter_tr",
        "configs/mujoco_halfcheetah_pc_actor_critic_parameter_tr.yaml",
        ("train.eval_every=10", "train.num_eval_episodes=10"),
    ),
    "parameter_npg": Method(
        "parameter_npg",
        "configs/mujoco_halfcheetah_pc_actor_critic_parameter_npg.yaml",
        ("train.eval_every=10", "train.num_eval_episodes=10"),
    ),
    "ppo": Method(
        "ppo",
        "configs/benchmark/halfcheetah_ppo_locked.yaml",
        ("train.eval_every=4", "train.num_eval_episodes=10"),
    ),
    "trpo": Method(
        "trpo",
        "configs/benchmark/halfcheetah_trpo_locked.yaml",
        ("train.eval_every=3", "train.num_eval_episodes=10"),
    ),
    "reinforce": Method(
        "reinforce",
        "configs/mujoco_halfcheetah.yaml",
        (
            "agent.algorithm=reinforce",
            "agent.experiment_name=reinforce_mujoco",
            "env.num_envs=1",
            "train.num_minibatches=1",
            "train.eval_every=40",
            "train.num_eval_episodes=10",
        ),
    ),
}


def is_complete(log_path: Path) -> bool:
    return log_path.exists() and "TRAINING END" in log_path.read_text(errors="replace")


def command_for(method: Method, seed: int, total_steps: int) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_train.py"),
        "--config",
        str(REPO_ROOT / method.config),
        "--no-save",
        "--overrides",
        f"train.total_steps={total_steps}",
        f"seed={seed}",
        *method.overrides,
    ]


def run_job(gpu: str, method: Method, seed: int, total_steps: int,
            out_dir: Path, retries: int, print_lock: threading.Lock) -> None:
    stem = f"halfcheetah_{method.name}_{total_steps}_seed{seed}"
    log_path = out_dir / f"{stem}.log"
    status_path = out_dir / f"{stem}.status.json"
    if is_complete(log_path):
        with print_lock:
            print(f"[gpu {gpu}] skip complete {method.name} seed {seed}", flush=True)
        return

    cmd = command_for(method, seed, total_steps)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    started = time.time()
    metadata = {
        "method": method.name,
        "seed": seed,
        "total_steps": total_steps,
        "gpu": gpu,
        "command": cmd,
        "started_unix": started,
        "state": "running",
    }
    proc = None
    for attempt in range(1, retries + 2):
        metadata["attempt"] = attempt
        status_path.write_text(json.dumps(metadata, indent=2) + "\n")
        with print_lock:
            print(
                f"[gpu {gpu}] start {method.name} seed {seed} attempt {attempt} "
                f"-> {log_path.name}",
                flush=True,
            )

        with log_path.open("w") as stream:
            stream.write("# " + " ".join(cmd) + "\n")
            stream.flush()
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if proc.returncode == 0 and is_complete(log_path):
            break
        failed_log = out_dir / f"{stem}.attempt{attempt}.failed.log"
        log_path.replace(failed_log)
        with print_lock:
            print(
                f"[gpu {gpu}] failed {method.name} seed {seed} attempt {attempt}; "
                f"saved {failed_log.name}",
                flush=True,
            )
        time.sleep(5)

    finished = time.time()
    succeeded = proc is not None and proc.returncode == 0 and is_complete(log_path)
    metadata.update({
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "returncode": None if proc is None else proc.returncode,
        "state": "complete" if succeeded else "failed",
    })
    status_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with print_lock:
        print(
            f"[gpu {gpu}] {metadata['state']} {method.name} seed {seed} "
            f"in {metadata['elapsed_seconds'] / 60:.1f} min",
            flush=True,
        )


def worker(gpu: str, jobs: queue.Queue, total_steps: int, out_dir: Path,
           retries: int, print_lock: threading.Lock) -> None:
    while True:
        try:
            method, seed = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(gpu, method, seed, total_steps, out_dir, retries, print_lock)
        finally:
            jobs.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=(
        REPO_ROOT / "results" / "mujoco_parameter_comparison_1m"))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=tuple(METHODS),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs: queue.Queue = queue.Queue()
    # Queue direct TR first so its three expensive seeds occupy separate GPUs.
    for name in args.methods:
        for seed in args.seeds:
            jobs.put((METHODS[name], seed))

    manifest = {
        "environment": "halfcheetah",
        "methods": args.methods,
        "seeds": args.seeds,
        "total_steps": args.total_steps,
        "gpus": args.gpus,
        "created_unix": time.time(),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu, jobs, args.total_steps, args.out_dir, args.retries, print_lock),
            daemon=False,
        )
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("comparison matrix finished", flush=True)


if __name__ == "__main__":
    main()
