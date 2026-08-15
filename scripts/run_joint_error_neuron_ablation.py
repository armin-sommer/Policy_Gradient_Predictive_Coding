"""Queue the two 30M direct-TR joint error-neuron ablations on four GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "joint": (
        "configs/"
        "mujoco_halfcheetah_pc_actor_critic_parameter_local_joint_tr.yaml"
    ),
    "precision": (
        "configs/"
        "mujoco_halfcheetah_pc_actor_critic_parameter_local_precision_tr.yaml"
    ),
}


def gpu_memory_used(gpu: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def wait_for_gpu(gpu: int, threshold_mib: int) -> None:
    while gpu_memory_used(gpu) >= threshold_mib:
        time.sleep(30)


def run_one(gpu: int, variant: str, seed: int, total_steps: int,
            output_dir: Path, retries: int) -> None:
    wait_for_gpu(gpu, threshold_mib=10_000)
    stem = f"halfcheetah_local_{variant}_tr_{total_steps}_seed{seed}"
    log_path = output_dir / f"{stem}.log"
    status_path = output_dir / f"{stem}.status.json"
    if log_path.exists() and "TRAINING END" in log_path.read_text(
            errors="replace"):
        print(f"[gpu {gpu}] skip complete {variant} seed {seed}", flush=True)
        return

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_train.py"),
        "--config", str(ROOT / VARIANTS[variant]),
        "--no-save", "--overrides",
        f"train.total_steps={total_steps}", f"seed={seed}",
        "train.eval_every=10", "train.num_eval_episodes=10",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT / "src")
    started = time.time()
    metadata = {
        "variant": variant,
        "seed": seed,
        "gpu": gpu,
        "total_steps": total_steps,
        "command": command,
        "started_unix": started,
        "state": "running",
    }
    process = None
    for attempt in range(1, retries + 2):
        metadata["attempt"] = attempt
        status_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[gpu {gpu}] start {variant} seed {seed}", flush=True)
        with log_path.open("w") as stream:
            stream.write("# " + " ".join(command) + "\n")
            stream.flush()
            process = subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, text=True)
        if process.returncode == 0 and "TRAINING END" in log_path.read_text(
                errors="replace"):
            break
        log_path.replace(output_dir / f"{stem}.attempt{attempt}.failed.log")
        time.sleep(10)

    succeeded = (
        process is not None and process.returncode == 0 and log_path.exists()
        and "TRAINING END" in log_path.read_text(errors="replace"))
    metadata.update({
        "finished_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "returncode": None if process is None else process.returncode,
        "state": "complete" if succeeded else "failed",
    })
    status_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[gpu {gpu}] {metadata['state']} {variant} seed {seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=30_000_000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "joint_error_neuron_ablation_30m")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # GPUs 2/3 start immediately. GPUs 0/1 wait for the old layerwise baselines,
    # then take seed 2 while seeds 1 and 3 remain paired on the same devices.
    assignments = {
        0: [("joint", 2)],
        1: [("precision", 2)],
        2: [("joint", 1), ("joint", 3)],
        3: [("precision", 1), ("precision", 3)],
    }

    def worker(gpu: int) -> None:
        for variant, seed in assignments[gpu]:
            run_one(
                gpu, variant, seed, args.total_steps,
                args.output_dir, args.retries)

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in assignments
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
