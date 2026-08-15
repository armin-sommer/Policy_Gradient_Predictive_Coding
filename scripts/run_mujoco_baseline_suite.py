"""Run matched PPO and TRPO seeds on separate GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def complete(path: Path) -> bool:
    return path.exists() and "TRAINING END" in path.read_text(errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "4", "5"])
    parser.add_argument("--total-steps", type=int, default=30_000_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs: queue.Queue[tuple[str, int]] = queue.Queue()
    for algorithm in ("ppo", "trpo"):
        for seed in args.seeds:
            jobs.put((algorithm, seed))
    print_lock = threading.Lock()
    failures = []

    def worker(gpu: str) -> None:
        while True:
            try:
                algorithm, seed = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                stem = f"{args.env}_{algorithm}_{args.total_steps}_seed{seed}"
                log_path = args.out_dir / f"{stem}.log"
                if complete(log_path):
                    continue
                config = ROOT / "configs" / f"mujoco_{args.env}_{algorithm}.yaml"
                command = [
                    sys.executable, str(ROOT / "scripts" / "run_train.py"),
                    "--config", str(config),
                    "--checkpoint-dir", str(args.out_dir / "checkpoints" / stem),
                    "--overrides", f"train.total_steps={args.total_steps}",
                    f"seed={seed}",
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env["PYTHONPATH"] = str(ROOT / "src")
                with print_lock:
                    print(f"[gpu {gpu}] start {algorithm} seed {seed}", flush=True)
                started = time.time()
                with log_path.open("w") as stream:
                    stream.write("# " + " ".join(command) + "\n")
                    stream.flush()
                    process = subprocess.run(
                        command, cwd=ROOT, env=env, stdout=stream,
                        stderr=subprocess.STDOUT, text=True)
                status = {
                    "algorithm": algorithm,
                    "seed": seed,
                    "gpu": gpu,
                    "returncode": process.returncode,
                    "complete": complete(log_path),
                    "elapsed_seconds": time.time() - started,
                }
                (args.out_dir / f"{stem}.status.json").write_text(
                    json.dumps(status, indent=2) + "\n")
                with print_lock:
                    print(f"[gpu {gpu}] finish {algorithm} seed {seed}: {status}",
                          flush=True)
                    if process.returncode != 0 or not status["complete"]:
                        failures.append((algorithm, seed))
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"failed baseline runs: {failures}")


if __name__ == "__main__":
    main()
