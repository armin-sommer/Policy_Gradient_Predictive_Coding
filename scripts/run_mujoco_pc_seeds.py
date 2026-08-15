"""Run one PCPG config across matched seeds on separate GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--total-steps", type=int, default=30_000_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if len(args.gpus) < len(args.seeds):
        raise ValueError("provide at least one GPU per seed")
    failures = []
    failure_lock = threading.Lock()

    def worker(gpu: str, seed: int) -> None:
        stem = f"{config.stem}_{args.total_steps}_seed{seed}"
        log_path = args.out_dir / f"{stem}.log"
        if log_path.exists() and "TRAINING END" in log_path.read_text(
                errors="replace"):
            return
        command = [
            sys.executable, str(ROOT / "scripts" / "run_train.py"),
            "--config", str(config),
            "--checkpoint-dir", str(args.out_dir / "checkpoints" / stem),
            "--overrides", f"train.total_steps={args.total_steps}", f"seed={seed}",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONPATH"] = str(ROOT / "src")
        print(f"[gpu {gpu}] start seed {seed}", flush=True)
        try:
            with log_path.open("w") as stream:
                stream.write("# " + " ".join(command) + "\n")
                stream.flush()
                subprocess.run(
                    command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, text=True, check=True)
        except Exception as exc:
            with failure_lock:
                failures.append((seed, str(exc)))

    threads = [
        threading.Thread(target=worker, args=(gpu, seed))
        for gpu, seed in zip(args.gpus, args.seeds)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"failed PCPG runs: {failures}")


if __name__ == "__main__":
    main()
