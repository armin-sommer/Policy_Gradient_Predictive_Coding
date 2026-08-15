"""Screen direct layerwise error-neuron TR-PCPG hyperparameters on MuJoCo."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs"
    / "mujoco_halfcheetah_pc_actor_critic_parameter_local_tr.yaml"
)


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: tuple[str, ...]


VARIANTS = (
    Variant("baseline", ()),
    Variant("radius_0p005", ("train.parameter_pc_max_radius=0.005",)),
    Variant("radius_0p02", ("train.parameter_pc_max_radius=0.02",)),
    Variant("cg_100", ("train.parameter_pc_cg_iters=100",)),
    Variant("std_floor_m1p5", ("agent.log_std_min=-1.5",)),
    Variant("radius_0p0025", ("train.parameter_pc_max_radius=0.0025",)),
    Variant("radius_0p04", ("train.parameter_pc_max_radius=0.04",)),
    Variant("cg_300", ("train.parameter_pc_cg_iters=300",)),
    Variant("std_floor_m1", ("agent.log_std_min=-1.0",)),
    Variant("cg_50", ("train.parameter_pc_cg_iters=50",)),
    Variant("cg_25", ("train.parameter_pc_cg_iters=25",)),
    Variant("cg_10", ("train.parameter_pc_cg_iters=10",)),
    Variant("std_floor_m0p5", ("agent.log_std_min=-0.5",)),
)


def complete(log_path: Path) -> bool:
    return log_path.exists() and "TRAINING END" in log_path.read_text(
        errors="replace")


def run_one(gpu: str, config: Path, environment: str, variant: Variant,
            seed: int, total_steps: int, output_dir: Path, retries: int,
            print_lock: threading.Lock) -> bool:
    stem = f"{environment}_local_tr_{variant.name}_{total_steps}_seed{seed}"
    log_path = output_dir / f"{stem}.log"
    status_path = output_dir / f"{stem}.status.json"
    if complete(log_path):
        with print_lock:
            print(f"[gpu {gpu}] skip {variant.name} seed {seed}", flush=True)
        return True

    command = [
        sys.executable, str(ROOT / "scripts" / "run_train.py"),
        "--config", str(config), "--no-save", "--overrides",
        f"train.total_steps={total_steps}", f"seed={seed}",
        "train.eval_every=10", "train.num_eval_episodes=10",
        *variant.overrides,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(ROOT / "src")
    started = time.time()
    metadata = {
        "variant": variant.name,
        "overrides": variant.overrides,
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
        with print_lock:
            print(f"[gpu {gpu}] start {variant.name} seed {seed}", flush=True)
        with log_path.open("w") as stream:
            stream.write("# " + " ".join(command) + "\n")
            stream.flush()
            process = subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, text=True)
        if process.returncode == 0 and complete(log_path):
            break
        if log_path.exists():
            log_path.replace(
                output_dir / f"{stem}.attempt{attempt}.failed.log")
        time.sleep(10)

    succeeded = process is not None and process.returncode == 0 and complete(
        log_path)
    finished = time.time()
    metadata.update({
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "returncode": None if process is None else process.returncode,
        "state": "complete" if succeeded else "failed",
    })
    status_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with print_lock:
        print(f"[gpu {gpu}] {metadata['state']} {variant.name} seed {seed}",
              flush=True)
    return succeeded


def write_ranking(output_dir: Path, environment: str, total_steps: int) -> None:
    score_pattern = re.compile(r"'eval/mean_score': np\.float64\(([^)]+)\)")
    ranking = []
    for variant in VARIANTS:
        scores = []
        for log_path in output_dir.glob(
                f"{environment}_local_tr_{variant.name}_{total_steps}_seed*.log"):
            matches = score_pattern.findall(log_path.read_text(errors="replace"))
            if matches and complete(log_path):
                scores.append(float(matches[-1]))
        if scores:
            ranking.append({
                "variant": variant.name,
                "scores": scores,
                "mean_score": sum(scores) / len(scores),
            })
    ranking.sort(key=lambda row: row["mean_score"], reverse=True)
    (output_dir / "ranking.json").write_text(
        json.dumps(ranking, indent=2) + "\n")
    print(json.dumps(ranking, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "local_tr_hparam_sweep_5m")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    config_name = config.stem.removeprefix("mujoco_")
    environment = config_name.split("_", maxsplit=1)[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "config": str(config),
        "environment": environment,
        "total_steps": args.total_steps,
        "seeds": args.seeds,
        "gpus": args.gpus,
        "variants": [
            {"name": variant.name, "overrides": variant.overrides}
            for variant in VARIANTS
        ],
        "created_unix": time.time(),
    }, indent=2) + "\n")

    jobs: queue.Queue[tuple[Variant, int]] = queue.Queue()
    for seed in args.seeds:
        for variant in VARIANTS:
            jobs.put((variant, seed))
    print_lock = threading.Lock()
    failures = []

    def worker(gpu: str) -> None:
        while True:
            try:
                variant, seed = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                succeeded = run_one(
                    gpu, config, environment, variant, seed, args.total_steps,
                    args.output_dir, args.retries, print_lock)
                if not succeeded:
                    with print_lock:
                        failures.append((variant.name, seed))
            finally:
                jobs.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_ranking(args.output_dir, environment, args.total_steps)
    if failures:
        raise RuntimeError(f"failed sweep cells: {failures}")
    print("local direct-TR hyperparameter sweep finished", flush=True)


if __name__ == "__main__":
    main()
