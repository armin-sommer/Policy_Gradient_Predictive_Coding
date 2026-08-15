"""Sweep NPG outer eta and the equivalent TR-PCPG radius scale on MuJoCo.

For TR-PCPG, multiplying a boundary step by ``m`` is represented without
breaking its fixed-point equations by setting ``radius = m**2 * base_radius``.
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
NPG_CONFIG = "configs/mujoco_halfcheetah_pc_actor_critic_parameter_npg.yaml"
TR_CONFIG = "configs/mujoco_halfcheetah_pc_actor_critic_parameter_tr.yaml"
EXACT_TR_CONFIG = (
    "configs/mujoco_halfcheetah_pc_actor_critic_parameter_exact_tr.yaml"
)
LOCAL_TR_CONFIG = (
    "configs/mujoco_halfcheetah_pc_actor_critic_parameter_local_tr.yaml"
)


@dataclass(frozen=True)
class Variant:
    name: str
    config: str
    override: str | None
    value: float | None


def value_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def variants(npg_etas: list[float], tr_multipliers: list[float],
             base_radius: float, include_exact_tr: bool = False,
             exact_tr_kls: list[float] | None = None,
             include_local_tr: bool = False) -> list[Variant]:
    result = [
        Variant(
            name=f"npg_eta_{value_token(eta)}",
            config=NPG_CONFIG,
            override="train.parameter_pc_step_size",
            value=eta,
        )
        for eta in npg_etas
    ]
    if include_local_tr:
        # Keep the newly requested local solver first for early validation.
        result.insert(0, Variant(
            name="local_tr",
            config=LOCAL_TR_CONFIG,
            override=None,
            value=None,
        ))
    result.extend(
        Variant(
            name=f"tr_multiplier_{value_token(multiplier)}",
            config=TR_CONFIG,
            override="train.parameter_pc_max_radius",
            value=base_radius * multiplier ** 2,
        )
        for multiplier in tr_multipliers
    )
    if include_exact_tr:
        for max_kl in exact_tr_kls or [0.01]:
            name = "exact_tr" if max_kl == 0.01 else (
                f"exact_tr_kl_{value_token(max_kl)}"
            )
            result.append(Variant(
                name=name,
                config=EXACT_TR_CONFIG,
                override="train.parameter_pc_max_kl",
                value=max_kl,
            ))
    return result


def is_complete(log_path: Path) -> bool:
    return log_path.exists() and "TRAINING END" in log_path.read_text(errors="replace")


def run_variant(gpu: str, variant: Variant, seed: int, total_steps: int,
                out_dir: Path, retries: int, print_lock: threading.Lock) -> None:
    stem = f"halfcheetah_{variant.name}_{total_steps}_seed{seed}"
    log_path = out_dir / f"{stem}.log"
    status_path = out_dir / f"{stem}.status.json"
    if is_complete(log_path):
        with print_lock:
            print(f"[gpu {gpu}] skip complete {variant.name} seed {seed}", flush=True)
        return

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_train.py"),
        "--config",
        str(REPO_ROOT / variant.config),
        "--no-save",
        "--overrides",
        f"train.total_steps={total_steps}",
        f"seed={seed}",
    ]
    if variant.override is not None:
        cmd.append(f"{variant.override}={variant.value}")
    cmd.extend(("train.eval_every=10", "train.num_eval_episodes=10"))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    started = time.time()
    metadata = {
        "variant": variant.name,
        "config": variant.config,
        "override": variant.override,
        "value": variant.value,
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
            print(f"[gpu {gpu}] start {variant.name} seed {seed} attempt {attempt}",
                  flush=True)
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
            print(f"[gpu {gpu}] failed {variant.name} seed {seed} attempt {attempt}",
                  flush=True)
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
            f"[gpu {gpu}] {metadata['state']} {variant.name} seed {seed} "
            f"in {metadata['elapsed_seconds'] / 60:.1f} min",
            flush=True,
        )


def worker(gpu: str, jobs: queue.Queue, total_steps: int, out_dir: Path,
           retries: int, print_lock: threading.Lock) -> None:
    while True:
        try:
            variant, seed = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            run_variant(gpu, variant, seed, total_steps, out_dir, retries, print_lock)
        finally:
            jobs.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--npg-etas", nargs="+", type=float,
                        default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    parser.add_argument("--tr-multipliers", nargs="+", type=float,
                        default=[0.5, 0.75, 1.25, 1.5, 2.0])
    parser.add_argument("--base-radius", type=float, default=0.01)
    parser.add_argument("--include-exact-tr", action="store_true")
    parser.add_argument("--include-local-tr", action="store_true")
    parser.add_argument(
        "--exact-tr-kls",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02, 0.04, 0.08],
    )
    parser.add_argument("--out-dir", type=Path, default=(
        REPO_ROOT / "results" / "mujoco_parameter_scale_sweep_1m"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sweep_variants = variants(
        args.npg_etas,
        args.tr_multipliers,
        args.base_radius,
        include_exact_tr=args.include_exact_tr,
        exact_tr_kls=args.exact_tr_kls,
        include_local_tr=args.include_local_tr,
    )
    jobs: queue.Queue = queue.Queue()
    # Validate all seeds of the explicit local solver first, then finish one
    # seed across the remaining grid for early coarse ranking.
    for variant in sweep_variants:
        if variant.name == "local_tr":
            for seed in args.seeds:
                jobs.put((variant, seed))
    for seed in args.seeds:
        for variant in sweep_variants:
            if variant.name != "local_tr":
                jobs.put((variant, seed))

    manifest = {
        "environment": "halfcheetah",
        "seeds": args.seeds,
        "total_steps": args.total_steps,
        "gpus": args.gpus,
        "npg_etas": args.npg_etas,
        "tr_multipliers": args.tr_multipliers,
        "base_radius": args.base_radius,
        "tr_radii": [args.base_radius * value ** 2 for value in args.tr_multipliers],
        "include_exact_tr": args.include_exact_tr,
        "exact_tr_kls": args.exact_tr_kls if args.include_exact_tr else [],
        "include_local_tr": args.include_local_tr,
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
    print("parameter-PCPG scale sweep finished", flush=True)


if __name__ == "__main__":
    main()
