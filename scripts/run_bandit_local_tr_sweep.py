"""Sweep categorical layerwise direct-TR PCPG on the two-armed bandit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "bandit_pc_reinforce_parameter_local_tr.yaml"
EVAL_RE = re.compile(
    r"'eval/mean_score':\s*(?:np\.float64\()?(-?[\d.eE+]+)")


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: tuple[str, ...]


VARIANTS = (
    Variant("baseline", ()),
    Variant("radius_0p0001", ("train.parameter_pc_max_radius=0.0001",)),
    Variant("radius_0p0005", ("train.parameter_pc_max_radius=0.0005",)),
    Variant("radius_0p001", ("train.parameter_pc_max_radius=0.001",)),
    Variant("radius_0p0025", ("train.parameter_pc_max_radius=0.0025",)),
    Variant("radius_0p005", ("train.parameter_pc_max_radius=0.005",)),
    Variant("radius_0p02", ("train.parameter_pc_max_radius=0.02",)),
    Variant("radius_0p04", ("train.parameter_pc_max_radius=0.04",)),
)


def to_pi_optimal(score: float) -> float:
    return float(np.clip((score - 0.9) / 0.1, 0.0, 1.0))


def complete(log_path: Path) -> bool:
    return log_path.exists() and "TRAINING END" in log_path.read_text(
        errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--total-steps", type=int, default=60_000)
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "bandit_parameter_local_tr_sweep")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "config": str(config),
        "seeds": args.seeds,
        "total_steps": args.total_steps,
        "variants": [
            {"name": variant.name, "overrides": variant.overrides}
            for variant in VARIANTS
        ],
        "created_unix": time.time(),
    }, indent=2) + "\n")

    results = []
    failures = []
    for variant in VARIANTS:
        for seed in args.seeds:
            stem = f"bandit_local_tr_{variant.name}_seed{seed}"
            log_path = args.out_dir / f"{stem}.log"
            if not complete(log_path):
                command = [
                    sys.executable, str(ROOT / "scripts" / "run_train.py"),
                    "--config", str(config),
                    "--checkpoint-dir", str(args.out_dir / "checkpoints" / stem),
                    "--overrides", f"train.total_steps={args.total_steps}",
                    f"seed={seed}", *variant.overrides,
                ]
                print(f"start {variant.name} seed {seed}", flush=True)
                with log_path.open("w") as stream:
                    stream.write("# " + " ".join(command) + "\n")
                    stream.flush()
                    process = subprocess.run(
                        command, cwd=ROOT, stdout=stream,
                        stderr=subprocess.STDOUT, text=True)
                if process.returncode != 0 or not complete(log_path):
                    print(f"FAILED {variant.name} seed {seed}", flush=True)
                    failures.append((variant.name, seed))
                    continue
            scores = EVAL_RE.findall(log_path.read_text(errors="replace"))
            if scores:
                results.append({
                    "variant": variant.name,
                    "seed": seed,
                    "final_score": float(scores[-1]),
                    "final_pi_optimal": to_pi_optimal(float(scores[-1])),
                })

    ranking = []
    for variant in VARIANTS:
        values = [
            row["final_pi_optimal"] for row in results
            if row["variant"] == variant.name
        ]
        if values:
            ranking.append({
                "variant": variant.name,
                "n": len(values),
                "mean_final_pi_optimal": float(np.mean(values)),
                "std_final_pi_optimal": float(np.std(values)),
                "successes": int(np.sum(np.asarray(values) >= 0.9)),
            })
    ranking.sort(key=lambda row: row["mean_final_pi_optimal"], reverse=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n")
    (args.out_dir / "ranking.json").write_text(
        json.dumps(ranking, indent=2) + "\n")
    print(json.dumps(ranking, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"failed bandit sweep cells: {failures}")


if __name__ == "__main__":
    main()
