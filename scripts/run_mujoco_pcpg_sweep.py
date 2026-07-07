"""Run Gaussian PCPG on Hopper across seeds and both PC variants.

Usage:
    python scripts/run_mujoco_pcpg_sweep.py --seeds 1 2 3
    python scripts/run_mujoco_pcpg_sweep.py --algos pc_reinforce --total-steps 100000
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ALGO_CONFIGS = {
    "pc_reinforce": "configs/mujoco_hopper_pc_reinforce.yaml",
    "pc_actor_critic": "configs/mujoco_hopper_pc_actor_critic.yaml",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--algos", nargs="+", default=list(ALGO_CONFIGS),
                        choices=list(ALGO_CONFIGS))
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--results-dir", type=str,
                        default=str(REPO_ROOT / "results" / "mujoco_pcpg_hopper"))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for algo in args.algos:
        cfg = REPO_ROOT / ALGO_CONFIGS[algo]
        for seed in args.seeds:
            log_path = results_dir / f"hopper_{algo}_seed{seed}.log"
            overrides = [f"seed={seed}"]
            if args.total_steps is not None:
                overrides.append(f"train.total_steps={args.total_steps}")
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_train.py"),
                "--config", str(cfg),
                "--overrides", *overrides,
            ]
            if args.no_save:
                cmd.append("--no-save")
            print(f"\n=== {algo} seed {seed} -> {log_path} ===\n")
            with log_path.open("w") as log_f:
                proc = subprocess.run(
                    cmd, cwd=REPO_ROOT, stdout=log_f, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                print(f"FAILED: {algo} seed {seed} (see {log_path})")
                sys.exit(proc.returncode)
            print(f"done: {log_path}")


if __name__ == "__main__":
    main()
