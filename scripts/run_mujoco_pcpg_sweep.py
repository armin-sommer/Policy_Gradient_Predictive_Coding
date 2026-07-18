"""Run Gaussian PCPG on a Brax MuJoCo env across seeds, algos, and network tiers.

Configs follow the pattern configs/mujoco_{env}_{algo}_{tier}.yaml.
Tiers (aligned with backprop baselines):
  bench — [64,64] MLP, 256 envs, 1M steps
  sota  — [256,256] MLP, 5M steps

Usage:
    python scripts/run_mujoco_pcpg_sweep.py --env halfcheetah --tiers bench sota --seeds 1 2 3
    python scripts/run_mujoco_pcpg_sweep.py --env hopper --tiers bench --total-steps 100000
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ALGOS = ["pc_reinforce", "pc_actor_critic"]
TIERS = ["bench", "sota"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper", choices=["hopper", "halfcheetah"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    parser.add_argument("--tiers", nargs="+", default=TIERS, choices=TIERS)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Defaults to results/mujoco_pcpg_{env}")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--skip-complete", action="store_true",
                        help="Skip runs whose log already contains TRAINING END")
    args = parser.parse_args()

    results_dir = Path(args.results_dir or REPO_ROOT / "results" / f"mujoco_pcpg_{args.env}")
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for tier in args.tiers:
        for algo in args.algos:
            cfg = REPO_ROOT / "configs" / f"mujoco_{args.env}_{algo}_{tier}.yaml"
            for seed in args.seeds:
                log_path = results_dir / f"{args.env}_{tier}_{algo}_seed{seed}.log"
                if args.skip_complete and log_path.exists():
                    if "TRAINING END" in log_path.read_text():
                        print(f"skip {tier} {algo} seed {seed} (complete)")
                        continue
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
                print(f"\n=== {tier} {algo} seed {seed} -> {log_path} ===\n")
                run_t0 = time.time()
                with log_path.open("w") as log_f:
                    proc = subprocess.run(
                        cmd, cwd=REPO_ROOT, stdout=log_f, stderr=subprocess.STDOUT)
                dt = time.time() - run_t0
                if proc.returncode != 0:
                    print(f"FAILED ({dt:.0f}s): {tier} {algo} seed {seed} (see {log_path})")
                    sys.exit(proc.returncode)
                print(f"done ({dt:.0f}s): {log_path}")

    print(f"\nAll runs finished in {time.time() - t0:.0f}s -> {results_dir}")


if __name__ == "__main__":
    main()
