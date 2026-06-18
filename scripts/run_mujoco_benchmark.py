"""Run the MuJoCo backprop baselines (PPO/TRPO/REINFORCE) across seeds and
aggregate final returns into a table + CSV.

Each run shells out to run_train.py; stdout streams to a per-run log file (so you
can `tail -f` it), and the final return is parsed from `final_eval/mean_score`.
REINFORCE is forced to num_envs=1 (its rollout requires it).

    python scripts/run_mujoco_benchmark.py
    python scripts/run_mujoco_benchmark.py --env hopper --seeds 1 2 3 --total-steps 2000000
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "mujoco_halfcheetah.yaml"
ALGOS = ["ppo", "trpo", "reinforce"]
FINAL_RE = re.compile(r"'final_eval/mean_score':\s*(?:np\.float64\()?(-?[\d.eE+]+)")


def final_score(text: str):
    matches = FINAL_RE.findall(text)
    return float(matches[-1]) if matches else None


def run_one(algo, seed, env_name, total_steps, out_dir):
    log_path = out_dir / f"{env_name}_{algo}_seed{seed}.log"
    if log_path.exists() and final_score(log_path.read_text()) is not None:
        print(f"skip {algo} seed{seed} (log exists)")
        return final_score(log_path.read_text())

    overrides = [f"agent.algorithm={algo}", f"seed={seed}",
                 f"env.env_name={env_name}", f"train.total_steps={total_steps}"]
    if algo == "reinforce":
        overrides.append("env.num_envs=1")  # required by REINFORCE's rollout
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_train.py"),
           "--config", str(CONFIG), "--no-save", "--overrides", *overrides]

    print(f"\n#### {algo} {env_name} seed {seed} ####  (tail -f {log_path})")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f,
                              stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  FAILED (rc={proc.returncode}, {dt:.0f}s) -> see {log_path}")
        return None
    score = final_score(log_path.read_text())
    print(f"  done ({dt:.0f}s)  final return = {score}")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", nargs="*", default=ALGOS, choices=ALGOS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--env", type=str, default="halfcheetah")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "mujoco"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    scores = {algo: [] for algo in args.algos}
    for algo in args.algos:
        for seed in args.seeds:
            s = run_one(algo, seed, args.env, args.total_steps, out_dir)
            if s is not None:
                scores[algo].append(s)

    csv = out_dir / f"{args.env}_summary.csv"
    lines = ["algo,n,final_mean,final_std,final_scores"]
    print(f"\n=== {args.env} final return (mean +/- std over seeds, "
          f"{args.total_steps} steps) ===")
    for algo in args.algos:
        vals = scores[algo]
        if vals:
            m, sd = float(np.mean(vals)), float(np.std(vals))
            print(f"  {algo:10} {m:10.1f} +/- {sd:6.1f}   (n={len(vals)})")
            lines.append(f"{algo},{len(vals)},{m:.3f},{sd:.3f},"
                         f"\"{[round(v, 1) for v in vals]}\"")
        else:
            print(f"  {algo:10} (no successful runs)")
            lines.append(f"{algo},0,nan,nan,[]")
    csv.write_text("\n".join(lines) + "\n")
    print(f"\nexpect ordering PPO ~ TRPO >> REINFORCE")
    print(f"wrote {csv}   (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
