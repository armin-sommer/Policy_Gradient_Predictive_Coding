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
EVAL_RE = re.compile(r"'eval/mean_score':\s*(?:np\.float64\()?(-?[\d.eE+]+)")


def best_score(text: str):
    matches = EVAL_RE.findall(text)
    return max(float(m) for m in matches) if matches else None


def run_one(algo, seed, env_name, total_steps, out_dir, wandb_project=None):
    log_path = out_dir / f"{env_name}_{algo}_seed{seed}.log"
    if log_path.exists() and "TRAINING END" in log_path.read_text():
        print(f"skip {algo} seed{seed} (complete)")
        return best_score(log_path.read_text())

    config = REPO_ROOT / "configs" / f"mujoco_{env_name}.yaml"
    if not config.exists():
        config = CONFIG  # fall back to the HalfCheetah config
    overrides = [f"agent.algorithm={algo}", f"seed={seed}", f"env.env_name={env_name}",
                 f"agent.experiment_name={algo}_mujoco"]  # run name: Exp_{algo}_mujoco__{env}__{seed}
    if total_steps is not None:
        overrides.append(f"train.total_steps={total_steps}")  # else use the config's budget
    if algo == "reinforce":
        overrides.append("env.num_envs=1")  # required by REINFORCE's rollout
        overrides.append("train.num_minibatches=1")  # single-env batch can't split into 32
    if algo == "trpo":
        # natural gradient needs a full-batch Fisher (one update/rollout, like sb3);
        # lighter eval so the ~1M-step run isn't dominated by 150s evals.
        overrides += ["train.num_minibatches=1", "train.batch_size=256",
                      "train.eval_every=20"]
    if wandb_project:
        overrides += [f"wandb.mode=online", f"wandb.project={wandb_project}",
                      f"wandb.group={env_name}_{algo}"]  # seeds overlay per (task, algo)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_train.py"),
           "--config", str(config), "--no-save", "--overrides", *overrides]

    print(f"\n#### {algo} {env_name} seed {seed} ####  (tail -f {log_path})")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f,
                              stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  FAILED (rc={proc.returncode}, {dt:.0f}s) -> see {log_path}")
        return None
    score = best_score(log_path.read_text())
    print(f"  done ({dt:.0f}s)  best return = {score}")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", nargs="*", default=ALGOS, choices=ALGOS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--env", type=str, default="halfcheetah")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override the per-task config budget (default: use config).")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "mujoco"))
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="If set, log to this W&B project (grouped by env_algo).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    scores = {algo: [] for algo in args.algos}
    for algo in args.algos:
        for seed in args.seeds:
            s = run_one(algo, seed, args.env, args.total_steps, out_dir, args.wandb_project)
            if s is not None:
                scores[algo].append(s)

    csv = out_dir / f"{args.env}_summary.csv"
    lines = ["algo,n,best_mean,best_std,best_scores"]
    budget = f"{args.total_steps} steps" if args.total_steps else "per-config budget"
    print(f"\n=== {args.env} best return (mean +/- std over seeds, {budget}) ===")
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
