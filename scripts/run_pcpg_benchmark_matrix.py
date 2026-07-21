"""Run the locked HalfCheetah benchmark matrix: config x seed.

Configs live in configs/benchmark/. Runs write a structured tree that
analyze_pcpg_logs.py consumes:

    results/benchmark_halfcheetah/<config_name>/seed_<seed>.log
    results/benchmark_halfcheetah/<config_name>/config.yaml   (exact config used)
    results/benchmark_halfcheetah/<config_name>/meta.json      (commit, env, algo, budget, timestamp)

`--skip-complete` skips a seed only if its log has TRAINING END AND actually reached
the config's step budget, so a short smoke run can never masquerade as a finished
benchmark run (the old --skip-complete footgun).

    python scripts/run_pcpg_benchmark_matrix.py --configs mechanism --seeds 1 2 3
    python scripts/run_pcpg_benchmark_matrix.py --configs all --seeds 1 2 3 --skip-complete
    python scripts/run_pcpg_benchmark_matrix.py --configs mechanism --total-steps 50000 --dry-run
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "configs" / "benchmark"

GROUPS = {
    "mechanism": [
        "halfcheetah_pc_actor_critic_adam_relu", "halfcheetah_pc_actor_critic_adam_tanh",
        "halfcheetah_pc_actor_critic_sgd_relu", "halfcheetah_pc_actor_critic_sgd_tanh",
    ],
    "target_scale": [
        "halfcheetah_pc_actor_critic_ts01", "halfcheetah_pc_actor_critic_ts03",
        "halfcheetah_pc_actor_critic_ts10",
    ],
    "adam_tanh_ts": [   # candidate: combine the two positive signals (adam critic + tanh policy)
        "halfcheetah_pc_actor_critic_adam_tanh_ts03",
        "halfcheetah_pc_actor_critic_adam_tanh_ts05",
        "halfcheetah_pc_actor_critic_adam_tanh_ts07",
    ],
    # pc_reinforce (no critic -> the PC policy update in isolation)
    "pcr_mechanism": [
        "halfcheetah_pc_reinforce_adam_relu", "halfcheetah_pc_reinforce_adam_tanh",
        "halfcheetah_pc_reinforce_sgd_relu", "halfcheetah_pc_reinforce_sgd_tanh",
    ],
    "pcr_target_scale": [
        "halfcheetah_pc_reinforce_ts01", "halfcheetah_pc_reinforce_ts03",
        "halfcheetah_pc_reinforce_ts10",
    ],
    "candidate_5m": [   # promoted candidate at extended budget (adam+tanh, ts 0.7/0.5)
        "halfcheetah_pc_actor_critic_adam_tanh_ts07_5m",
        "halfcheetah_pc_actor_critic_adam_tanh_ts05_5m",
    ],
    "capacity_5m": [   # capacity/batch matched to PPO/TRPO SOTA ([256,256], 1024 envs)
        "halfcheetah_pc_actor_critic_adam_tanh_ts07_sota_5m",
        "halfcheetah_pc_actor_critic_adam_tanh_ts05_sota_5m",
        "halfcheetah_pc_actor_critic_adam_tanh_ts03_sota_5m",
    ],
    "baselines": ["halfcheetah_ppo_locked", "halfcheetah_trpo_locked"],
}


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def config_budget_and_meta(cfg_path):
    c = yaml.safe_load(cfg_path.read_text())
    return (c.get("train", {}).get("total_steps"),
            dict(env=c.get("env", {}).get("env_name"),
                 algo=c.get("agent", {}).get("algorithm")))


def is_complete(log_path, expected_total):
    """Complete iff TRAINING END present AND the run reached ~the step budget."""
    if not log_path.exists():
        return False
    txt = log_path.read_text()
    if "TRAINING END" not in txt:
        return False
    steps = [int(s) for s in re.findall(r"'training/total_steps': (\d+)", txt)]
    return bool(steps) and max(steps) >= 0.95 * expected_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["mechanism"],
                    help="group (mechanism/target_scale/baselines/all) or explicit config stems")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--results-dir",
                    default=str(REPO_ROOT / "results" / "benchmark_halfcheetah"))
    ap.add_argument("--total-steps", type=int, default=None, help="override budget (smoke)")
    ap.add_argument("--skip-complete", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configs = []
    for c in args.configs:
        if c == "all":
            configs += sum(GROUPS.values(), [])
        elif c in GROUPS:
            configs += GROUPS[c]
        elif "*" in c:                       # glob over generated configs
            configs += sorted(p.stem for p in CFG_DIR.glob(f"{c}.yaml"))
        else:
            configs.append(c)

    results_dir = Path(args.results_dir)
    commit = git_commit()
    import time

    for name in configs:
        cfg_path = CFG_DIR / f"{name}.yaml"
        if not cfg_path.exists():
            print(f"MISSING config: {cfg_path}")
            sys.exit(1)
        cfg_budget, meta = config_budget_and_meta(cfg_path)
        budget = args.total_steps or cfg_budget
        out = results_dir / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(cfg_path.read_text())
        (out / "meta.json").write_text(json.dumps(dict(
            config=name, commit=commit, total_steps=budget,
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"), **meta), indent=2))

        for seed in args.seeds:
            log = out / f"seed_{seed}.log"
            if args.skip_complete and is_complete(log, budget):
                print(f"skip {name} seed {seed} (complete)")
                continue
            overrides = [f"seed={seed}"]
            if args.total_steps:
                overrides.append(f"train.total_steps={args.total_steps}")
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_train.py"),
                   "--config", str(cfg_path), "--overrides", *overrides, "--no-save"]
            print(f"=== {name} seed {seed} -> {log} ===")
            if args.dry_run:
                print("  DRY:", " ".join(cmd))
                continue
            t0 = time.time()
            with log.open("w") as f:
                proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
            dt = time.time() - t0
            print(f"{'done' if proc.returncode == 0 else 'FAILED'} ({dt:.0f}s)")
            # a failure does not abort the batch — record it and continue

    print(f"\n-> {results_dir}\n   analyze with: python scripts/analyze_pcpg_logs.py "
          f"--results-dir {results_dir}")


if __name__ == "__main__":
    main()
