"""Run the 6-7h capacity-matched HalfCheetah PCPG sweep.

This is a structured overnight mechanism sweep, not a final benchmark. It writes
one exact generated config per condition into the results tree, then runs
config x seed sequentially:

    results/overnight_halfcheetah_sota_sweep/<config_name>/config.yaml
    results/overnight_halfcheetah_sota_sweep/<config_name>/meta.json
    results/overnight_halfcheetah_sota_sweep/<config_name>/seed_<N>.log

The sweep asks three questions:
  1. Does SGD+tanh at SOTA capacity show the implicit trust-region behavior?
  2. Is Adam+tanh best around target_scale 0.4-0.6?
  3. Does lowering Adam actor LR stabilize ts05/ts07 without clipping targets?

Usage on RunPod:
    python scripts/run_pcpg_overnight_sweep.py --seeds 1 2 3 --skip-complete
    python scripts/analyze_pcpg_logs.py --results-dir results/overnight_halfcheetah_sota_sweep
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = REPO_ROOT / "configs" / "benchmark" / (
    "halfcheetah_pc_actor_critic_adam_tanh_ts07_sota_5m.yaml"
)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def load_base():
    cfg = yaml.safe_load(BASE_CFG.read_text())
    cfg["env"]["num_envs"] = 1024
    cfg["agent"]["width"] = 256
    cfg["agent"]["depth"] = 2
    cfg["agent"]["act_fn"] = "tanh"
    cfg["agent"]["algorithm"] = "pc_actor_critic"
    cfg["train"]["total_steps"] = 5_000_000
    cfg["train"]["max_t1"] = 20
    cfg["train"]["pc_steps_per_update"] = 1
    cfg["train"]["update_epochs"] = 2
    cfg["train"]["num_minibatches"] = 4
    cfg["train"]["normalize_advantages"] = True
    cfg["train"]["normalize_rewards"] = True
    cfg["train"]["eval_every"] = 10
    cfg["train"]["num_eval_episodes"] = 10
    return cfg


def set_cfg(cfg, *, name, optimizer, target_scale, learning_rate,
            value_learning_rate=3.0e-4):
    cfg = json.loads(json.dumps(cfg))
    cfg["agent"]["experiment_name"] = name.replace("halfcheetah_", "")
    cfg["train"]["optimizer"] = optimizer
    cfg["train"]["target_scale"] = target_scale
    cfg["train"]["learning_rate"] = learning_rate
    cfg["train"]["value_learning_rate"] = value_learning_rate
    return cfg


def sweep_configs():
    base = load_base()
    configs = []

    # 1. SGD trust-region mechanism test. Keep critic value LR at the existing
    # conservative value; sweep the actor LR that controls realized policy steps.
    for lr_tag, lr in [
        ("lr0003", 3.0e-3),
        ("lr001", 1.0e-2),
        ("lr003", 3.0e-2),
    ]:
        name = f"halfcheetah_pc_actor_critic_sgd_tanh_ts05_sota_{lr_tag}_5m"
        configs.append((name, set_cfg(
            base, name=name, optimizer="sgd", target_scale=0.5,
            learning_rate=lr)))

    # 2. Adam target-scale fine sweep around the current promising range.
    for ts_tag, ts in [("ts04", 0.4), ("ts06", 0.6)]:
        name = f"halfcheetah_pc_actor_critic_adam_tanh_{ts_tag}_sota_5m"
        configs.append((name, set_cfg(
            base, name=name, optimizer="adam", target_scale=ts,
            learning_rate=3.0e-4)))

    # 3. Adam lower actor LR checks for ts07 and ts05. Value LR stays fixed so the
    # test isolates actor-update size as much as possible.
    for ts_tag, ts in [("ts07", 0.7), ("ts05", 0.5)]:
        for lr_tag, lr in [("lr0001", 1.0e-4), ("lr0002", 2.0e-4)]:
            name = f"halfcheetah_pc_actor_critic_adam_tanh_{ts_tag}_sota_{lr_tag}_5m"
            configs.append((name, set_cfg(
                base, name=name, optimizer="adam", target_scale=ts,
                learning_rate=lr)))

    return configs


def is_complete(log_path, expected_total):
    if not log_path.exists():
        return False
    txt = log_path.read_text(errors="replace")
    if "TRAINING END" not in txt:
        return False
    steps = [int(s) for s in re.findall(r"'training/total_steps': (\d+)", txt)]
    return bool(steps) and max(steps) >= 0.95 * expected_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--results-dir",
                    default=str(REPO_ROOT / "results" / "overnight_halfcheetah_sota_sweep"))
    ap.add_argument("--skip-complete", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    configs = sweep_configs()
    expected_total = 5_000_000

    print(f"Running {len(configs)} configs x {len(args.seeds)} seeds "
          f"= {len(configs) * len(args.seeds)} seed-runs")
    print(f"Results: {results_dir}")

    for name, cfg in configs:
        out = results_dir / name
        out.mkdir(parents=True, exist_ok=True)
        cfg_path = out / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        (out / "meta.json").write_text(json.dumps(dict(
            config=name,
            commit=commit,
            total_steps=expected_total,
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            env=cfg["env"]["env_name"],
            algo=cfg["agent"]["algorithm"],
            optimizer=cfg["train"]["optimizer"],
            target_scale=cfg["train"]["target_scale"],
            learning_rate=cfg["train"]["learning_rate"],
            value_learning_rate=cfg["train"]["value_learning_rate"],
            width=cfg["agent"]["width"],
            num_envs=cfg["env"]["num_envs"],
        ), indent=2))

        for seed in args.seeds:
            log = out / f"seed_{seed}.log"
            if args.skip_complete and is_complete(log, expected_total):
                print(f"skip {name} seed {seed} (complete)")
                continue
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_train.py"),
                   "--config", str(cfg_path), "--overrides", f"seed={seed}",
                   "--no-save"]
            print(f"\n=== {name} seed {seed} -> {log} ===")
            if args.dry_run:
                print("DRY:", " ".join(cmd))
                continue
            t0 = time.time()
            with log.open("w") as f:
                proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f,
                                      stderr=subprocess.STDOUT)
            dt = time.time() - t0
            status = "done" if proc.returncode == 0 else "FAILED"
            print(f"{status} ({dt:.0f}s)")

    print(f"\nAnalyze with:\n  python scripts/analyze_pcpg_logs.py "
          f"--results-dir {results_dir}")


if __name__ == "__main__":
    main()
