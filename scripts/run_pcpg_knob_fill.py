"""Fill the two open cells in the PCPG knob table (docs/PCPG_KNOBS_SUMMARY.md).

Both knobs are motivated by the seed-level collapse analysis of the `_nat` runs
but were never swept:

  1. log_std_min  -- targets the *Adam* failure mode (policy-side: tanh-saturation
     + entropy drift). Raise the sigma floor on the collapsing Adam nat mt20 config
     and see if the collapse goes away.
  2. value_learning_rate -- targets the *SGD* failure mode (critic-side:
     value_explained_var diverging to ~-3). Sweep the critic LR on the diverging
     SGD nat mt80 config and see if the divergence is controllable.

Each condition layers overrides on an existing bench base config, writes the exact
resolved config + meta next to the logs, then runs config x seed sequentially --
same layout as run_pcpg_overnight_sweep.py so analyze_pcpg_logs.py just works.

Usage (on RunPod):
    python scripts/run_pcpg_knob_fill.py --seeds 1 2 3 --skip-complete
    python scripts/analyze_pcpg_logs.py --results-dir results/knob_fill_smin_vlr

Each bench run is ~7 min, so 4 configs x 3 seeds ~= 1.4 h.
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
CFG_DIR = REPO_ROOT / "configs" / "benchmark"
EXPECTED_TOTAL = 1_000_000

# base config file  ->  the collapsing config each knob is tested on
ADAM_MT20 = CFG_DIR / "halfcheetah_pc_actor_critic_adam_tanh_ts10_bench_mt20.yaml"
SGD_MT80 = CFG_DIR / "halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt80.yaml"


def conditions():
    """(name, base_yaml, {dotted.path: value}) per run condition.

    The baselines (log_std_min=-2 Adam nat mt20, value_lr=3e-4 SGD nat mt80)
    already exist in results/trust_region_kl_natural, so we only run the *new*
    points here and compare against those.
    """
    return [
        # -- Cell 1: sigma floor vs Adam saturation collapse (baseline smin=-2). --
        ("halfcheetah_pc_actor_critic_adam_tanh_ts10_bench_mt20_nat_sminm15",
         ADAM_MT20, {"train.natural_target": True, "agent.log_std_min": -1.5}),
        ("halfcheetah_pc_actor_critic_adam_tanh_ts10_bench_mt20_nat_sminm10",
         ADAM_MT20, {"train.natural_target": True, "agent.log_std_min": -1.0}),
        # -- Cell 2: critic LR vs SGD value divergence (baseline value_lr=3e-4). --
        ("halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt80_nat_vlr1e4",
         SGD_MT80, {"train.natural_target": True, "train.value_learning_rate": 1.0e-4}),
        ("halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt80_nat_vlr1e3",
         SGD_MT80, {"train.natural_target": True, "train.value_learning_rate": 1.0e-3}),
    ]


def resolve_config(base_yaml, overrides, name):
    cfg = yaml.safe_load(base_yaml.read_text())
    for path, value in overrides.items():
        section, key = path.split(".", 1)
        cfg.setdefault(section, {})[key] = value
    cfg["agent"]["experiment_name"] = name.replace("halfcheetah_", "")
    return cfg


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def is_complete(log_path):
    if not log_path.exists():
        return False
    txt = log_path.read_text(errors="replace")
    if "TRAINING END" not in txt:
        return False
    steps = [int(s) for s in re.findall(r"'training/total_steps': (\d+)", txt)]
    return bool(steps) and max(steps) >= 0.95 * EXPECTED_TOTAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--results-dir",
                    default=str(REPO_ROOT / "results" / "knob_fill_smin_vlr"))
    ap.add_argument("--skip-complete", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    conds = conditions()

    print(f"Running {len(conds)} configs x {len(args.seeds)} seeds "
          f"= {len(conds) * len(args.seeds)} seed-runs (~7 min each)")
    print(f"Results: {results_dir}")

    for name, base_yaml, overrides in conds:
        if not base_yaml.exists():
            print(f"!! base config missing: {base_yaml}; skipping {name}")
            continue
        out = results_dir / name
        out.mkdir(parents=True, exist_ok=True)
        cfg = resolve_config(base_yaml, overrides, name)
        cfg_path = out / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        (out / "meta.json").write_text(json.dumps(dict(
            config=name,
            commit=commit,
            base_config=base_yaml.name,
            overrides=overrides,
            total_steps=EXPECTED_TOTAL,
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            env=cfg["env"]["env_name"],
            algo=cfg["agent"]["algorithm"],
        ), indent=2))

        for seed in args.seeds:
            log = out / f"seed_{seed}.log"
            if args.skip_complete and is_complete(log):
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
            print(f"{'done' if proc.returncode == 0 else 'FAILED'} ({dt:.0f}s)")

    print(f"\nAnalyze with:\n  python scripts/analyze_pcpg_logs.py "
          f"--results-dir {results_dir}")


if __name__ == "__main__":
    main()
