"""Compare the two routes to the natural gradient on HalfCheetah.

  A. TARGET route  -- natural_target=True. The geometry is computed by hand and
                      baked into the target; PC hits it with squared error.
                      This is the current best config (781 +/- 47, 0/3 collapse).

  B. ENERGY route  -- likelihood_energy=True. No target; PC's output energy IS
                      the advantage-weighted Gaussian log-likelihood, so sigma
                      enters the objective PC minimises. This is the route that
                      actually tests "PC controls the geometry implicitly".

Everything else is held fixed: SGD, lr=0.03, ts=1.0, max_t1=20, [64,64], 256
envs, 1M steps. The only variable is where the geometry comes from.

Cells:
  natural            A, the baseline
  likelihood_signed  B with E = sum_i A_i * NLL_i
  likelihood_exp1    B with E = sum_i exp(A_i/1.0) * NLL_i   (RWR/MPO form)
  likelihood_exp03   B with tau=0.3 (sharper weighting)

WARNING on `signed`: with signed advantages the energy is unbounded below -- a
negative advantage rewards fleeing that sample. scripts/test_likelihood_energy.py
shows this concretely: one step at A=-2 moved |mu - z| from 0.84 to 30.8. Finite
max_t1 is the only thing bounding it. Expect `signed` to be unstable; it is
included because it is the faithful policy-gradient form, and its failure mode is
itself informative. The `exp` cells are the ones expected to be trainable.

Usage (pod):
    python scripts/run_likelihood_vs_natural.py --seeds 1 2 3 --skip-complete
    python scripts/analyze_pcpg_logs.py --results-dir results/likelihood_vs_natural
    python scripts/plot_collapse_anatomy.py --results-dir results/likelihood_vs_natural

~7 min/run: 4 cells x 3 seeds = 12 runs ~= 1.4 h.
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
    "halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt20.yaml")
RESULTS = REPO_ROOT / "results" / "likelihood_vs_natural"
EXPECTED_TOTAL = 1_000_000


def cells():
    return [
        ("natural",           {"train.natural_target": True}),
        ("likelihood_signed", {"train.natural_target": False,
                               "train.likelihood_energy": True,
                               "train.likelihood_adv_mode": "signed"}),
        ("likelihood_exp1",   {"train.natural_target": False,
                               "train.likelihood_energy": True,
                               "train.likelihood_adv_mode": "exp",
                               "train.likelihood_tau": 1.0}),
        ("likelihood_exp03",  {"train.natural_target": False,
                               "train.likelihood_energy": True,
                               "train.likelihood_adv_mode": "exp",
                               "train.likelihood_tau": 0.3}),
    ]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def is_complete(log):
    if not log.exists():
        return False
    txt = log.read_text(errors="replace")
    if "TRAINING END" not in txt:
        return False
    steps = [int(s) for s in re.findall(r"'training/total_steps': (\d+)", txt)]
    return bool(steps) and max(steps) >= 0.95 * EXPECTED_TOTAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--skip-complete", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    conds = cells()
    print(f"{len(conds)} cells x {len(args.seeds)} seeds = "
          f"{len(conds)*len(args.seeds)} runs (~7 min each)\nResults: {RESULTS}")

    for name, overrides in conds:
        cfg = yaml.safe_load(BASE_CFG.read_text())
        for path, value in overrides.items():
            section, key = path.split(".", 1)
            cfg.setdefault(section, {})[key] = value
        cfg["agent"]["experiment_name"] = name
        out = RESULTS / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (out / "meta.json").write_text(json.dumps(dict(
            config=name, commit=commit, base_config=BASE_CFG.name,
            overrides=overrides, total_steps=EXPECTED_TOTAL,
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        ), indent=2))

        for seed in args.seeds:
            log = out / f"seed_{seed}.log"
            if args.skip_complete and is_complete(log):
                print(f"skip {name} seed {seed} (complete)")
                continue
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_train.py"),
                   "--config", str(out / "config.yaml"),
                   "--overrides", f"seed={seed}", "--no-save"]
            print(f"\n=== {name} seed {seed} -> {log} ===")
            if args.dry_run:
                print("DRY:", " ".join(cmd))
                continue
            t0 = time.time()
            with log.open("w") as f:
                proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f,
                                      stderr=subprocess.STDOUT)
            print(f"{'done' if proc.returncode == 0 else 'FAILED'} ({time.time()-t0:.0f}s)")

    print(f"\nAnalyze:\n  python scripts/analyze_pcpg_logs.py --results-dir {RESULTS}")
    print(f"  python scripts/plot_collapse_anatomy.py --results-dir {RESULTS}")


if __name__ == "__main__":
    main()
