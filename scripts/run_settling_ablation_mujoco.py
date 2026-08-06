"""Settled-vs-unsettled inference ablation on MuJoCo (GPU).

The MuJoCo counterpart of scripts/run_settled_ablation.py (which is bandit-only).
Both arms of `train.inference_rate_correction` on matched seeds, paired comparison.

Why: every committed PCPG result was produced with inference ~1% settled -- the
per-sample inference time is tau = max_t1/N, and at bench N=2048 that is 20/2048 ~
0.01 where the residual is still 0.988 of its initial value. The flag makes tau =
max_t1, so inference actually equilibrates. On the bandit, settling significantly
SLOWED learning; whether MuJoCo behaves the same way is the open question.
See docs/PCPG_GEOMETRY_FINDINGS.md sections 4d-4e.

Run --calibrate FIRST. The ~9.5x cost estimate for the settled arm was extrapolated
from CPU, and one short run replaces that extrapolation with a measured number before
you commit GPU-hours to the full grid.

    python scripts/run_settling_ablation_mujoco.py --calibrate
    python scripts/run_settling_ablation_mujoco.py --seeds 1 2 3
    python scripts/run_settling_ablation_mujoco.py --seeds 1 2 3 --algos pc_reinforce
"""

import argparse
import json
import re
import subprocess
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

STEP_RE = re.compile(r"'training/total_steps':\s*(\d+)")
EVAL_RE = re.compile(r"'eval/mean_score':\s*(?:np\.float64\()?(-?[\d.eE+]+)")
WALL_RE = re.compile(r"'training/walltime':\s*(?:np\.float64\()?([\d.]+)")

# two-sided t critical values, df = n-1, alpha = 0.05
TCRIT = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228}


def run_one(env, algo, tier, seed, settled, steps, max_t1, out_dir, extra=()):
    """One (env, algo, seed, arm) cell. Resumable: a log with TRAINING END is skipped."""
    arm = "settled" if settled else "unsettled"
    tag = f"{env}_{algo}_{tier}_{arm}_seed{seed}"
    log = out_dir / f"{tag}.log"
    if log.exists() and "TRAINING END" in log.read_text(errors="replace"):
        print(f"  skip {tag} (complete)")
    else:
        config = REPO_ROOT / "configs" / f"mujoco_{env}_{algo}_{tier}.yaml"
        if not config.exists():
            raise FileNotFoundError(config)
        overrides = [
            f"seed={seed}",
            # Explicit on both arms: never rely on what the YAML happens to carry,
            # because the committed configs now enable it by default.
            f"train.inference_rate_correction={'true' if settled else 'false'}",
            f"train.max_t1={max_t1}",
            *extra,
        ]
        if steps:
            overrides.append(f"train.total_steps={steps}")
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_train.py"),
               "--config", str(config), "--no-save", "--overrides", *overrides]
        t0 = time.time()
        with open(log, "w") as f:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f,
                                  stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        ok = proc.returncode == 0 and "TRAINING END" in log.read_text(errors="replace")
        print(f"  {tag:52} {'ok' if ok else 'FAILED':7} {dt/60:6.1f} min")
        if not ok:
            return None

    text = log.read_text(errors="replace")
    scores = [float(x) for x in EVAL_RE.findall(text)]
    walls = WALL_RE.findall(text)
    if not scores:
        return None
    return {"env": env, "algo": algo, "tier": tier, "seed": seed, "settled": settled,
            "final": scores[-1], "best": max(scores), "curve": scores,
            "walltime": float(walls[-1]) if walls else None}


def paired(results, algo, seeds, key):
    d = []
    for s in seeds:
        a = next((r[key] for r in results if r["algo"] == algo and r["seed"] == s
                  and not r["settled"]), None)
        b = next((r[key] for r in results if r["algo"] == algo and r["seed"] == s
                  and r["settled"]), None)
        if a is not None and b is not None:
            d.append(b - a)
    return np.array(d, dtype=float)


def report(results, args, key, label):
    print(f"\ndelta (settled - unsettled), paired by seed -- {label}:")
    for algo in args.algos:
        d = paired(results, algo, args.seeds, key)
        if d.size == 0:
            continue
        n, m = d.size, float(d.mean())
        if n < 3:
            print(f"  {algo:>17}  {m:+9.1f} (n={n})  too few seeds to judge")
            continue
        sem = float(d.std(ddof=1) / sqrt(n))
        tc = TCRIT.get(n - 1, 1.96)
        t = m / sem if sem > 0 else float("inf")
        verdict = ("SIGNIFICANT -- settling changes MuJoCo learning" if abs(t) > tc
                   else "no detectable effect")
        print(f"  {algo:>17}  {m:+9.1f} +/- {sem:.1f} (n={n})  t({n-1})={t:+.2f} "
              f"vs {tc}  95% CI [{m-tc*sem:+.1f}, {m+tc*sem:+.1f}]  {verdict}")


def calibrate(args):
    """Measure the real settled/unsettled cost ratio on this GPU, cheaply."""
    out = Path(args.out_dir) / "calibrate"
    out.mkdir(parents=True, exist_ok=True)
    steps = args.calibrate_steps
    print(f"calibration: {args.envs[0]} / {args.algos[0]} / {args.tier}, "
          f"{steps:,} steps per arm, max_t1={args.max_t1}\n")
    times = {}
    for settled in (False, True):
        r = run_one(args.envs[0], args.algos[0], args.tier, 1, settled, steps,
                    args.max_t1, out)
        times[settled] = r["walltime"] if r and r["walltime"] else None
    u, s = times.get(False), times.get(True)
    print()
    if u and s:
        print(f"unsettled {u:8.1f} s   settled {s:8.1f} s   ratio {s/u:5.2f}x")
        print(f"\nExtrapolating to a full {args.steps or 'config-budget'}-step run:")
        for label, per in (("unsettled", u), ("settled", s)):
            scaled = per * ((args.steps or 1_000_000) / steps)
            print(f"  {label:10} ~{scaled/60:6.1f} min/run")
        n_runs = len(args.envs) * len(args.algos) * len(args.seeds)
        tot = (u + s) * ((args.steps or 1_000_000) / steps) * n_runs / 3600
        print(f"  full grid ({n_runs} seed-cells x 2 arms): ~{tot:.1f} GPU-hours")
        print("\nCPU extrapolation predicted ~9.5x on the PC step alone; the ratio "
              "above is what actually matters and supersedes it.")
    else:
        print("calibration incomplete -- check the logs in", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["halfcheetah"])
    ap.add_argument("--algos", nargs="+", default=["pc_actor_critic"],
                    choices=["pc_actor_critic", "pc_reinforce"])
    ap.add_argument("--tier", default="bench", choices=["bench", "sota"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=0,
                    help="override the config's step budget (default: use config)")
    ap.add_argument("--max-t1", type=int, default=10,
                    help="10 already converges when corrected and is ~1.6x cheaper "
                         "than 20 (39 vs 62 solver steps).")
    ap.add_argument("--calibrate", action="store_true",
                    help="run one short pair and report the measured cost ratio")
    ap.add_argument("--calibrate-steps", type=int, default=100_000)
    ap.add_argument("--out-dir",
                    default=str(REPO_ROOT / "results" / "settling_ablation_mujoco"))
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args)
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"envs={args.envs} algos={args.algos} tier={args.tier} seeds={args.seeds} "
          f"max_t1={args.max_t1}")

    results = []
    for env in args.envs:
        for algo in args.algos:
            for seed in args.seeds:
                for settled in (False, True):
                    r = run_one(env, algo, args.tier, seed, settled, args.steps,
                                args.max_t1, out)
                    if r:
                        results.append(r)

    (out / "results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print("final eval return, mean +/- SEM over seeds")
    print("=" * 78)
    print(f"{'algo':>17} {'arm':>10} {'n':>3} {'final':>18} {'best':>12}")
    for algo in args.algos:
        for settled in (False, True):
            v = [r["final"] for r in results
                 if r["algo"] == algo and r["settled"] == settled]
            b = [r["best"] for r in results
                 if r["algo"] == algo and r["settled"] == settled]
            if not v:
                continue
            sem = float(np.std(v, ddof=1) / sqrt(len(v))) if len(v) > 1 else 0.0
            print(f"{algo:>17} {'settled' if settled else 'unsettled':>10} {len(v):3} "
                  f"{np.mean(v):11.1f} +/- {sem:5.1f} {np.mean(b):12.1f}")

    report(results, args, "final", "final eval return")
    # best-eval is progress-only per BENCHMARK_PROTOCOL section 5; reported as a
    # sensitivity check, never as the headline.
    report(results, args, "best", "best eval return (sensitivity only)")
    print(f"\nwrote {out/'results.json'}")


if __name__ == "__main__":
    main()
