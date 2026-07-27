"""Marco's gradient-clip check, done as a *numerical guard* test on the OLD
(Euclidean, 1/sigma^2) weight update.

Framing: this is not a trust region. A numerical guard should catch rare
pathological updates and otherwise stay out of the way. So the threshold is not
guessed -- it is read off the measured grad-norm tail, and the headline number is
the *bind rate*, not the return:

    bind rate < ~1%  and the collapse goes away  -> the instability was rare
                                                    numerical spikes (Marco right);
                                                    1/sigma^2 is keepable + guarded.
    only helps when it binds >10% of updates     -> structural: the amplifier
                                                    over-moves the policy on average,
                                                    and the clip is just a step limiter.

Two stages:

  stage 1 (measure):  one clip-free run (max_grad_norm=None) on the Euclidean SGD
      config at the WORKING lr=0.03. Read diag/policy_grad_norm_p99 / _p999 from
      the log. (The old clip runs in results/trust_region_kl_clip are unusable:
      they ran at lr=0.0003, 100x too small, so they failed for LR reasons.)

  stage 2 (bracket):  --clips p999 p99 p99/3  x 3 seeds, same config + clip on.
      Compare against the stage-1 run as the no-clip baseline.

Usage:
    # stage 1
    python scripts/run_gradclip_probe.py --stage measure
    python scripts/run_gradclip_probe.py --report          # prints the tail

    # stage 2 (values from the report)
    python scripts/run_gradclip_probe.py --stage bracket --clips 4.0 1.5 0.5 --seeds 1 2 3
    python scripts/analyze_pcpg_logs.py --results-dir results/gradclip_probe

Euclidean is the default (natural_target unset), so 1/sigma^2 is retained.
Runs are SGD: under Adam a global-norm clip is renormalised away, so it cannot
be tested there.
"""

import argparse
import ast
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
# SGD, lr=0.03 (the LR that actually learns), Euclidean target -> keeps 1/sigma^2.
BASE_CFG = REPO_ROOT / "configs" / "benchmark" / (
    "halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt20.yaml")
RESULTS = REPO_ROOT / "results" / "gradclip_probe"
MEASURE_NAME = "sgd_euclid_mt20_lr003_noclip_measure"


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def write_run(name, overrides):
    cfg = yaml.safe_load(BASE_CFG.read_text())
    for path, value in overrides.items():
        section, key = path.split(".", 1)
        cfg.setdefault(section, {})[key] = value
    cfg["agent"]["experiment_name"] = name
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "meta.json").write_text(json.dumps(dict(
        config=name, commit=git_commit(), base_config=BASE_CFG.name,
        overrides=overrides, natural_target=False,
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
    ), indent=2))
    return out


def run_seed(out, seed, dry):
    log = out / f"seed_{seed}.log"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_train.py"),
           "--config", str(out / "config.yaml"),
           "--overrides", f"seed={seed}", "--no-save"]
    print(f"\n=== {out.name} seed {seed} -> {log} ===")
    if dry:
        print("DRY:", " ".join(cmd))
        return
    t0 = time.time()
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f,
                              stderr=subprocess.STDOUT)
    print(f"{'done' if proc.returncode == 0 else 'FAILED'} ({time.time()-t0:.0f}s)")


def read_tail():
    """Last logged grad-norm percentiles from the measurement run."""
    log = RESULTS / MEASURE_NAME / "seed_1.log"
    if not log.exists():
        sys.exit(f"no measurement run yet: {log}\nrun --stage measure first")
    last = None
    for line in log.open():
        line = line.strip()
        if not line.startswith("{") or "policy_grad_norm_p99" not in line:
            continue
        try:
            last = ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", line))
        except Exception:
            continue
    if last is None:
        sys.exit("no grad-norm percentiles in the log -- is the logging patch in?")
    return last


def derive_clips(tail):
    """Bracket the guard around the measured tail: p999 (rare), p99, p99/3 (bites)."""
    p99, p999 = tail["diag/policy_grad_norm_p99"], tail["diag/policy_grad_norm_p999"]
    clips = [round(float(c), 4) for c in (p999, p99, p99 / 3.0)]
    # de-duplicate while preserving order (tails can be flat enough to collide)
    seen, out = set(), []
    for c in clips:
        if c > 0 and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def report():
    """Print the measured grad-norm tail + the clip values it implies."""
    last = read_tail()
    p50, p90 = last["diag/policy_grad_norm_p50"], last["diag/policy_grad_norm_p90"]
    p99, p999 = last["diag/policy_grad_norm_p99"], last["diag/policy_grad_norm_p999"]
    print("\nGrad-norm distribution (Euclidean SGD, clip-free, cumulative):")
    print(f"  p50   {p50:.4f}\n  p90   {p90:.4f}\n  p99   {p99:.4f}\n  p999  {p999:.4f}")
    print(f"  max   {last['diag/policy_grad_norm_max']:.4f}  (last update)")
    print(f"\n  tail ratio p99/p50 = {p99 / max(p50, 1e-9):.1f}x")
    if p99 <= 2.0 * p50:
        print("  -> essentially NO tail. Nothing for a numerical guard to catch;")
        print("     that itself answers Marco: the collapse is not spike-driven.")
    else:
        print("  -> real tail. A guard belongs near p99-p999.")
    print(f"\nSuggested stage 2:\n  python scripts/run_gradclip_probe.py --stage bracket "
          f"--clips {p999:.3g} {p99:.3g} {p99/3:.3g} --seeds 1 2 3\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["measure", "bracket", "auto"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--clips", nargs="+", type=float)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.report:
        return report()
    if args.stage in ("measure", "auto"):
        # clip off -> logged norms are the true, unclipped distribution.
        out = write_run(MEASURE_NAME, {"train.max_grad_norm": None})
        run_seed(out, 1, args.dry_run)
        if args.stage == "measure":
            print("\nNow read the tail:\n  python scripts/run_gradclip_probe.py --report")
            return

    if args.stage == "auto":
        # unattended: derive the bracket from the measured tail, no human step.
        if args.dry_run:
            print("\nDRY: would read the tail and bracket around p999 / p99 / p99/3")
            return
        report()
        args.clips = derive_clips(read_tail())
        print(f"\nauto-derived clips: {args.clips}\n")

    if args.stage in ("bracket", "auto"):
        if not args.clips:
            sys.exit("--clips required (get them from --report)")
        for clip in args.clips:
            tag = f"clip{str(clip).replace('.', 'p')}"
            out = write_run(f"sgd_euclid_mt20_lr003_{tag}",
                            {"train.max_grad_norm": float(clip)})
            for seed in args.seeds:
                run_seed(out, seed, args.dry_run)
        print(f"\nAnalyze:\n  python scripts/analyze_pcpg_logs.py --results-dir {RESULTS}")
        print("Then compare diag/policy_grad_norm_bind_rate vs collapse count.")
    else:
        ap.error("need --stage measure|bracket|auto or --report")


if __name__ == "__main__":
    main()
