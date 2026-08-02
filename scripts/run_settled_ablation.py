"""Settled-vs-unsettled ablation: does actually equilibrating inference change learning?

Every prior PCPG result was produced at ~1% settled (per-sample inference time
tau = t1/N ~ 0.01 at bench scale; see docs/PCPG_GEOMETRY_FINDINGS.md 4d). The
`train.inference_rate_correction` flag makes tau = max_t1, so inference genuinely
reaches equilibrium. This runs both arms on matched seeds and reports the difference.

The prediction from the geometry probe is NO CHANGE: `cos(d_BP, d_EQ) ~ 0.99` says
PC-at-equilibrium is directionally backprop in this architecture, so settling should
not move learning. Either outcome is informative -- no change confirms that reading,
a change means `d_EQ` is missing something.

Bandit (CPU) is the gating task here; the MuJoCo cells need a GPU.

    python scripts/run_settled_ablation.py                       # 5 seeds, both algos
    python scripts/run_settled_ablation.py --seeds 1 2 3 --algos pc_reinforce
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from bandit_inits import INIT_PRESETS, pi_optimal

ARM_MEANS = (1.0, 0.9)
EVAL_RE = re.compile(r"'eval/mean_score':\s*(?:np\.float64\()?(-?[\d.eE+]+)")


def to_pi_opt(score):
    """Stochastic-eval mean score -> pi(optimal arm); same mapping as
    run_bandit_comparison.to_pi_optimal."""
    return float(np.clip((score - ARM_MEANS[1]) / (ARM_MEANS[0] - ARM_MEANS[1]), 0, 1))


def run_one(algo, seed, corrected, steps, init, out_dir):
    tag = f"{algo}_seed{seed}_{'settled' if corrected else 'unsettled'}"
    log = out_dir / f"{tag}.log"
    if log.exists() and "TRAINING END" in log.read_text(errors="replace"):
        print(f"  skip {tag} (complete)")
    else:
        overrides = [
            f"agent.algorithm={algo}",
            f"seed={seed}",
            f"train.total_steps={steps}",
            f"agent.policy_init_logit_bias={INIT_PRESETS[init]}",
            f"train.inference_rate_correction={'true' if corrected else 'false'}",
        ]
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_train.py"),
               "--config", str(REPO_ROOT / "configs" / "bandit.yaml"),
               "--no-save", "--overrides", *overrides]
        t0 = time.time()
        with open(log, "w") as f:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f,
                                  stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        ok = "TRAINING END" in log.read_text(errors="replace")
        print(f"  {tag:44} {'ok' if (proc.returncode == 0 and ok) else 'FAILED':7} "
              f"{dt:6.0f}s")
        if proc.returncode != 0 or not ok:
            return None

    scores = EVAL_RE.findall(log.read_text(errors="replace"))
    if not scores:
        return None
    return {"algo": algo, "seed": seed, "corrected": corrected,
            "final_pi": to_pi_opt(float(scores[-1])),
            "curve": [to_pi_opt(float(s)) for s in scores]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["pc_reinforce", "pc_actor_critic"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--init", default="favor_suboptimal", choices=list(INIT_PRESETS))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "settled_ablation"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"init={args.init} (pi0(opt)={pi_optimal(INIT_PRESETS[args.init]):.3f})  "
          f"steps={args.steps}  seeds={args.seeds}")
    results = []
    for algo in args.algos:
        for seed in args.seeds:
            for corrected in (False, True):
                r = run_one(algo, seed, corrected, args.steps, args.init, out_dir)
                if r:
                    results.append(r)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 74)
    print("final pi(opt), mean +/- SEM over seeds   (success = final pi >= 0.9)")
    print("=" * 74)
    print(f"{'algo':>17} {'arm':>10} {'n':>3} {'final pi(opt)':>16} {'success':>9}")
    summary = {}
    for algo in args.algos:
        for corrected in (False, True):
            vals = [r["final_pi"] for r in results
                    if r["algo"] == algo and r["corrected"] == corrected]
            if not vals:
                continue
            m = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            succ = sum(v >= 0.9 for v in vals)
            summary[(algo, corrected)] = (m, sem, len(vals))
            print(f"{algo:>17} {'settled' if corrected else 'unsettled':>10} "
                  f"{len(vals):3} {m:9.3f} +/- {sem:.3f} {succ:6}/{len(vals)}")

    _paired_report(results, args, "final pi(opt)",
                   lambda r: r["final_pi"], "{:+.4f}")
    # Final pi saturates on this task (both arms ~0.98), so it has little power.
    # Steps-to-threshold uses the whole curve and discriminates learning SPEED.
    n_ev = max(len(r["curve"]) for r in results)
    per_eval = args.steps / n_ev
    print(f"\n({n_ev} evals over {args.steps} steps -> {per_eval:.0f} steps/eval)")
    _paired_report(results, args, "steps to pi(opt) >= 0.9",
                   lambda r: _steps_to(r["curve"], per_eval), "{:+.0f}")
    print(f"\nwrote {out_dir/'results.json'}")


def _steps_to(curve, per_eval, thr=0.9):
    for i, v in enumerate(curve):
        if v >= thr:
            return (i + 1) * per_eval
    return float("nan")


# two-sided t critical values, df = n-1, alpha = 0.05
_TCRIT = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262, 10: 2.228}


def _paired_report(results, args, label, metric, fmt):
    """Paired-by-seed comparison with a real t-test.

    A 2*SEM rule is too liberal at these n (t_crit is 2.776 at n=5), so the
    verdict uses the t distribution and reports the CI.
    """
    print(f"\ndelta (settled - unsettled), paired by seed — {label}:")
    for algo in args.algos:
        pairs = []
        for seed in args.seeds:
            a = next((metric(r) for r in results if r["algo"] == algo
                      and r["seed"] == seed and not r["corrected"]), None)
            b = next((metric(r) for r in results if r["algo"] == algo
                      and r["seed"] == seed and r["corrected"]), None)
            if a is not None and b is not None and not (np.isnan(a) or np.isnan(b)):
                pairs.append(b - a)
        if not pairs:
            continue
        n = len(pairs)
        m = float(np.mean(pairs))
        if n < 3:
            print(f"  {algo:>17}  {fmt.format(m)} (n={n})  too few seeds to judge")
            continue
        sem = float(np.std(pairs, ddof=1) / np.sqrt(n))
        tc = _TCRIT.get(n - 1, 1.96)
        t = m / sem if sem > 0 else float("inf")
        sig = abs(t) > tc
        lo, hi = m - tc * sem, m + tc * sem
        verdict = ("SIGNIFICANT — settling changes learning" if sig
                   else "no detectable effect")
        print(f"  {algo:>17}  {fmt.format(m)} +/- {fmt.format(sem).lstrip('+')} "
              f"(n={n})  t({n-1})={t:+.2f} vs {tc}  "
              f"95% CI [{fmt.format(lo)}, {fmt.format(hi)}]  {verdict}")


if __name__ == "__main__":
    main()
