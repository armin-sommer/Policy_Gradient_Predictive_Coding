"""Plot a convergence curve (eval return vs env steps) from a training log.

Each algorithm logs metric dicts via logging.info(...). Training dicts carry
'training/total_steps'; eval dicts carry 'eval/mean_score' but no step, so we
associate each eval with the most recent training step (same convention as
run_bandit_comparison / summarize_mujoco).

Usage:
    python scripts/plot_curve.py results/mujoco/halfcheetah_trpo_sota_seed1.log
    python scripts/plot_curve.py <log1> <log2> ... --out curve.png --title "TRPO HalfCheetah"
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"'training/total_steps':\s*(?:np\.\w+\()?(-?[\d.eE+]+)")
EVAL_RE = re.compile(r"'eval/mean_score':\s*(?:np\.\w+\()?(-?[\d.eE+]+)")
FINAL_RE = re.compile(r"'final_eval/mean_score':\s*(?:np\.\w+\()?(-?[\d.eE+]+)")


def parse(path):
    steps, scores = [], []
    last_step = 0
    for line in Path(path).read_text().splitlines():
        m = STEP_RE.search(line)
        if m:
            last_step = int(float(m.group(1)))
        e = EVAL_RE.search(line) or FINAL_RE.search(line)
        if e:
            steps.append(last_step)
            scores.append(float(e.group(1)))
    return steps, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None, help="default: <first-log>_curve.png")
    ap.add_argument("--title", default="Convergence (eval return vs env steps)")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for log in args.logs:
        steps, scores = parse(log)
        if not scores:
            print(f"  no eval points found in {log}")
            continue
        label = Path(log).stem
        ax.plot(steps, scores, marker="o", markersize=3, label=label)
        print(f"  {label}: {len(scores)} evals, final={scores[-1]:.1f}, max={max(scores):.1f}")

    ax.set_xlabel("environment steps")
    ax.set_ylabel("eval mean return")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = args.out or (str(Path(args.logs[0]).with_suffix("")) + "_curve.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
