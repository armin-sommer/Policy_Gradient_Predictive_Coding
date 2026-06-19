"""Aggregate the MuJoCo benchmark runs into a shareable summary.

Reads the per-run logs in results/mujoco/ (one file per env/algo/seed, named
`{env}_{algo}_seed{N}.log`), reconstructs each run's eval/mean_score trajectory
over env steps, then produces:

  1. results/mujoco/summary_all.csv  -- best & final score, mean +/- std over seeds
  2. results/mujoco/<env>_curve.png  -- learning curves (mean line + SEM band) per algo
  3. results/mujoco/SUMMARY.md        -- a markdown table you can paste anywhere
  4. the same table printed to stdout

The logs are Python dict reprs from logging.info(...). Eval dicts carry
`eval/mean_score` (wrapped in np.float64(...)) but no step, so we tag each eval
with the most recent `training/total_steps` seen above it -- the same way the
wandb handler does.

    python scripts/summarize_mujoco.py
    python scripts/summarize_mujoco.py --results-dir results/mujoco
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ALGOS = ["ppo", "trpo", "reinforce"]
ALGO_ORDER = {a: i for i, a in enumerate(ALGOS)}

# {env}_{algo}_seed{N}.log  (env may contain underscores, so anchor on the algo)
FILENAME_RE = re.compile(r"^(?P<env>.+)_(?P<algo>ppo|trpo|reinforce)_seed(?P<seed>\d+)\.log$")
STEP_RE = re.compile(r"'training/total_steps':\s*(\d+)")
EVAL_RE = re.compile(r"'eval/mean_score':\s*(?:np\.float64\()?(-?[\d.eE+]+)")


def parse_run(path):
    """Return (steps, scores) arrays of the eval trajectory for one run."""
    steps, scores, last_step = [], [], 0
    for line in path.read_text().splitlines():
        m_step = STEP_RE.search(line)
        if m_step:
            last_step = int(m_step.group(1))
        m_eval = EVAL_RE.search(line)
        if m_eval:
            steps.append(last_step)
            scores.append(float(m_eval.group(1)))
    return np.asarray(steps, dtype=float), np.asarray(scores, dtype=float)


def collect(results_dir):
    """runs[env][algo] = list of (steps, scores) per seed."""
    runs = defaultdict(lambda: defaultdict(list))
    for path in sorted(results_dir.glob("*_seed*.log")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        steps, scores = parse_run(path)
        if scores.size:
            runs[m["env"]][m["algo"]].append((m["seed"], steps, scores))
    return runs


def stack_seeds(seed_runs):
    """Align seeds to the shortest trajectory; return (steps, mean, sem, n)."""
    min_len = min(len(s) for _, _, s in seed_runs)
    steps = seed_runs[0][1][:min_len]
    mat = np.vstack([s[:min_len] for _, _, s in seed_runs])  # (n_seeds, T)
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0) / np.sqrt(mat.shape[0])
    return steps, mean, sem, mat.shape[0]


def make_table(runs):
    """rows: (env, algo, n, best_mean, best_std, final_mean, final_std)."""
    rows = []
    for env in sorted(runs):
        for algo in sorted(runs[env], key=lambda a: ALGO_ORDER.get(a, 99)):
            seed_runs = runs[env][algo]
            bests = np.array([s.max() for _, _, s in seed_runs])
            finals = np.array([s[-1] for _, _, s in seed_runs])
            rows.append((env, algo, len(seed_runs),
                         bests.mean(), bests.std(),
                         finals.mean(), finals.std()))
    return rows


def write_csv(rows, out):
    lines = ["env,algo,n_seeds,best_mean,best_std,final_mean,final_std"]
    for env, algo, n, bm, bs, fm, fs in rows:
        lines.append(f"{env},{algo},{n},{bm:.2f},{bs:.2f},{fm:.2f},{fs:.2f}")
    out.write_text("\n".join(lines) + "\n")


def markdown_table(rows):
    out = ["| Environment | Algo | Seeds | Best (mean ± std) | Final (mean ± std) |",
           "|---|---|---|---|---|"]
    cur_env = None
    for env, algo, n, bm, bs, fm, fs in rows:
        env_cell = env if env != cur_env else ""
        cur_env = env
        out.append(f"| {env_cell} | {algo} | {n} | {bm:.0f} ± {bs:.0f} | {fm:.0f} ± {fs:.0f} |")
    return "\n".join(out)


def plot_curves(runs, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> skipping plots (CSV/markdown still written)")
        return []
    colors = {"ppo": "#1f77b4", "trpo": "#ff7f0e", "reinforce": "#2ca02c"}
    saved = []
    for env in sorted(runs):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for algo in sorted(runs[env], key=lambda a: ALGO_ORDER.get(a, 99)):
            seed_runs = runs[env][algo]
            steps, mean, sem, n = stack_seeds(seed_runs)
            c = colors.get(algo, None)
            ax.plot(steps, mean, label=f"{algo} (n={n})", color=c, linewidth=2)
            ax.fill_between(steps, mean - sem, mean + sem, color=c, alpha=0.2)
        ax.set_title(f"MuJoCo {env} — eval return (mean ± SEM)")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("eval/mean_score")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png = results_dir / f"{env}_curve.png"
        fig.savefig(png, dpi=130)
        plt.close(fig)
        saved.append(png)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str,
                        default=str(REPO_ROOT / "results" / "mujoco"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    runs = collect(results_dir)
    if not runs:
        print(f"no *_seed*.log runs found in {results_dir}")
        return

    rows = make_table(runs)
    write_csv(rows, results_dir / "summary_all.csv")
    md = markdown_table(rows)
    pngs = plot_curves(runs, results_dir)

    (results_dir / "SUMMARY.md").write_text(
        "# MuJoCo backprop benchmark (PPO / TRPO / REINFORCE)\n\n"
        + md + "\n\n"
        + "Learning curves: " + ", ".join(f"`{p.name}`" for p in pngs) + "\n")

    print(md)
    print(f"\nwrote {results_dir / 'summary_all.csv'}")
    print(f"wrote {results_dir / 'SUMMARY.md'}")
    for p in pngs:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
