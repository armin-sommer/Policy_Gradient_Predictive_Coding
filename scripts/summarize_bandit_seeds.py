"""Aggregate bandit comparison results across RNG seeds."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = list(range(1, 11))

ALGO_ORDER = [
    "reinforce",
    "trpo",
    "cleanba_ppo",
    "pc_reinforce",
    "pc_actor_critic",
]

LABELS = {
    "reinforce": "REINFORCE",
    "trpo": "TRPO",
    "cleanba_ppo": "Cleanba PPO",
    "pc_reinforce": "PC-REINFORCE",
    "pc_actor_critic": "PC actor-critic",
}

SUCCESS_THRESHOLD = 0.9


def load_seed_csv(seed: int, results_root: Path):
    path = results_root / f"bandit_seed{seed}" / "pi_optimal.csv"
    if not path.exists():
        return None
    by_algo = {a: {"steps": [], "pi": []} for a in ALGO_ORDER}
    with path.open() as f:
        for row in csv.DictReader(f):
            algo = row["algo"]
            if algo not in by_algo:
                continue
            by_algo[algo]["steps"].append(int(row["env_steps"]))
            by_algo[algo]["pi"].append(float(row["pi_optimal"]))
    for algo in ALGO_ORDER:
        if by_algo[algo]["steps"]:
            order = np.argsort(by_algo[algo]["steps"])
            steps = np.array(by_algo[algo]["steps"])[order]
            pi = np.array(by_algo[algo]["pi"])[order]
            by_algo[algo] = {"steps": steps, "pi": pi}
    return by_algo


def stat_row(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else float("nan")
    median = np.median(values)
    if n > 1:
        # two-sided 95% t critical values for small n (else ~normal)
        t_table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
                   8: 2.365, 9: 2.306, 10: 2.262, 15: 2.131, 20: 2.086, 21: 2.080,
                   22: 2.074, 30: 2.042}
        t_crit = t_table.get(n - 1, 1.96)
        ci_lo = mean - t_crit * sem
        ci_hi = mean + t_crit * sem
    else:
        ci_lo = ci_hi = mean
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "median": median,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def steps_to_threshold(steps, pi, threshold=SUCCESS_THRESHOLD):
    hit = np.where(pi >= threshold)[0]
    if len(hit) == 0:
        return np.nan
    return int(steps[hit[0]])


def fmt_stats(s):
    return (
        f"{s['mean']:.3f} ± {s['sem']:.3f} "
        f"(std {s['std']:.3f}, med {s['median']:.3f}, "
        f"95% CI [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}])"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    results_root = REPO_ROOT / "results"
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "bandit_multi_seed"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = []
    curves = {a: [] for a in ALGO_ORDER}
    final_pi = {a: [] for a in ALGO_ORDER}
    avg_pi = {a: [] for a in ALGO_ORDER}
    steps90 = {a: [] for a in ALGO_ORDER}

    for seed in args.seeds:
        data = load_seed_csv(seed, results_root)
        if data is None:
            print(f"seed {seed}: missing pi_optimal.csv, skipping")
            continue
        seeds.append(seed)
        for algo in ALGO_ORDER:
            steps, pi = data[algo]["steps"], data[algo]["pi"]
            if len(pi) == 0:
                continue
            final_pi[algo].append(pi[-1])
            avg_pi[algo].append(np.mean(pi))
            steps90[algo].append(steps_to_threshold(steps, pi))
            curves[algo].append((steps, pi))

    if not seeds:
        raise SystemExit("no seed data found")

    lines = [f"bandit multi-seed summary ({len(seeds)} seeds: {seeds})", ""]

    rows = []
    for algo in ALGO_ORDER:
        fp = final_pi[algo]
        ap = avg_pi[algo]
        s90 = [x for x in steps90[algo] if not np.isnan(x)]
        fs = stat_row(fp)
        als = stat_row(ap)
        success = sum(p >= SUCCESS_THRESHOLD for p in fp) / len(fp) if fp else 0.0
        row = {
            "algo": algo,
            "label": LABELS[algo],
            "n_seeds": len(fp),
            "final_mean": fs["mean"],
            "final_std": fs["std"],
            "final_sem": fs["sem"],
            "final_median": fs["median"],
            "final_ci_lo": fs["ci_lo"],
            "final_ci_hi": fs["ci_hi"],
            "avg_mean": als["mean"],
            "avg_std": als["std"],
            "avg_sem": als["sem"],
            "avg_median": als["median"],
            "avg_ci_lo": als["ci_lo"],
            "avg_ci_hi": als["ci_hi"],
            "success_rate": success,
            "steps90_mean": np.mean(s90) if s90 else float("nan"),
            "steps90_median": np.median(s90) if s90 else float("nan"),
            "steps90_n": len(s90),
        }
        rows.append(row)
        lines.append(f"{LABELS[algo]}")
        lines.append(f"  final pi(opt): {fmt_stats(fs)}")
        lines.append(f"  avg pi(opt):   {fmt_stats(als)}")
        lines.append(
            f"  success (final >= {SUCCESS_THRESHOLD}): "
            f"{success * 100:.0f}% ({int(success * len(fp))}/{len(fp)})")
        if s90:
            lines.append(
                f"  steps to pi >= {SUCCESS_THRESHOLD}: "
                f"mean {row['steps90_mean']:.0f}, median {row['steps90_median']:.0f} "
                f"({row['steps90_n']}/{len(fp)} seeds)")
        else:
            lines.append(f"  steps to pi >= {SUCCESS_THRESHOLD}: n/a")
        lines.append("")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines))

    csv_path = out_dir / "stats.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # boxplot: final pi(opt) per algo
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data_box = [final_pi[a] for a in ALGO_ORDER if final_pi[a]]
    labels_box = [LABELS[a] for a in ALGO_ORDER if final_pi[a]]
    ax.boxplot(data_box, tick_labels=labels_box)
    ax.axhline(SUCCESS_THRESHOLD, color="gray", ls="--", lw=0.8)
    ax.set_ylabel(r"final $\pi$(optimal arm)")
    ax.set_title(f"bandit final pi(opt) over {len(seeds)} seeds")
    fig.tight_layout()
    box_path = out_dir / "final_pi_boxplot.png"
    fig.savefig(box_path, dpi=150)
    plt.close(fig)

    # mean learning curve ± SEM
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in ALGO_ORDER:
        if not curves[algo]:
            continue
        common_steps = curves[algo][0][0]
        stacked = np.stack([np.interp(common_steps, s, p) for s, p in curves[algo]])
        mean = stacked.mean(axis=0)
        sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        ax.plot(common_steps, mean, label=LABELS[algo])
        ax.fill_between(common_steps, mean - sem, mean + sem, alpha=0.2)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("env steps")
    ax.set_ylabel(r"$\pi$(optimal arm)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"mean learning curve ± SEM ({len(seeds)} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    curve_path = out_dir / "mean_learning_curve_sem.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {box_path}")
    print(f"wrote {curve_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
