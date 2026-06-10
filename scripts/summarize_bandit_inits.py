"""Aggregate bandit results across policy inits and RNG seeds."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from bandit_inits import load_inits, result_dir
from summarize_bandit_seeds import (
    ALGO_ORDER,
    LABELS,
    SUCCESS_THRESHOLD,
    fmt_stats,
    stat_row,
    steps_to_threshold,
)

DEFAULT_SEEDS = list(range(1, 11))


def load_init_seed_csv(init_name, seed, results_root):
    path = result_dir(init_name, seed, results_root) / "pi_optimal.csv"
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


def aggregate_init(init_name, seeds, results_root):
    curves = {a: [] for a in ALGO_ORDER}
    final_pi = {a: [] for a in ALGO_ORDER}
    avg_pi = {a: [] for a in ALGO_ORDER}
    steps90 = {a: [] for a in ALGO_ORDER}
    used_seeds = []

    for seed in seeds:
        data = load_init_seed_csv(init_name, seed, results_root)
        if data is None:
            print(f"{init_name} seed {seed}: missing pi_optimal.csv, skipping")
            continue
        used_seeds.append(seed)
        for algo in ALGO_ORDER:
            steps, pi = data[algo]["steps"], data[algo]["pi"]
            if len(pi) == 0:
                continue
            final_pi[algo].append(pi[-1])
            avg_pi[algo].append(np.mean(pi))
            steps90[algo].append(steps_to_threshold(steps, pi))
            curves[algo].append((steps, pi))

    rows = []
    lines = []
    for algo in ALGO_ORDER:
        fp = final_pi[algo]
        ap = avg_pi[algo]
        s90 = [x for x in steps90[algo] if not np.isnan(x)]
        fs = stat_row(fp)
        als = stat_row(ap)
        success = sum(p >= SUCCESS_THRESHOLD for p in fp) / len(fp) if fp else 0.0
        row = {
            "init": init_name,
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

    return used_seeds, rows, lines, curves, final_pi


def main():
    parser = argparse.ArgumentParser()
    inits = load_inits()
    parser.add_argument("--inits", nargs="*", default=list(inits.keys()),
                        choices=list(inits.keys()))
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    results_root = REPO_ROOT / "results"
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "bandit_multi_init"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    cross_lines = [
        f"bandit multi-init summary (seeds {args.seeds})",
        "",
        "init presets (pi0 = sigmoid(b0-b1), arm 0 optimal):",
    ]
    for name, spec in inits.items():
        if name in args.inits:
            cross_lines.append(
                f"  {name}: bias={spec['logit_bias']}  pi0={spec['pi_opt']:.4f}")
    cross_lines.append("")

    per_init_summaries = {}

    for init_name in args.inits:
        used_seeds, rows, lines, curves, final_pi = aggregate_init(
            init_name, args.seeds, results_root)
        if not used_seeds:
            print(f"{init_name}: no seed data, skipping")
            continue
        all_rows.extend(rows)
        pi0 = inits[init_name]["pi_opt"]
        header = (
            f"=== {init_name} (pi0={pi0:.4f}, {len(used_seeds)} seeds: "
            f"{used_seeds}) ===")
        per_init_summaries[init_name] = "\n".join([header, ""] + lines)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for algo in ALGO_ORDER:
            if not curves[algo]:
                continue
            common_steps = curves[algo][0][0]
            stacked = np.stack(
                [np.interp(common_steps, s, p) for s, p in curves[algo]])
            mean = stacked.mean(axis=0)
            sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
            ax.plot(common_steps, mean, label=LABELS[algo])
            ax.fill_between(common_steps, mean - sem, mean + sem, alpha=0.2)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.axhline(pi0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("env steps")
        ax.set_ylabel(r"$\pi$(optimal arm)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{init_name} (pi0={pi0:.3f}), mean ± SEM ({len(used_seeds)} seeds)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        curve_path = out_dir / f"mean_learning_curve_{init_name}.png"
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        data_box = [final_pi[a] for a in ALGO_ORDER if final_pi[a]]
        labels_box = [LABELS[a] for a in ALGO_ORDER if final_pi[a]]
        ax.boxplot(data_box, tick_labels=labels_box)
        ax.axhline(SUCCESS_THRESHOLD, color="gray", ls="--", lw=0.8)
        ax.set_ylabel(r"final $\pi$(optimal arm)")
        ax.set_title(f"{init_name}: final pi(opt) over {len(used_seeds)} seeds")
        fig.tight_layout()
        box_path = out_dir / f"final_pi_boxplot_{init_name}.png"
        fig.savefig(box_path, dpi=150)
        plt.close(fig)

    if not all_rows:
        raise SystemExit("no init data found")

    summary_path = out_dir / "summary.txt"
    full_text = cross_lines + [""]
    for init_name in args.inits:
        if init_name in per_init_summaries:
            full_text.append(per_init_summaries[init_name])
            full_text.append("")
    summary_path.write_text("\n".join(full_text))

    csv_path = out_dir / "stats.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # cross-init comparison: final pi(opt) mean per algo × init
    cross_rows = []
    for init_name in args.inits:
        init_rows = [r for r in all_rows if r["init"] == init_name]
        for r in init_rows:
            cross_rows.append({
                "init": init_name,
                "pi0": inits[init_name]["pi_opt"],
                "algo": r["algo"],
                "label": r["label"],
                "final_mean": r["final_mean"],
                "final_sem": r["final_sem"],
                "success_rate": r["success_rate"],
            })

    cross_csv = out_dir / "cross_init_comparison.csv"
    with cross_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cross_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cross_rows)

    fig, axes = plt.subplots(1, len(args.inits), figsize=(3.5 * len(args.inits), 4.5),
                             sharey=True, squeeze=False)
    for i, init_name in enumerate(args.inits):
        ax = axes[0, i]
        init_rows = [r for r in all_rows if r["init"] == init_name]
        if not init_rows:
            continue
        means = [r["final_mean"] for r in init_rows]
        sems = [r["final_sem"] for r in init_rows]
        labels = [r["label"] for r in init_rows]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sems, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        pi0 = inits[init_name]["pi_opt"]
        ax.set_title(f"{init_name}\npi0={pi0:.3f}")
        ax.axhline(SUCCESS_THRESHOLD, color="gray", ls="--", lw=0.8)
    axes[0, 0].set_ylabel(r"final $\pi$(optimal arm)")
    fig.suptitle(f"final pi(opt) by init ({len(args.seeds)} seeds each)")
    fig.tight_layout()
    cross_plot = out_dir / "cross_init_final_pi.png"
    fig.savefig(cross_plot, dpi=150)
    plt.close(fig)

    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {cross_csv}")
    print(f"wrote {cross_plot}")
    print()
    print("\n".join(full_text))


if __name__ == "__main__":
    main()
