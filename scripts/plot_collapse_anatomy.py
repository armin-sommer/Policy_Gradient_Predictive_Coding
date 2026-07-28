"""Per-config collapse-anatomy panels: seeds overlaid, one figure per config.

The analyzer's learning_curve.png answers "which config wins". This answers
"*how* did a seed die" -- the diagnostics that separate a run that collapses from
one that survives, on a shared x-axis so the ordering of events is visible.

Panels are chosen from whatever the logs actually contain, so it works on older
runs (no grad-norm percentiles) and on the gradclip probe (which has them).

Usage:
    python scripts/plot_collapse_anatomy.py --results-dir results/gradclip_probe
    python scripts/plot_collapse_anatomy.py --results-dir results/knob_fill_smin_vlr

Writes collapse_anatomy_<config>.png next to the logs.
"""

import argparse
import ast
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED_COLORS = {1: "#2c7fb2", 2: "#31a354", 3: "#d62728",
               4: "#f0a03a", 5: "#8856a7"}

# (log key, panel title, source) -- source "eval" uses the eval stream.
CANDIDATE_PANELS = [
    ("eval/mean_score",              "Eval return (deterministic)", "eval"),
    ("training/mean_reward",         "Train mean_reward (normalized)", "train"),
    ("diag/value_explained_var",     "Value explained_var", "train"),
    ("diag/policy_kl_max",           "Policy KL max", "train"),
    ("diag/log_std_mean",            "log_std_mean (entropy)", "train"),
    ("diag/pretanh_sat_frac",        "pre-tanh saturation frac", "train"),
    ("diag/policy_grad_norm_p99",    "Grad-norm p99 (cumulative)", "train"),
    ("diag/policy_grad_norm_bind_rate", "Clip bind rate", "train"),
]


def parse_log(path):
    train, evals, last = [], [], 0
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        s = re.sub(r"np\.float64\(([^)]*)\)", r"\1", line)
        try:
            d = ast.literal_eval(s)
        except Exception:
            continue
        if "training/total_steps" in d:
            last = d["training/total_steps"]
            train.append(d)
        elif "eval/mean_score" in d:
            evals.append((last, float(d["eval/mean_score"])))
    return train, evals


def col(train, key):
    return np.array([t.get(key, np.nan) for t in train], dtype=float)


def plot_config(cfg_dir, out_png, title):
    runs = {}
    for fn in sorted(os.listdir(cfg_dir)):
        m = re.fullmatch(r"seed_(\d+)\.log", fn)
        if not m:
            continue
        train, evals = parse_log(os.path.join(cfg_dir, fn))
        if train:
            runs[int(m.group(1))] = (train, evals)
    if not runs:
        return False

    any_train = next(iter(runs.values()))[0]
    panels = [p for p in CANDIDATE_PANELS
              if p[2] == "eval" or not np.all(np.isnan(col(any_train, p[0])))]
    # drop an all-zero bind-rate panel (clip off -> nothing to show)
    panels = [p for p in panels if not (
        p[0].endswith("bind_rate")
        and np.nanmax(np.abs(col(any_train, p[0]))) == 0)]

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow),
                             squeeze=False)
    fig.suptitle(title, fontsize=13)

    for ax, (key, ptitle, src) in zip(axes.ravel(), panels):
        for seed, (train, evals) in sorted(runs.items()):
            color = SEED_COLORS.get(seed, None)
            if src == "eval":
                if not evals:
                    continue
                xs = np.array([e[0] for e in evals]) / 1e3
                ys = np.array([e[1] for e in evals])
            else:
                xs = col(train, "training/total_steps") / 1e3
                ys = col(train, key)
            ax.plot(xs, ys, color=color, lw=1.6, alpha=0.9, label=f"seed {seed}")
        ax.set_title(ptitle, fontsize=10)
        ax.set_xlabel("step (k)")
        ax.grid(alpha=0.25)
        if key in ("eval/mean_score", "training/mean_reward",
                   "diag/value_explained_var"):
            ax.axhline(0, color="k", lw=0.6, ls=":")
        if key.endswith("bind_rate"):
            ax.set_ylabel("fraction of updates")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()

    base = args.results_dir
    made = 0
    for cfg in sorted(os.listdir(base)):
        cfg_dir = os.path.join(base, cfg)
        if not os.path.isdir(cfg_dir):
            continue
        short = cfg.replace("halfcheetah_pc_actor_critic_", "")
        out = os.path.join(base, f"collapse_anatomy_{short}.png")
        if plot_config(cfg_dir, out, f"Collapse anatomy — {short}"):
            print("wrote", out)
            made += 1
    if not made:
        print(f"no seed logs found under {base}")


if __name__ == "__main__":
    main()
