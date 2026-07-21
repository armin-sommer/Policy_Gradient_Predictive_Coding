"""Parse PCPG / backprop benchmark logs into per-run metrics, summary tables, and
diagnostic plots.

Reads a results tree written by run_pcpg_benchmark_matrix.py:

    <results-dir>/<config_name>/seed_<N>.log      (+ optional meta.json per folder)

and falls back to a flat layout of  <results-dir>/*_seed*.log  files.

Writes into <results-dir>:
    per_run.csv          one row per run (returns, AUC, collapse, diagnostics, walltime)
    summary_all.csv      aggregated per config (mean +/- std, collapse count / N)
    SUMMARY.md           the same tables as markdown
    learning_curve.png   eval return vs env-steps, mean +/- SEM per config
    diagnostic_plots.png diag/* vs env-steps, mean per config

Collapse rule (docs/PCPG_BENCHMARK_PLAN.md sec 5): a run is `collapsed` if it first
reaches the viability threshold and then eval return stays below COLLAPSE_FRAC x its
best for >= COLLAPSE_RUN consecutive eval checkpoints. `severe` if best>=viability and
final<0. degradation_ratio = final / best.

    python scripts/analyze_pcpg_logs.py --results-dir results/benchmark_halfcheetah
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

NUM = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"
VIABILITY = 300.0        # eval return that counts as "learned" (HalfCheetah dev)
COLLAPSE_FRAC = 0.30     # collapse if return drops below this fraction of best
COLLAPSE_RUN = 3         # ... for this many consecutive eval checkpoints

# training-dict diagnostics to track as (step, value) series
TRAIN_KEYS = [
    "diag/policy_drift_max", "diag/mu_target_mag_max", "diag/value_explained_var",
    "training/value_pc_loss", "training/policy_pc_loss", "diag/mu_abs_mean",
    "diag/pretanh_sat_frac", "diag/log_std_mean", "diag/log_std_min",
    "diag/frac_std_at_min", "diag/frac_std_at_max",
    "diag/policy_kl_max", "diag/policy_kl_mean",
]
ALGO_TOKENS = ["pc_actor_critic", "pc_reinforce", "ppo", "trpo", "reinforce"]


def _find(line, key):
    m = re.search(rf"'{re.escape(key)}':\s*(?:np\.\w+\()?({NUM})", line)
    return float(m.group(1)) if m else None


def parse_run(path):
    """Return eval trajectory, diagnostic series, walltime, and step count."""
    evals = []                                   # (step, eval/mean_score)
    train = {k: [] for k in TRAIN_KEYS}          # key -> [(step, value), ...]
    last_step, walltime, duration = 0, None, None
    for line in path.read_text().splitlines():
        s = _find(line, "training/total_steps")
        if s is not None:
            last_step = int(s)
            wt = _find(line, "training/walltime")
            if wt is not None:
                walltime = wt
            for k in TRAIN_KEYS:
                v = _find(line, k)
                if v is not None:
                    train[k].append((last_step, v))
        ev = _find(line, "eval/mean_score")
        if ev is not None:
            evals.append((last_step, ev))
        final_ev = _find(line, "final_eval/mean_score")
        if final_ev is not None:
            evals.append((last_step, final_ev))
        if "TRAINING END" in line and "duration:" in line:
            m = re.search(NUM, line.split("duration:")[-1])
            if m:
                duration = float(m.group(0))
    return dict(evals=evals, train=train, total_steps=last_step,
                walltime=walltime or duration, duration=duration)


def collapse_flags(scores):
    """(collapsed, severe) per the plan's rule: only after the run first reaches
    viability, count consecutive eval points below COLLAPSE_FRAC x the running best."""
    if not scores:
        return False, False
    best = max(scores)
    severe = best >= VIABILITY and scores[-1] < 0
    running_best, reached, run = float("-inf"), False, 0
    for x in scores:
        running_best = max(running_best, x)
        reached = reached or running_best >= VIABILITY
        if reached and x < COLLAPSE_FRAC * running_best:
            run += 1
            if run >= COLLAPSE_RUN:
                return True, severe
        else:
            run = 0
    return False, severe


def auc(steps, scores):
    """Normalized area under the eval curve = time-averaged return over training."""
    if len(scores) < 2:
        return float("nan")
    steps, scores = np.asarray(steps, float), np.asarray(scores, float)
    span = steps[-1] - steps[0]
    if span <= 0:
        return float(np.mean(scores))
    area = np.sum((scores[:-1] + scores[1:]) / 2 * np.diff(steps))  # trapezoid
    return float(area / span)


def _series_stats(series):
    if not series:
        return {}
    v = np.array([x[1] for x in series], float)
    n = max(1, len(v) // 10)
    return dict(max=float(v.max()), p95=float(np.percentile(v, 95)),
                final=float(v[-1]), mean=float(v.mean()), min=float(v.min()),
                trend=float(v[-n:].mean() / (v[:n].mean() + 1e-12)))


def run_row(path, env, algo, config_name, seed, meta):
    r = parse_run(path)
    steps = [s for s, _ in r["evals"]]
    scores = [v for _, v in r["evals"]]
    best = max(scores) if scores else float("nan")
    final = scores[-1] if scores else float("nan")
    collapsed, severe = collapse_flags(scores)
    d = {k: _series_stats(r["train"][k]) for k in TRAIN_KEYS}
    return {
        "env": env, "algo": algo, "config_name": config_name, "seed": seed,
        "final_return": round(final, 2), "best_return": round(best, 2),
        "auc_return": round(auc(steps, scores), 2),
        "collapse": int(collapsed), "severe_collapse": int(severe),
        "degradation_ratio": round(final / best, 3) if best not in (0, float("nan")) else float("nan"),
        "walltime_s": round(meta.get("walltime_s") or r["walltime"] or float("nan"), 1),
        "total_steps": r["total_steps"],
        "drift_max": d["diag/policy_drift_max"].get("max"),
        "drift_p95": d["diag/policy_drift_max"].get("p95"),
        "drift_final": d["diag/policy_drift_max"].get("final"),
        "kl_max": d["diag/policy_kl_max"].get("max"),
        "kl_p95": d["diag/policy_kl_max"].get("p95"),
        "kl_final": d["diag/policy_kl_max"].get("final"),
        "mu_target_max": d["diag/mu_target_mag_max"].get("max"),
        "mu_target_p95": d["diag/mu_target_mag_max"].get("p95"),
        "value_ev_mean": d["diag/value_explained_var"].get("mean"),
        "value_ev_final": d["diag/value_explained_var"].get("final"),
        "value_ev_min": d["diag/value_explained_var"].get("min"),
        "value_pc_loss_final": d["training/value_pc_loss"].get("final"),
        "value_pc_loss_trend": d["training/value_pc_loss"].get("trend"),
        "pretanh_sat_max": d["diag/pretanh_sat_frac"].get("max"),
        "pretanh_sat_final": d["diag/pretanh_sat_frac"].get("final"),
        "log_std_mean_final": d["diag/log_std_mean"].get("final"),
        "frac_std_at_min_final": d["diag/frac_std_at_min"].get("final"),
        "frac_std_at_max_final": d["diag/frac_std_at_max"].get("final"),
        "_evals": r["evals"], "_train": r["train"],   # kept for plotting; dropped from CSV
    }


def discover(results_dir):
    """Yield (log_path, config_name, seed, meta) over benchmark or flat layouts."""
    subdirs = [p for p in results_dir.iterdir() if p.is_dir()]
    if any(sub.glob("seed_*.log") for sub in subdirs):
        for sub in sorted(subdirs):
            meta = {}
            mp = sub / "meta.json"
            if mp.exists():
                meta = json.loads(mp.read_text())
            for log in sorted(sub.glob("seed_*.log")):
                seed = int(re.search(r"seed_(\d+)", log.name).group(1))
                yield log, sub.name, seed, meta
    else:                                              # flat *_seed*.log fallback
        for log in sorted(results_dir.glob("*_seed*.log")):
            m = re.search(r"(.+)_seed_?(\d+)\.log$", log.name)
            if m:
                yield log, m.group(1), int(m.group(2)), {}


def _algo_of(config_name, meta):
    if meta.get("algo"):
        return meta["algo"]
    for tok in ALGO_TOKENS:
        if tok in config_name:
            return tok
    return "?"


def aggregate(rows):
    """config -> aggregated stats over seeds."""
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config_name"]].append(r)
    out = []
    for cfg, rs in sorted(by_cfg.items()):
        def ms(key):
            v = np.array([r[key] for r in rs if r[key] == r[key]], float)  # drop NaN
            return (float(v.mean()), float(v.std())) if v.size else (float("nan"), 0.0)
        fm, fs = ms("final_return")
        bm, bs = ms("best_return")
        am, as_ = ms("auc_return")
        out.append(dict(
            config_name=cfg, algo=rs[0]["algo"], n=len(rs),
            final_mean=fm, final_std=fs, best_mean=bm, best_std=bs,
            auc_mean=am, auc_std=as_,
            collapse_count=sum(r["collapse"] for r in rs),
            walltime_mean=ms("walltime_s")[0]))
    return out


def write_csv(rows, path):
    cols = [k for k in rows[0] if not k.startswith("_")]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join("" if r[c] is None else str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def write_summary_md(agg, path):
    out = ["# PCPG benchmark summary\n",
           "| config | algo | n | final (mean±std) | best (mean±std) | AUC (mean±std) | collapse | walltime |",
           "|---|---|---|---|---|---|---|---|"]
    for a in agg:
        out.append("| {config_name} | {algo} | {n} | {fm:.0f} ± {fs:.0f} | "
                   "{bm:.0f} ± {bs:.0f} | {am:.0f} ± {as_:.0f} | {cc}/{n} | {wt:.0f}s |".format(
                       fm=a["final_mean"], fs=a["final_std"], bm=a["best_mean"], bs=a["best_std"],
                       am=a["auc_mean"], as_=a["auc_std"], cc=a["collapse_count"],
                       wt=a["walltime_mean"], **a))
    path.write_text("\n".join(out) + "\n")
    return "\n".join(out)


def _mean_curve(runs):
    """Align seeds to the shortest series; return (steps, mean)."""
    series = [runs_i for runs_i in runs if runs_i]
    if not series:
        return None, None
    n = min(len(s) for s in series)
    steps = np.array([x[0] for x in series[0][:n]], float)
    mat = np.vstack([[x[1] for x in s[:n]] for s in series])
    return steps, mat.mean(axis=0)


def plot(rows, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> skipping plots")
        return
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config_name"]].append(r)

    # learning curves (eval return)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = 0
    for cfg, rs in sorted(by_cfg.items()):
        # Draw faint individual seed curves so a config with uneven eval counts
        # still leaves visible evidence, then overlay the common-prefix mean.
        for r in rs:
            if not r["_evals"]:
                continue
            s = np.array([x[0] for x in r["_evals"]], float)
            v = np.array([x[1] for x in r["_evals"]], float)
            ax.plot(s, v, alpha=0.22, linewidth=0.9)
            plotted += 1
        steps, mean = _mean_curve([r["_evals"] for r in rs])
        if steps is not None:
            ax.plot(steps, mean, label=f"{cfg} (n={len(rs)})", linewidth=2)
            plotted += 1
    ax.set(title="eval return vs steps", xlabel="env steps", ylabel="eval/mean_score")
    ax.grid(alpha=0.3); ax.legend(fontsize=7); fig.tight_layout()
    if plotted == 0:
        ax.text(0.5, 0.5, "No eval points parsed from logs",
                transform=ax.transAxes, ha="center", va="center")
        print("warning: no eval points parsed; learning_curve.png will be empty")
    fig.savefig(results_dir / "learning_curve.png", dpi=130); plt.close(fig)

    # diagnostic panels
    panels = ["diag/policy_kl_max", "diag/policy_drift_max", "diag/mu_target_mag_max",
              "diag/value_explained_var", "diag/pretanh_sat_frac", "diag/log_std_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, key in zip(axes.flat, panels):
        for cfg, rs in sorted(by_cfg.items()):
            steps, mean = _mean_curve([r["_train"][key] for r in rs])
            if steps is not None:
                ax.plot(steps, mean, label=cfg, linewidth=1.5)
        ax.set(title=key, xlabel="env steps"); ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(results_dir / "diagnostic_plots.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    env = results_dir.name.replace("benchmark_", "") or "?"

    rows = []
    for log, cfg, seed, meta in discover(results_dir):
        rows.append(run_row(log, meta.get("env", env), _algo_of(cfg, meta), cfg, seed, meta))
    if not rows:
        print(f"no runs found under {results_dir}")
        return

    write_csv(rows, results_dir / "per_run.csv")
    agg = aggregate(rows)
    md = write_summary_md(agg, results_dir / "SUMMARY.md")
    plot(rows, results_dir)

    print(md)
    print(f"\nwrote per_run.csv, SUMMARY.md, learning_curve.png, diagnostic_plots.png "
          f"-> {results_dir}")
    # summary_all.csv (aggregated) alongside the per-run table
    keys = [k for k in agg[0]]
    (results_dir / "summary_all.csv").write_text(
        ",".join(keys) + "\n" + "\n".join(",".join(str(a[k]) for k in keys) for a in agg) + "\n")


if __name__ == "__main__":
    main()
