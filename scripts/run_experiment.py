"""Generic experiment driver. Reads a Python config under configs/experiments/
and runs it.

Usage:
    python scripts/run_experiment.py configs/experiments/tier1_bandit.py

A config module must define:
    NAME    str               unique name; outputs go under runs/<NAME>/
    ENV     module            with .ENV: Env and optional .OBS for fixed_obs
    SEEDS   int
    STEPS   int
    HIDDEN  int               hidden width of the TinyPolicy
    REF     str               algorithm key used as cosine reference in plot
    UPDATES dict[str, (module, dict)]   each module exposes make_step(...)
"""
import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pc_algorithms  # noqa: F401  (vendored-JPC sys.path wire-up)
from mdp_experiments.policies import make_tiny_policy
from mdp_experiments.runner import run_all
from mdp_experiments.plotting import four_panel


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save_results(results, out_dir: Path):
    """Pickle raw Log arrays (converted to numpy) for later re-plotting."""
    payload = {name: {f: np.asarray(getattr(log, f)) for f in log._fields}
               for name, log in results.items()}
    with open(out_dir / "results.pkl", "wb") as f:
        pickle.dump(payload, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="Path to a config .py file")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    env = cfg.ENV.ENV
    fixed_obs = getattr(cfg.ENV, "OBS", None)
    policy = make_tiny_policy(obs_dim=env.OBS_DIM, hidden=cfg.HIDDEN,
                              n_actions=env.N_ACTIONS,
                              fixed_obs=fixed_obs)

    updates = {name: mod.make_step(env, policy, **hp)
               for name, (mod, hp) in cfg.UPDATES.items()}

    results = run_all(updates, policy, env,
                      num_seeds=cfg.SEEDS, num_steps=cfg.STEPS)

    out_dir = REPO_ROOT / "runs" / cfg.NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = four_panel(results, ref=cfg.REF,
                         out_path=str(out_dir / "figure.png"))
    _save_results(results, out_dir)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved {out_dir / 'figure.png'}")
    for name, val in summary.items():
        print(f"  late-mean cos({name}, {cfg.REF}) = {val:+.3f}")


if __name__ == "__main__":
    main()
