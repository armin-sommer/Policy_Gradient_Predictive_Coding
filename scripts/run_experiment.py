"""Generic experiment driver.

Usage:
    python scripts/run_experiment.py configs/experiments/tier1_bandit.py

A config module must define:
    NAME    str                    unique name; outputs go under runs/<NAME>/
    ENV     str                    key into envs._registry
    POLICY  str                    key into policies._registry
    SEEDS   int
    ITERS   int                    number of update iterations
    T       int                    rollout length per iteration
    N       int                    parallel envs per iteration
    HIDDEN  int                    hidden width of the policy network
    REF     str                    algorithm key used as cosine reference
    UPDATES dict[str, (module, dict)]   each module exposes make_step(...)
"""
import argparse
import importlib.util
import json
import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pc_algorithms  # noqa: F401  (vendored-JPC sys.path wire-up)
from envs._registry import make_env
from policies._registry import make_policy
from rollout.scan_rollout import collect_rollout
from runner import run_all
from plotting import four_panel


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save_results(results, out_dir: Path):
    payload = {name: {f: np.asarray(getattr(log, f)) for f in log._fields}
               for name, log in results.items()}
    with open(out_dir / "results.pkl", "wb") as f:
        pickle.dump(payload, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="Path to a config .py file")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    env, fixed_obs = make_env(cfg.ENV)
    policy = make_policy(cfg.POLICY, env,
                         hidden=cfg.HIDDEN, fixed_obs=fixed_obs)
    rollout_fn = partial(collect_rollout, env=env, policy=policy,
                         T=cfg.T, N=cfg.N)

    updates = {name: mod.make_step(rollout_fn, policy, env_J=env.J, **hp)
               for name, (mod, hp) in cfg.UPDATES.items()}

    results = run_all(updates, policy, J_fn=env.J,
                      num_seeds=cfg.SEEDS, num_iters=cfg.ITERS)

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
