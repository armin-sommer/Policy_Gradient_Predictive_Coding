"""Tier 1: REINFORCE / PPO / TRPO / PCPG vs closed-form NPG on the
Mei-style one-state two-action MDP.

To swap the environment: change `from mdp_experiments.envs import bandit as env`
to a different module under mdp_experiments/envs/. The five update modules
and the runner/plotting code do not change.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Make src/ importable (same pattern as scripts/run_train.py).
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pc_algorithms  # triggers vendored-JPC sys.path wiring

from mdp_experiments.envs import bandit as env_mod
from mdp_experiments.policies import make_tiny_policy
from mdp_experiments.runner import run_all
from mdp_experiments.plotting import four_panel
from mdp_experiments.updates import reinforce, ppo, trpo, npg, pcpg


NUM_SEEDS = 50
NUM_STEPS = 100
LR = 0.05


def main():
    env = env_mod.ENV
    policy = make_tiny_policy(obs_dim=env.OBS_DIM, hidden=4,
                              n_actions=env.N_ACTIONS,
                              fixed_obs=env_mod.OBS)

    updates = {
        "REINFORCE": reinforce.make_step(env, policy, lr=LR),
        "PPO":       ppo.make_step(env, policy, lr=LR, clip=0.2, epochs=4),
        "TRPO":      trpo.make_step(env, policy, delta=0.01),
        "PCPG":      pcpg.make_step(env, policy, lr=LR,
                                    inference_steps=20, inference_lr=0.3,
                                    output_eta=0.5),
        "NPG":       npg.make_step(env, policy, lr=LR),
    }

    results = run_all(updates, policy, env,
                      num_seeds=NUM_SEEDS, num_steps=NUM_STEPS)

    out = os.path.join(os.path.dirname(__file__), "tier1_bandit.png")
    summary = four_panel(results, ref="NPG", out_path=out)
    print(f"saved {out}")
    for name, val in summary.items():
        print(f"  late-mean cos({name}, NPG) = {val:+.3f}")


if __name__ == "__main__":
    main()
