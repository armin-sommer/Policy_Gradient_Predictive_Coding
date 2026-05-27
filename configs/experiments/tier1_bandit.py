"""Tier 1: Mei one-state two-action MDP. REINFORCE / PPO / TRPO / PCPG vs NPG.

To add an experiment: copy this file, change ENV/POLICY/hyperparameters.
Outputs land under runs/<NAME>/.

Conventions:
    ITERS   number of update iterations
    T       rollout length (steps per env per iteration)
    N       number of parallel envs
"""
from algorithms.mdp_experiments import reinforce, ppo, trpo, npg, pcpg


NAME    = "simple_bandit"
ENV     = "bandit"
POLICY  = "tiny_mlp"
SEEDS   = 50
ITERS   = 100
T       = 1
N       = 64
HIDDEN  = 4
LR      = 0.05
REF     = "NPG"

UPDATES = {
    "REINFORCE": (reinforce, dict(lr=LR)),
    "PPO":       (ppo,       dict(lr=LR, clip=0.2, epochs=4)),
    "TRPO":      (trpo,      dict(delta=0.01)),
    "PCPG":      (pcpg,      dict(lr=LR, inference_steps=20,
                                  inference_lr=0.3, output_eta=0.5)),
    "NPG":       (npg,       dict(lr=LR)),
}
