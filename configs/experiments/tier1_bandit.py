"""Tier 1: Mei one-state two-action MDP. REINFORCE / PPO / TRPO / PCPG vs NPG.

To add an experiment: copy this file, change the env import and/or the
hyperparameters. Outputs land under runs/<NAME>/.
"""
from mdp_experiments.envs import bandit as ENV
from mdp_experiments.updates import reinforce, ppo, trpo, npg, pcpg


NAME    = "simple_bandit"
SEEDS   = 50
STEPS   = 100
HIDDEN  = 4
LR      = 0.05
REF     = "NPG"          # algorithm whose update direction is the cosine reference

UPDATES = {
    "REINFORCE": (reinforce, dict(lr=LR)),
    "PPO":       (ppo,       dict(lr=LR, clip=0.2, epochs=4)),
    "TRPO":      (trpo,      dict(delta=0.01)),
    "PCPG":      (pcpg,      dict(lr=LR, inference_steps=20,
                                  inference_lr=0.3, output_eta=0.5)),
    "NPG":       (npg,       dict(lr=LR)),
}
