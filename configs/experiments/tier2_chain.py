"""Tier 2: 5-state Mei chain. REINFORCE / PPO / PCPG.

NPG and TRPO are omitted because their Fisher solve is built at a single
fixed observation (bandit-style); they would be incorrect on a multi-state
env without redesign. The cosine-similarity panel uses REINFORCE as the
reference direction.
"""
from algorithms.mdp_experiments import reinforce, ppo, pcpg


NAME    = "tier2_chain"
ENV     = "chain"
POLICY  = "tiny_mlp"
SEEDS   = 20
ITERS   = 200
T       = 16
N       = 64
HIDDEN  = 8
LR      = 0.05
REF     = "REINFORCE"

UPDATES = {
    "REINFORCE": (reinforce, dict(lr=LR)),
    "PPO":       (ppo,       dict(lr=LR, clip=0.2, epochs=4)),
    "PCPG":      (pcpg,      dict(lr=LR, inference_steps=20,
                                  inference_lr=0.3, output_eta=0.5)),
}
