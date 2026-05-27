"""Infrastructure for comparing policy-gradient update rules on small MDPs.

Layout:
  envs/      one module per MDP, each exposing an Env NamedTuple.
  updates/   one module per update rule, each exposing make_step(env, policy, **hp).
  policies.py  tiny MLP policy factory.
  runner.py    seed-vmapped, step-scanned training loop, identical for all algos.
  plotting.py  the four-panel comparison figure.

A driver script picks one env + a dict of updates, then calls runner.run_all
and plotting.four_panel. See scripts/experiments/tier1_bandit.py.
"""
from mdp_experiments._base import Env, Policy, Log
