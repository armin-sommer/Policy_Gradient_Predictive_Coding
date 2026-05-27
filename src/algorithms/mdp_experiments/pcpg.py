"""PCPG: thin wrapper around src/pc_algorithms/pcpg.py.

The PC math (activity relaxation, weight update from prediction errors)
lives in pc_algorithms.pcpg via the vendored JPC. This module collects a
rollout (through rollout_fn) and forwards (x, actions, returns) to pcpg_step.
"""
import jax

from pc_algorithms.pcpg import (PCPGConfig, make_eqx_template,
                                parity_check, pcpg_step)
from rollout import discounted_returns


def make_step(rollout_fn, policy, *, lr: float,
              inference_steps: int = 20, inference_lr: float = 0.3,
              output_eta: float = 0.5, gamma: float = 0.99,
              env_J=None, run_parity_check: bool = True):
    del env_J
    cfg = PCPGConfig(
        input_dim=policy.obs_dim,
        hidden=policy.hidden,
        output_dim=policy.n_actions,
        inference_steps=inference_steps,
        inference_lr=inference_lr,
        output_eta=output_eta,
        param_lr=lr,
    )
    template = make_eqx_template(cfg)
    logits_fn = policy.logits_fn

    if run_parity_check:
        dummy_params = policy.init(jax.random.PRNGKey(42))
        parity_check(logits_fn, dummy_params, template,
                     policy.fixed_obs[None])

    def step(params, opt_state, key):
        roll = rollout_fn(params, key)
        T, N = roll.actions.shape
        x = roll.obs.reshape(T * N, -1)
        actions = roll.actions.reshape(T * N)
        returns = discounted_returns(roll.rewards, gamma).reshape(T * N)
        new_params = pcpg_step(params, x, actions, returns,
                               template=template, cfg=cfg)
        mean_ep_return = roll.rewards.sum(axis=0).mean(axis=-1)
        return new_params, opt_state, {"mean_ep_return": mean_ep_return}

    return (lambda _: ()), step
