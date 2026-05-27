"""PCPG: thin wrapper around src/pc_algorithms/pcpg.py.

The PC math (activity relaxation, weight update from prediction errors)
lives in pc_algorithms.pcpg via the vendored JPC. This module just samples
a batch from the env and forwards everything to pcpg_step.
"""
from pc_algorithms.pcpg import (PCPGConfig, make_eqx_template,
                                parity_check, pcpg_step)


def make_step(env, policy, *, lr: float, inference_steps: int = 20,
              inference_lr: float = 0.3, output_eta: float = 0.5,
              batch_size: int = 64, run_parity_check: bool = True):
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
    sample = env.sample_batch

    if run_parity_check:
        import jax
        dummy_params = policy.init(jax.random.PRNGKey(42))
        parity_check(logits_fn, dummy_params, template,
                     policy.fixed_obs[None])

    def step(params, key):
        x, actions, rewards = sample(params, key, logits_fn=logits_fn,
                                     batch_size=batch_size)
        return pcpg_step(params, x, actions, rewards,
                         template=template, cfg=cfg)

    return step
