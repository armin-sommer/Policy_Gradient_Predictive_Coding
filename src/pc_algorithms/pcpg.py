"""PCPG: policy-gradient update via Predictive Coding (JPC backend).

Wraps the JPC library (Innocenti et al. 2024, arXiv:2412.03676) so a PC-based
parameter update is callable from anywhere in this repo. The PC math itself
(activity-relaxation dynamics, prediction-error -> weight update) lives in
JPC. What is *ours* and lives here:

  1. The mapping from a policy-gradient signal to a JPC "supervised" target.
     For a sampled batch (actions a_b, rewards r_b):
         logits_ff = forward(model, x)
         pi        = softmax(logits_ff)
         g_logits  = mean_b[ (onehot(a_b) - pi) * r_b ]   # PG signal at logits
         y_pg      = logits_ff + eta * g_logits
     We hand y_pg to JPC as the output target under loss_id="mse". At the
     forward-pass activity initialisation this makes the output-layer error
     equal -eta * g_logits, i.e. proportional to the PG direction.

  2. A Flax <-> Equinox bridge. The rest of the repo uses Flax MLPs; JPC
     requires Equinox. flax_to_eqx / eqx_to_flax move weights between the
     two each step. parity_check asserts the two frameworks compute the same
     forward pass given matched weights.

Usage:
    cfg      = PCPGConfig(input_dim=1, hidden=4, output_dim=2)
    template = make_eqx_template(cfg)
    parity_check(flax_init_fn, flax_logits_fn, template, dummy_x_batched)
    new_params = pcpg_step(flax_params, x_batched, actions, rewards,
                           template=template, cfg=cfg)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jpc
import optax


@dataclass(frozen=True)
class PCPGConfig:
    input_dim: int = 1
    hidden: int = 4
    output_dim: int = 2
    inference_steps: int = 20      # T iterations of jpc.update_pc_activities
    inference_lr: float = 0.3      # optax SGD lr for activity updates
    output_eta: float = 0.5        # PG target perturbation magnitude
    param_lr: float = 0.05         # optax SGD lr for the parameter step


def make_eqx_template(cfg: PCPGConfig, key: jax.Array | None = None):
    """Build the Equinox MLP template JPC will operate on.

    Architecture: input_dim -> tanh(hidden) -> output_dim. Two layers, the
    first un-activated and the second with tanh applied to its input
    (matches jpc.make_mlp's convention).
    Weights are placeholders; flax_to_eqx overwrites them every step.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    return jpc.make_mlp(key,
                        input_dim=cfg.input_dim,
                        width=cfg.hidden,
                        depth=2,
                        output_dim=cfg.output_dim,
                        act_fn="tanh",
                        use_bias=True)


def flax_to_eqx(params: dict, template) -> Any:
    """Inject Flax Dense weights into an Equinox template MLP.

    Flax Dense stores kernels as (in, out); Equinox Linear as (out, in).
    Assumes the Flax module names its layers 'h' (hidden) and 'out' (output).
    """
    W1 = params["params"]["h"]["kernel"].T
    b1 = params["params"]["h"]["bias"]
    W2 = params["params"]["out"]["kernel"].T
    b2 = params["params"]["out"]["bias"]
    lin1 = template[0].layers[1]
    lin2 = template[1].layers[1]
    new_lin1 = eqx.tree_at(lambda l: (l.weight, l.bias), lin1, (W1, b1))
    new_lin2 = eqx.tree_at(lambda l: (l.weight, l.bias), lin2, (W2, b2))
    seq0 = eqx.tree_at(lambda s: s.layers[1], template[0], new_lin1)
    seq1 = eqx.tree_at(lambda s: s.layers[1], template[1], new_lin2)
    return [seq0, seq1]


def eqx_to_flax(model) -> dict:
    lin1 = model[0].layers[1]
    lin2 = model[1].layers[1]
    return {
        "params": {
            "h":   {"kernel": lin1.weight.T, "bias": lin1.bias},
            "out": {"kernel": lin2.weight.T, "bias": lin2.bias},
        }
    }


def eqx_forward(model, x_batched: jnp.ndarray) -> jnp.ndarray:
    """Forward through the JPC model. x shape (B, in_dim) -> (B, out_dim)."""
    def fwd_one(x):
        h = model[0](x)
        return model[1](h)
    return jax.vmap(fwd_one)(x_batched)


def parity_check(flax_logits_fn: Callable[[dict], jnp.ndarray],
                 flax_params: dict,
                 template,
                 x_batched: jnp.ndarray,
                 atol: float = 1e-5) -> None:
    """Assert Flax and Equinox forward passes agree on matched weights."""
    model = flax_to_eqx(flax_params, template)
    flax_out = flax_logits_fn(flax_params)
    eqx_out = eqx_forward(model, x_batched)[0]
    diff = jnp.max(jnp.abs(flax_out - eqx_out))
    if diff >= atol:
        raise AssertionError(
            f"Flax/Equinox forward mismatch: max|diff|={diff:.2e} (atol={atol})")
    # Round-trip.
    rt = eqx_to_flax(model)
    diff_rt = jnp.max(jnp.abs(flax_logits_fn(rt) - flax_out))
    if diff_rt >= atol:
        raise AssertionError(
            f"Flax->Eqx->Flax round-trip mismatch: max|diff|={diff_rt:.2e}")


def pcpg_step(flax_params: dict,
              x_batched: jnp.ndarray,
              actions: jnp.ndarray,
              rewards: jnp.ndarray,
              *,
              template,
              cfg: PCPGConfig) -> dict:
    """One PCPG update step. Pure function; safe under jit/vmap.

    Args:
      flax_params: Flax params dict for the policy MLP.
      x_batched:   (B, input_dim) observations the batch was sampled at.
      actions:     (B,) integer actions in [0, output_dim).
      rewards:     (B,) scalar rewards for each action.
      template:    Equinox MLP template from make_eqx_template(cfg).
      cfg:         PCPGConfig.
    Returns:
      Updated Flax params dict.
    """
    model = flax_to_eqx(flax_params, template)
    jpc_params = (model, None)

    # PG pseudo-target at the output activity (averaged over the batch).
    logits_ff = eqx_forward(model, x_batched).mean(0)        # (output_dim,)
    pi = jax.nn.softmax(logits_ff)
    onehot = jax.nn.one_hot(actions, cfg.output_dim)
    g_logits = jnp.mean((onehot - pi) * rewards[:, None], axis=0)
    y_pg = (logits_ff + cfg.output_eta * g_logits)[None]      # (1, output_dim)
    x_one = x_batched.mean(0, keepdims=True)                  # (1, input_dim)

    # Inference: equilibrate activities under MSE to y_pg.
    activities0 = jpc.init_activities_with_ffwd(model=model, input=x_one)
    inf_optim = optax.sgd(learning_rate=cfg.inference_lr)
    inf_state0 = inf_optim.init(activities0)

    def inf_body(carry, _):
        acts, st = carry
        out = jpc.update_pc_activities(
            params=jpc_params, activities=acts,
            optim=inf_optim, opt_state=st,
            output=y_pg, input=x_one)
        return (out["activities"], out["opt_state"]), None
    (acts_eq, _), _ = jax.lax.scan(
        inf_body, (activities0, inf_state0), None,
        length=cfg.inference_steps)

    # One parameter update at equilibrated activities.
    p_optim = optax.sgd(learning_rate=cfg.param_lr)
    p_state = p_optim.init(jpc_params)
    out = jpc.update_pc_params(
        params=jpc_params, activities=acts_eq,
        optim=p_optim, opt_state=p_state,
        output=y_pg, input=x_one)
    return eqx_to_flax(out["model"])
