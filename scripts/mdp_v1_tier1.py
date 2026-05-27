"""Tier 1: one-state two-action MDP, comparing REINFORCE / PPO / TRPO / PCPG
against closed-form NPG on identical tiny MLP policies.

MDP (Mei et al. 2020 style): single constant observation x = [1.0], two actions
with deterministic rewards r(a=0)=0, r(a=1)=1. J(theta) = pi_theta(a=1).

Policy: tiny MLP   x (1) -> hidden (H=4, tanh) -> logits (2) -> Categorical.
REINFORCE / PPO / TRPO / NPG run on a Flax MLP. PCPG runs on an Equinox MLP
built by jpc.make_mlp with identical shape; weights are copied from the same
Flax init each step (forward-pass parity is asserted at startup).

PCPG (JPC backend, Innocenti et al. 2024):
  Output pseudo-target for predictive coding is the policy-gradient nudge to
  the logits:
      logits_ff = forward(model, x)                      # FF output activity
      pi        = softmax(logits_ff)
      g_logits  = mean_b[ (onehot(a_b) - pi) * r_b ]     # PG signal
      y_pg      = logits_ff + eta * g_logits
  Inference: T steps of jpc.update_pc_activities under an SGD optimizer
  (lr = gamma) with output=y_pg, loss_id="mse". The MSE objective makes the
  output-layer error equal to (activities[-1] - y_pg), which equals
  -eta * g_logits when activities are still at the FF init. Then one
  jpc.update_pc_params step (lr = alpha) propagates the equilibrated
  activities to weight updates.

Run:  python scripts/mdp_v1_tier1.py
Outputs: scripts/mdp_v1_tier1.png
"""
from __future__ import annotations
import functools
import os
import sys
from pathlib import Path
from typing import NamedTuple

# Make src/ importable (matches scripts/run_train.py).
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn

from pc_algorithms.pcpg import (
    PCPGConfig, make_eqx_template, parity_check, pcpg_step as _pcpg_step)


# --- config ----------------------------------------------------------------
NUM_SEEDS    = 50
NUM_STEPS    = 100
BATCH_SIZE   = 64       # samples per update step
HIDDEN       = 4
LR           = 0.05     # generic learning rate for REINFORCE / PCPG outer
PPO_CLIP     = 0.2
PPO_EPOCHS   = 4
TRPO_DELTA   = 0.01
TRPO_BACKTRACK = 10
TRPO_DECAY   = 0.5
PC_T         = 20       # inference iterations
PC_GAMMA     = 0.3      # inference step size
PC_ETA       = 0.5      # output target perturbation magnitude
DAMPING      = 1e-3     # for inverting the Fisher
OBS          = jnp.array([1.0])
REWARDS      = jnp.array([0.0, 1.0])


# --- policy ---------------------------------------------------------------
class TinyPolicy(nn.Module):
    hidden: int = HIDDEN

    @nn.compact
    def __call__(self, x):
        h = nn.tanh(nn.Dense(self.hidden, name="h")(x))
        return nn.Dense(2, name="out")(h)


policy = TinyPolicy()


def init_params(key):
    return policy.init(key, OBS)


def logits_fn(params):
    return policy.apply(params, OBS)


def pi_a1(params):
    return jax.nn.softmax(logits_fn(params))[1]


def J(params):
    p = jax.nn.softmax(logits_fn(params))
    return jnp.dot(p, REWARDS)


def kl(p_old, p_new):
    return jnp.sum(p_old * (jnp.log(p_old + 1e-12) - jnp.log(p_new + 1e-12)))


def flatten(params):
    leaves, _ = jax.tree_util.tree_flatten(params)
    return jnp.concatenate([l.ravel() for l in leaves])


# --- sampling -------------------------------------------------------------
def sample_batch(params, key):
    logits = logits_fn(params)
    dist = distrax.Categorical(logits=logits)
    actions = dist.sample(seed=key, sample_shape=(BATCH_SIZE,))
    rewards = REWARDS[actions]
    return actions, rewards


# --- update rules ---------------------------------------------------------
def reinforce_grad(params, actions, rewards):
    def loss(p):
        logp = distrax.Categorical(logits=logits_fn(p)).log_prob(actions)
        return -jnp.mean(logp * rewards)
    return jax.grad(loss)(params)


def reinforce_step(params, key):
    actions, rewards = sample_batch(params, key)
    g = reinforce_grad(params, actions, rewards)
    return jax.tree_util.tree_map(lambda p, dp: p - LR * dp, params, g)


def ppo_step(params, key):
    actions, rewards = sample_batch(params, key)
    old_logp = distrax.Categorical(logits=logits_fn(params)).log_prob(actions)

    def loss(p):
        logp = distrax.Categorical(logits=logits_fn(p)).log_prob(actions)
        ratio = jnp.exp(logp - old_logp)
        unclipped = ratio * rewards
        clipped = jnp.clip(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * rewards
        return -jnp.mean(jnp.minimum(unclipped, clipped))

    def body(carry, _):
        p = carry
        g = jax.grad(loss)(p)
        return jax.tree_util.tree_map(lambda x, dx: x - LR * dx, p, g), None
    new_params, _ = jax.lax.scan(body, params, None, length=PPO_EPOCHS)
    return new_params


def _fisher_inv_g(params, g):
    """Return F^{-1} g using full softmax Fisher on a single state."""
    g_flat = flatten(g)
    n = g_flat.size

    def logp_vec(theta_flat, a):
        # rebuild params from flat to compute logits.
        unflat = unflatten(params, theta_flat)
        return distrax.Categorical(logits=logits_fn(unflat)).log_prob(a)

    theta_flat = flatten(params)
    # Fisher = E_a[grad logp grad logp^T]
    pi = jax.nn.softmax(logits_fn(params))

    def per_a(a):
        ga = jax.grad(logp_vec, argnums=0)(theta_flat, a)
        return pi[a] * jnp.outer(ga, ga)
    F = jnp.sum(jnp.stack([per_a(0), per_a(1)]), axis=0)
    F = F + DAMPING * jnp.eye(n)
    return jnp.linalg.solve(F, g_flat)


def unflatten(template, flat):
    leaves, treedef = jax.tree_util.tree_flatten(template)
    out, idx = [], 0
    for l in leaves:
        sz = l.size
        out.append(flat[idx:idx + sz].reshape(l.shape))
        idx += sz
    return jax.tree_util.tree_unflatten(treedef, out)


def npg_step(params, key):
    """Closed-form NPG reference: F^{-1} grad J, full-batch expectation."""
    g = jax.grad(lambda p: -J(p))(params)  # ascent direction = -grad of -J
    nat = _fisher_inv_g(params, g)
    return unflatten(params, flatten(params) - LR * nat)


def trpo_step(params, key):
    """TRPO-style: NPG direction + backtracking line search on KL."""
    actions, rewards = sample_batch(params, key)
    g = reinforce_grad(params, actions, rewards)
    nat_flat = _fisher_inv_g(params, g)
    # step size from trust-region: sqrt(2 delta / (g^T F^-1 g))
    gTnat = jnp.dot(flatten(g), nat_flat)
    step_size = jnp.sqrt(2 * TRPO_DELTA / (jnp.abs(gTnat) + 1e-12))
    p_old = jax.nn.softmax(logits_fn(params))
    theta0 = flatten(params)

    def body(carry, _):
        accepted, theta, frac = carry
        cand = theta0 - frac * step_size * nat_flat
        cand_params = unflatten(params, cand)
        p_new = jax.nn.softmax(logits_fn(cand_params))
        ok = (kl(p_old, p_new) <= TRPO_DELTA) & (J(cand_params) >= J(params))
        take = (~accepted) & ok
        new_theta = jnp.where(take, cand, theta)
        new_accepted = accepted | ok
        return (new_accepted, new_theta, frac * TRPO_DECAY), None

    init = (jnp.array(False), theta0, jnp.array(1.0))
    (accepted, theta_out, _), _ = jax.lax.scan(
        body, init, None, length=TRPO_BACKTRACK)
    # if nothing accepted, stay put.
    theta_final = jnp.where(accepted, theta_out, theta0)
    return unflatten(params, theta_final)


# PCPG: delegates to src/pc_algorithms/pcpg.py (JPC backend).
PCPG_CFG = PCPGConfig(
    input_dim=1, hidden=HIDDEN, output_dim=2,
    inference_steps=PC_T, inference_lr=PC_GAMMA,
    output_eta=PC_ETA, param_lr=LR)
EQX_TEMPLATE = make_eqx_template(PCPG_CFG)
parity_check(logits_fn, init_params(jax.random.PRNGKey(42)), EQX_TEMPLATE,
             OBS[None])


def pcpg_step(params, key):
    actions, rewards = sample_batch(params, key)
    return _pcpg_step(params, OBS[None], actions, rewards,
                      template=EQX_TEMPLATE, cfg=PCPG_CFG)


ALGOS = {
    "REINFORCE": reinforce_step,
    "PPO":       ppo_step,
    "TRPO":      trpo_step,
    "PCPG":      pcpg_step,
    "NPG":       npg_step,
}


# --- per-seed rollout -----------------------------------------------------
class Log(NamedTuple):
    theta:   jnp.ndarray   # (T, D)
    pa1:     jnp.ndarray   # (T,)
    dtheta:  jnp.ndarray   # (T, D) unit update direction
    kl:      jnp.ndarray   # (T,)
    dJ:      jnp.ndarray   # (T,)


def run_seed(step_fn, seed):
    key = jax.random.PRNGKey(seed)
    init_key, *step_keys = jax.random.split(key, NUM_STEPS + 1)
    params0 = init_params(init_key)

    def body(carry, k):
        params, J_prev, theta_prev = carry
        p_old = jax.nn.softmax(logits_fn(params))
        new_params = step_fn(params, k)
        p_new = jax.nn.softmax(logits_fn(new_params))
        theta_new = flatten(new_params)
        d = theta_new - theta_prev
        d_unit = d / (jnp.linalg.norm(d) + 1e-12)
        J_new = J(new_params)
        log = (theta_new, p_new[1], d_unit, kl(p_old, p_new), J_new - J_prev)
        return (new_params, J_new, theta_new), log

    init_carry = (params0, J(params0), flatten(params0))
    _, logs = jax.lax.scan(body, init_carry, jnp.stack(step_keys))
    return Log(*logs)


run_all_seeds = jax.jit(jax.vmap(
    lambda step_fn, seeds: jax.vmap(lambda s: run_seed(step_fn, s))(seeds),
    in_axes=(None, 0))) if False else None  # not used; see below.


def run_algo(step_fn, seeds):
    return jax.jit(jax.vmap(lambda s: run_seed(step_fn, s)))(seeds)


# --- main -----------------------------------------------------------------
def main():
    seeds = jnp.arange(NUM_SEEDS)
    results = {}
    for name, fn in ALGOS.items():
        print(f"running {name}...")
        results[name] = run_algo(fn, seeds)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {"REINFORCE": "C0", "PPO": "C1", "TRPO": "C2",
              "PCPG": "C3", "NPG": "k"}

    # (a) learning curves: mean pi(a=1) over seeds
    ax = axes[0, 0]
    for name, log in results.items():
        m = np.asarray(log.pa1).mean(0)
        s = np.asarray(log.pa1).std(0)
        ax.plot(m, label=name, color=colors[name])
        ax.fill_between(np.arange(NUM_STEPS), m - s, m + s,
                        color=colors[name], alpha=0.15)
    ax.set_title("(a) learning curve  pi(a=1)")
    ax.set_xlabel("step"); ax.set_ylabel("pi(a=1)"); ax.legend()

    # (b) realized KL per step
    ax = axes[0, 1]
    for name, log in results.items():
        ax.plot(np.asarray(log.kl).mean(0), label=name, color=colors[name])
    ax.set_yscale("log")
    ax.set_title("(b) realized KL per step"); ax.set_xlabel("step")
    ax.set_ylabel("KL(pi_t || pi_{t-1})"); ax.legend()

    # (c) PCA-projected parameter trajectory (seed 0)
    ax = axes[1, 0]
    stacked = np.concatenate(
        [np.asarray(results[name].theta[0]) for name in ALGOS], axis=0)
    mean = stacked.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(stacked - mean, full_matrices=False)
    basis = Vt[:2]                                      # (2, D)
    for name, log in results.items():
        proj = (np.asarray(log.theta[0]) - mean) @ basis.T
        ax.plot(proj[:, 0], proj[:, 1], label=name, color=colors[name])
        ax.scatter(proj[0, 0], proj[0, 1], color=colors[name], marker="o")
        ax.scatter(proj[-1, 0], proj[-1, 1], color=colors[name], marker="x")
    ax.set_title("(c) theta trajectory (PCA, seed 0)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()

    # (d) cosine similarity to NPG update direction
    ax = axes[1, 1]
    npg_d = np.asarray(results["NPG"].dtheta)           # (S, T, D)
    for name, log in results.items():
        if name == "NPG":
            continue
        d = np.asarray(log.dtheta)
        cos = (d * npg_d).sum(-1)                       # (S, T)
        m = cos.mean(0); s = cos.std(0)
        ax.plot(m, label=name, color=colors[name])
        ax.fill_between(np.arange(NUM_STEPS), m - s, m + s,
                        color=colors[name], alpha=0.15)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("(d) cosine sim of update dir vs NPG")
    ax.set_xlabel("step"); ax.set_ylabel("cos(.,NPG)"); ax.legend()

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "mdp_v1_tier1.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # Quick numeric summary for the success criterion.
    late = slice(NUM_STEPS // 2, NUM_STEPS)
    for name, log in results.items():
        if name == "NPG":
            continue
        cos = (np.asarray(log.dtheta) * npg_d).sum(-1)
        print(f"  late-mean cos({name}, NPG) = {cos[:, late].mean():+.3f}")


if __name__ == "__main__":
    main()
