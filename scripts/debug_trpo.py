"""Isolated CPU reproduction of the TRPO non-convergence bug.

Synthetic 1-context continuous task: reward = -mean((action - TARGET)^2), so the
optimal policy drives the (tanh-squashed) action to TARGET. Runs the REAL
trpo_policy_update for N iterations and prints whether the deterministic action
converges to TARGET (TRPO works) or collapses toward 0 (the HalfCheetah symptom).
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backprop_algorithms.common import make_networks, make_inference_fn, NetworkParams, Transition
from backprop_algorithms.trpo import (
    compute_policy_objective, compute_policy_objective_and_kl,
    jacobian_vector_product, hessian_vector_product, trpo_policy_update)

OBS_DIM, ACT_DIM, B, T = 4, 2, 64, 8
TARGET = jnp.array([0.5, -0.5])

net = make_networks(observation_size=OBS_DIM, action_size=ACT_DIM,
                    policy_hidden_layer_sizes=(64, 64), value_hidden_layer_sizes=(64, 64),
                    discrete_policy=False, use_cnn=False)
make_policy = make_inference_fn(net)
key = jax.random.PRNGKey(0)
kp, kv, key = jax.random.split(key, 3)
params = NetworkParams(policy=net.policy_network.init(kp), value=net.value_network.init(kv))

# wiring, copied verbatim from trpo.main()
pog_fn = jax.value_and_grad(compute_policy_objective, has_aux=True)
pokl_fn = partial(compute_policy_objective_and_kl, policy_objective_grad_fn=pog_fn)
pokl_grad_fn = jax.value_and_grad(pokl_fn, has_aux=True)
jvp_fn = partial(jacobian_vector_product, policy_objective_and_kl_grad_fn=pokl_grad_fn)
hvp_fn = partial(hessian_vector_product, hessian_fn=jax.grad(jvp_fn), cg_damping=0.1)
update = partial(trpo_policy_update, network=net,
                 policy_objective_and_kl_grad_fn=pokl_grad_fn, hessian_vector_product=hvp_fn,
                 target_kl=0.01, line_search_max_iter=10, line_search_shrinking_factor=0.8,
                 cg_max_iterations=10, discounting=0.99, reward_scaling=1.0, gae_lambda=0.95,
                 normalize_advantage=True, pmap_axis_name="i")
update_p = jax.pmap(update, axis_name="i")  # pmean needs a named axis -> pmap over 1 CPU device

rep = lambda x: jax.tree_util.tree_map(lambda a: a[None], x)
unrep = lambda x: jax.tree_util.tree_map(lambda a: a[0], x)
OBS_SCALE = float(os.environ.get("OBS_SCALE", "1"))  # 1 = normalized; large = unnormalized
key, kobs = jax.random.split(key)
obs = jax.random.normal(kobs, (B, T, OBS_DIM)) * OBS_SCALE
if os.environ.get("NORMALIZE"):  # the proposed fix
    obs = (obs - obs.mean()) / (obs.std() + 1e-8)
print(f"obs scale = {OBS_SCALE}  normalize={bool(os.environ.get('NORMALIZE'))}  "
      f"(|obs| mean = {float(jnp.abs(obs).mean()):.1f})")

print(f"target action = {np.asarray(TARGET)}")
for it in range(40):
    key, ks = jax.random.split(key)
    acts, extras = make_policy(params.policy)(obs.reshape(-1, OBS_DIM), ks)
    acts = acts.reshape(B, T, ACT_DIM)
    data = Transition(
        observation=obs, action=acts,
        reward=-jnp.mean((acts - TARGET) ** 2, axis=-1),
        discount=jnp.ones((B, T)), next_observation=obs,
        extras={"policy_extras": {"raw_action": extras["raw_action"].reshape(B, T, ACT_DIM),
                                  "log_prob": extras["log_prob"].reshape(B, T)},
                "state_extras": {"truncation": jnp.zeros((B, T))}})
    new_params, metrics = update_p(rep(params), rep(data))
    params = unrep(new_params)

    det_act, _ = make_policy(params.policy, deterministic=True)(obs.reshape(-1, OBS_DIM), ks)
    det_act = np.asarray(det_act.reshape(B, T, ACT_DIM).mean((0, 1)))
    eval_reward = -float(np.mean((det_act - np.asarray(TARGET)) ** 2))
    if it % 2 == 0 or it < 5:
        print(f"it {it:2d}  eval_reward={eval_reward:7.4f}  mean_act={np.round(det_act, 3)}  "
              f"kl={float(metrics['kl_div'][0]):.4f}  ls_ok={float(metrics['line_search_success'][0]):.0f}")

print("\nWORKS if eval_reward -> 0 and mean_act -> target; BUG if mean_act -> ~0 / diverges.")
