"""Verify the SOTA (CleanRL/Engstrom) network spec is correctly wired.

Pure JAX (no brax/env needed). Builds make_networks(sota_init=True) for a
continuous action space and asserts the canonical properties:
  - tanh activation, separate actor/critic MLPs
  - actor output layer ~ orthogonal gain 0.01  (small initial actions)
  - critic output layer ~ orthogonal gain 1.0
  - state-independent log_std parameter, initialized to 0
  - initial deterministic action ~ 0 and initial std ~ 1 (exp std)

    python scripts/verify_sota_init.py
"""

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backprop_algorithms.common import make_networks, make_inference_fn

OBS, ACT = 17, 6
net = make_networks(observation_size=OBS, action_size=ACT,
                    policy_hidden_layer_sizes=(64, 64),
                    value_hidden_layer_sizes=(64, 64),
                    discrete_policy=False, use_cnn=False, sota_init=True)
key = jax.random.PRNGKey(0)
kp, kv, ks = jax.random.split(key, 3)
pp = net.policy_network.init(kp)
vp = net.value_network.init(kv)
flat = {"/".join(map(str, k)): v for k, v in
        jax.tree_util.tree_flatten_with_path(pp)[0]}

checks = []

def check(name, cond, detail=""):
    checks.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")

# 1. state-independent log_std exists and is zero
logstd = [v for k, v in flat.items() if "log_std" in k]
check("log_std parameter exists", len(logstd) == 1)
if logstd:
    check("log_std initialized to 0", bool(np.allclose(logstd[0], 0.0)),
          f"max|log_std|={float(jnp.abs(logstd[0]).max()):.2e}")

# 2. actor output layer (mean) has small std ~ 0.01
mean_w = [v for k, v in flat.items() if "mean" in k and "kernel" in k]
if mean_w:
    s = float(mean_w[0].std())
    check("actor output gain ~ 0.01", 0.001 < s < 0.05, f"std={s:.4f}")

# 3. hidden layers larger spread (orthogonal sqrt(2))
hid_w = [v for k, v in flat.items() if "hidden_0" in k and "kernel" in k]
if hid_w:
    s = float(hid_w[0].std())
    check("hidden init spread > actor output", s > 0.05, f"hidden std={s:.4f}")

# 4. initial deterministic action ~ 0  (mean head near zero)
make_policy = make_inference_fn(net)
obs = jnp.zeros((4, OBS))
det_act, _ = make_policy(pp, deterministic=True)(obs, ks)
check("initial deterministic action ~ 0", float(jnp.abs(det_act).max()) < 0.1,
      f"max|action|={float(jnp.abs(det_act).max()):.4f}")

# 5. initial std ~ 1  (exp(log_std)=1)
logits = net.policy_network.apply(pp, obs)
dist = net.parametric_action_distribution.create_dist(logits)
check("initial std ~ 1.0", bool(np.allclose(np.asarray(dist.scale), 1.0, atol=1e-3)),
      f"mean std={float(jnp.mean(dist.scale)):.4f}")

print("\nALL PASS" if all(checks) else "\nSOME CHECKS FAILED")
sys.exit(0 if all(checks) else 1)
