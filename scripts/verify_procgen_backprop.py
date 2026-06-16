"""Verify the Procgen-specific backprop code paths the bandit never exercised.

  gae  (pure JAX)  : multi-step GAE accumulation + bootstrap   [runs anywhere]
  cnn  (pure JAX)  : NatureCNN forward on uint8 (N,64,64,3)     [runs anywhere]
  env  (Procgen)   : ProcgenVecEnv done/reward alignment        [needs Procgen]

Usage:
    python scripts/verify_procgen_backprop.py            # all checks (env auto-skips)
    python scripts/verify_procgen_backprop.py --check gae cnn
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import jax
import jax.numpy as jnp


def check_gae() -> bool:
    """compute_gae must match the textbook identity A_t = G_t - V_t at lambda=1."""
    from backprop_algorithms.common import compute_gae

    gamma = 0.99
    rewards = jnp.array([1.0, 0.5, 2.0, -1.0])
    values = jnp.array([0.3, 0.7, 0.1, 0.4])
    bootstrap = jnp.array(0.9)
    T = rewards.shape[0]
    zeros = jnp.zeros(T)

    def mc_return_to_go(term_last, use_bootstrap):
        # G_t = sum_{k>=t} gamma^{k-t} r_k (+ gamma^{T-t} bootstrap if not terminal)
        g = np.zeros(T)
        future = float(bootstrap) if use_bootstrap else 0.0
        for t in range(T - 1, -1, -1):
            future = float(rewards[t]) + gamma * future
            g[t] = future
        return jnp.asarray(g)

    ok = True

    # Case 1: terminating episode (termination=1 on the last step), no bootstrap.
    termination = jnp.array([0.0, 0.0, 0.0, 1.0])
    _, adv = compute_gae(truncation=zeros, termination=termination,
                         rewards=rewards, values=values, bootstrap_value=bootstrap,
                         lambda_=1.0, discount=gamma)
    expected = mc_return_to_go(term_last=True, use_bootstrap=False) - values
    same = bool(jnp.allclose(adv, expected, atol=1e-5))
    print(f"  [gae] terminating  : {'ok' if same else 'MISMATCH'}  "
          f"adv={np.round(np.asarray(adv), 4)} expected={np.round(np.asarray(expected), 4)}")
    ok = ok and same

    # Case 2: continuing rollout (no termination), bootstrap used at the end.
    termination = zeros
    _, adv = compute_gae(truncation=zeros, termination=termination,
                         rewards=rewards, values=values, bootstrap_value=bootstrap,
                         lambda_=1.0, discount=gamma)
    expected = mc_return_to_go(term_last=False, use_bootstrap=True) - values
    same = bool(jnp.allclose(adv, expected, atol=1e-5))
    print(f"  [gae] bootstrapped : {'ok' if same else 'MISMATCH'}  "
          f"adv={np.round(np.asarray(adv), 4)} expected={np.round(np.asarray(expected), 4)}")
    ok = ok and same
    return ok


def check_cnn() -> bool:
    """NatureCNN policy/value must produce correct shapes on uint8 image obs."""
    from backprop_algorithms.common import make_networks, make_inference_fn

    obs_shape = (64, 64, 3)
    action_size = 15
    n = 8
    net = make_networks(observation_size=obs_shape, action_size=action_size,
                        discrete_policy=True, use_cnn=True)
    key = jax.random.PRNGKey(0)
    kp, kv, ks = jax.random.split(key, 3)
    policy_params = net.policy_network.init(kp)
    value_params = net.value_network.init(kv)

    obs = jnp.asarray(np.random.randint(0, 256, size=(n,) + obs_shape, dtype=np.uint8))
    logits = net.policy_network.apply(policy_params, obs)
    values = net.value_network.apply(value_params, obs)

    logits_ok = logits.shape == (n, action_size)
    values_ok = values.shape == (n,)
    print(f"  [cnn] logits shape : {'ok' if logits_ok else 'BAD'}  {logits.shape} (want {(n, action_size)})")
    print(f"  [cnn] value shape  : {'ok' if values_ok else 'BAD'}  {values.shape} (want {(n,)})")

    # The jitted inference kernel must sample valid actions from image obs.
    policy = make_inference_fn(net)(policy_params)
    actions, _ = policy(obs, ks)
    actions = np.asarray(actions)
    range_ok = bool((actions >= 0).all() and (actions < action_size).all())
    print(f"  [cnn] sampled acts : {'ok' if range_ok else 'OUT OF RANGE'}  "
          f"min={actions.min()} max={actions.max()}")
    return logits_ok and values_ok and range_ok


def check_env() -> bool:
    """ProcgenVecEnv: obs/done/reward sanity + eval episode accounting."""
    try:
        import procgen  # noqa: F401
    except ImportError:
        print("  [env] SKIP — Procgen not installed (run this check on the Linux box)")
        return True

    from env import make_vec_env
    from utils.utils import EnvConfig

    cfg = EnvConfig(env_name="coinrun", num_envs=8, num_train_levels=200,
                    distribution_mode="easy")
    env = make_vec_env(cfg)
    env.seed(0)
    state = env.reset()

    obs_ok = state.obs.shape == (8, 64, 64, 3) and state.obs.dtype == np.uint8
    print(f"  [env] reset obs    : {'ok' if obs_ok else 'BAD'}  {state.obs.shape} {state.obs.dtype}")

    rng = np.random.default_rng(0)
    done_seen, reward_finite, terminal_reward_ok = False, True, True
    for _ in range(400):
        actions = rng.integers(0, env.action_space.n, size=8)
        nstate = env.step(actions)
        if not np.isfinite(nstate.reward).all():
            reward_finite = False
        if ((nstate.done != 0) & (nstate.done != 1)).any():
            terminal_reward_ok = False
        if nstate.done.any():
            done_seen = True
        state = nstate
    print(f"  [env] done flips   : {'ok' if done_seen else 'NO EPISODE ENDED in 400 steps'}")
    print(f"  [env] rewards finite: {'ok' if reward_finite else 'NON-FINITE'}")
    print(f"  [env] done in {{0,1}}: {'ok' if terminal_reward_ok else 'BAD'}")

    eval_env = make_vec_env(cfg, evaluate=True)
    eval_env.seed(1)
    eval_env.reset()
    for _ in range(600):
        eval_env.step(rng.integers(0, eval_env.action_space.n, size=8))
    returns, lengths = eval_env.evaluate()
    eval_ok = len(returns) > 0 and np.isfinite(returns).all() and all(l > 0 for l in lengths)
    print(f"  [env] eval episodes: {'ok' if eval_ok else 'BAD'}  "
          f"n={len(returns)} mean_return={np.mean(returns) if returns else float('nan'):.3f}")
    return obs_ok and done_seen and reward_finite and terminal_reward_ok and eval_ok


CHECKS = {"gae": check_gae, "cnn": check_cnn, "env": check_env}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", nargs="*", default=list(CHECKS), choices=list(CHECKS))
    args = parser.parse_args()

    results = {}
    for name in args.check:
        print(f"\n=== {name} ===")
        try:
            results[name] = CHECKS[name]()
        except Exception as e:  # surface the failure, keep going
            print(f"  [{name}] ERROR: {type(e).__name__}: {e}")
            results[name] = False

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name:4} : {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
