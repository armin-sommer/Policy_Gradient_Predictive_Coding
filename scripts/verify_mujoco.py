"""Verify MuJoCo backprop support (continuous control).

  dist (pure JAX)  : NormalTanhDistribution log_prob/entropy/kl/sample shapes
  net  (pure JAX)  : continuous MLP policy/value forward + sampled action range
  env  (needs brax): reset/step shapes, continuous actions, done, eval episodes

dist + net run anywhere; env needs brax + a JAX backend (the A100 box) and
auto-skips otherwise. Usage:
    python scripts/verify_mujoco.py
    python scripts/verify_mujoco.py --check dist net
    python scripts/verify_mujoco.py --check env --env hopper
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


def check_dist() -> bool:
    """NormalTanhDistribution must give correct shapes for continuous actions."""
    from networks.distributions import NormalTanhDistribution

    act_dim, n = 6, 8
    dist = NormalTanhDistribution(event_size=act_dim)
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, (n, dist.param_size))  # 2*act_dim
    actions = dist.sample(params, key)
    raw = dist.sample_no_postprocessing(params, key)
    logp = dist.log_prob(params, raw)
    ent = dist.entropy(params, key)
    kl = dist.kl_divergence(params, params)

    param_ok = dist.param_size == 2 * act_dim
    act_ok = actions.shape == (n, act_dim) and bool((np.abs(actions) <= 1.0).all())
    logp_ok = logp.shape == (n,)
    kl_ok = bool(jnp.allclose(kl, 0.0, atol=1e-5))  # KL(p||p) == 0
    print(f"  [dist] param_size  : {'ok' if param_ok else 'BAD'}  {dist.param_size} (want {2*act_dim})")
    print(f"  [dist] sampled acts: {'ok' if act_ok else 'BAD'}  {actions.shape} in [-1,1]")
    print(f"  [dist] log_prob    : {'ok' if logp_ok else 'BAD'}  {logp.shape} (want {(n,)})")
    print(f"  [dist] KL(p||p)=0  : {'ok' if kl_ok else 'BAD'}")
    return param_ok and act_ok and logp_ok and kl_ok


def check_net() -> bool:
    """Continuous MLP policy/value (discrete_policy=False) shapes + action range."""
    from backprop_algorithms.common import make_networks, make_inference_fn

    obs_dim, act_dim, n = 17, 6, 8
    net = make_networks(observation_size=obs_dim, action_size=act_dim,
                        policy_hidden_layer_sizes=(128, 128, 128, 128),
                        value_hidden_layer_sizes=(128, 128, 128, 128),
                        discrete_policy=False, use_cnn=False)
    key = jax.random.PRNGKey(0)
    kp, kv, ks = jax.random.split(key, 3)
    pp = net.policy_network.init(kp)
    vp = net.value_network.init(kv)
    obs = jax.random.normal(ks, (n, obs_dim))

    params = net.policy_network.apply(pp, obs)
    values = net.value_network.apply(vp, obs)
    params_ok = params.shape == (n, 2 * act_dim)  # mean + std
    values_ok = values.shape == (n,)
    print(f"  [net] policy params: {'ok' if params_ok else 'BAD'}  {params.shape} (want {(n, 2*act_dim)})")
    print(f"  [net] value shape  : {'ok' if values_ok else 'BAD'}  {values.shape} (want {(n,)})")

    policy = make_inference_fn(net)(pp)
    actions, _ = policy(obs, ks)
    actions = np.asarray(actions)
    range_ok = bool((np.abs(actions) <= 1.0).all()) and actions.shape == (n, act_dim)
    print(f"  [net] sampled acts : {'ok' if range_ok else 'BAD'}  {actions.shape} in [-1,1]")
    return params_ok and values_ok and range_ok


def check_env(env_name: str, num_envs: int = 8, episode_length: int = 50) -> bool:
    """MujocoVecEnv: obs/action/reward/done sanity + eval episode accounting."""
    try:
        import brax  # noqa: F401
    except ImportError:
        print("  [env] SKIP - brax not installed (run this on the GPU box)")
        return True

    from env import make_vec_env
    from utils.utils import EnvConfig

    cfg = EnvConfig(env_name=env_name, num_envs=num_envs, episode_length=episode_length)
    env = make_vec_env(cfg)
    env.seed(0)
    state = env.reset()

    obs_dim = env.observation_size
    act_dim = env.action_space.n
    obs_ok = state.obs.shape == (num_envs, obs_dim) and state.obs.dtype == np.float32
    cont_ok = getattr(env.action_space, "continuous", False)
    print(f"  [env] reset obs    : {'ok' if obs_ok else 'BAD'}  {state.obs.shape} {state.obs.dtype} (obs_dim={obs_dim})")
    print(f"  [env] action space : {'ok' if cont_ok else 'BAD'}  continuous, act_dim={act_dim}")

    rng = np.random.default_rng(0)
    done_seen, reward_finite, done_binary = False, True, True
    # episode_length is short, so truncation-driven done should appear quickly.
    for _ in range(episode_length + 20):
        actions = rng.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32)
        nstate = env.step(actions)
        if not np.isfinite(nstate.reward).all():
            reward_finite = False
        if ((nstate.done != 0) & (nstate.done != 1)).any():
            done_binary = False
        if nstate.done.any():
            done_seen = True
        state = nstate
        if done_seen:
            break
    print(f"  [env] done flips   : {'ok' if done_seen else f'NO EPISODE ENDED in {episode_length + 20} steps'}")
    print(f"  [env] rewards finite: {'ok' if reward_finite else 'NON-FINITE'}")
    print(f"  [env] done in {{0,1}}: {'ok' if done_binary else 'BAD'}")

    eval_env = make_vec_env(cfg, evaluate=True)
    eval_env.seed(1)
    eval_env.reset()
    for _ in range(2 * episode_length + 20):
        eval_env.step(rng.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32))
    returns, lengths = eval_env.evaluate()
    eval_ok = len(returns) > 0 and np.isfinite(returns).all() and all(l > 0 for l in lengths)
    print(f"  [env] eval episodes: {'ok' if eval_ok else 'BAD'}  "
          f"n={len(returns)} mean_return={np.mean(returns) if returns else float('nan'):.3f}")

    return obs_ok and cont_ok and done_seen and reward_finite and done_binary and eval_ok


CHECKS = {"dist": check_dist, "net": check_net, "env": None}  # env handled below


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", nargs="*", default=list(CHECKS), choices=list(CHECKS))
    parser.add_argument("--env", type=str, default="halfcheetah")
    args = parser.parse_args()

    results = {}
    for name in args.check:
        print(f"\n=== {name} ===")
        try:
            results[name] = check_env(args.env) if name == "env" else CHECKS[name]()
        except Exception as e:  # surface the failure, keep going
            print(f"  [{name}] ERROR: {type(e).__name__}: {e}")
            results[name] = False

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name:4} : {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
