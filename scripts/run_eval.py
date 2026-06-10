"""Eval entry point: load a checkpoint, roll out N episodes, report scores.

"""

import argparse
import sys
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _deep_get(cfg, path, default=None):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Override agent.algorithm from the YAML. One of: ppo, trpo, reinforce.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions instead of sampling.",
    )
    parser.add_argument(
        "--test-levels",
        action="store_true",
        help="Evaluate on held-out test levels (start_level=num_train_levels).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    algo = (args.algorithm or _deep_get(cfg, ("agent", "algorithm"), "ppo")).lower()
    network = _deep_get(cfg, ("agent", "network"), "impala_cnn")
    use_cnn = "cnn" in str(network).lower()

    from env import make_vec_env  # noqa: E402
    from utils.utils import EnvConfig  # noqa: E402

    env_cfg = EnvConfig(
        env_name=_deep_get(cfg, ("env", "env_name"), "coinrun"),
        num_envs=1,
        num_train_levels=_deep_get(cfg, ("env", "num_train_levels"), 200),
        num_test_levels=0 if not args.test_levels else 0,
        distribution_mode=_deep_get(cfg, ("env", "distribution_mode"), "easy"),
        arm_means=tuple(_deep_get(cfg, ("env", "arm_means"), (1.0, 0.9))),
        deterministic_rewards=_deep_get(cfg, ("env", "deterministic_rewards"), True),
    )
    eval_env = make_vec_env(env_cfg, evaluate=True)
    eval_env.seed(args.seed)
    state = eval_env.reset()

    action_size = eval_env.action_space.n
    obs_shape = tuple(state.obs.shape[1:]) if use_cnn else int(np.prod(state.obs.shape[1:]))

    if algo in ("ppo", "trpo", "reinforce", "cleanba_ppo"):
        from backprop_algorithms.common import make_networks, make_inference_fn
        networks = make_networks(
            observation_size=obs_shape,
            action_size=action_size,
            discrete_policy=True,
            use_cnn=use_cnn,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Initialize params with the right shape, then load checkpoint into them.
    key = jax.random.PRNGKey(args.seed)
    key_policy, key_value = jax.random.split(key)
    init_policy = networks.policy_network.init(key_policy)
    init_value = networks.value_network.init(key_value)

    with open(args.checkpoint, "rb") as f:
        raw = f.read()
    loaded = flax.serialization.from_bytes(
        {"policy": init_policy, "value": init_value}, raw
    )
    policy_params = loaded["policy"]

    make_policy = make_inference_fn(networks)
    policy = make_policy(policy_params, deterministic=args.deterministic)

    rng = jax.random.PRNGKey(args.seed + 1)
    while len(eval_env.returns) < args.num_episodes:
        rng, sub = jax.random.split(rng)
        obs = (state.obs.astype(np.uint8) if use_cnn
               else eval_env.normalize_obs(state.obs.reshape(state.obs.shape[0], -1).astype(np.float32)))
        actions, _ = policy(obs, sub)
        state = eval_env.step(np.asarray(actions))

    returns, ep_lengths = eval_env.evaluate()
    returns = returns[: args.num_episodes]
    ep_lengths = ep_lengths[: args.num_episodes]
    print(f"[eval] algorithm={algo} env={env_cfg.env_name} "
          f"episodes={len(returns)} mean_return={np.mean(returns):.3f} "
          f"std_return={np.std(returns):.3f} mean_length={np.mean(ep_lengths):.1f}")


if __name__ == "__main__":
    main()
