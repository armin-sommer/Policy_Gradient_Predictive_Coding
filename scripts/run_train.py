"""Train entry point: load YAML config, dispatch to the chosen algorithm.

Usage:
    python scripts/run_train.py --config configs/default.yaml
    python scripts/run_train.py --config configs/default.yaml --overrides agent.algorithm=trpo seed=7
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# YAML key -> algorithm Config attribute. Keys missing from the algo's Config
# are silently skipped (e.g. clip_coef is PPO-only).
KEY_MAP = {
    ("env", "env_name"): "env_name",
    ("env", "num_envs"): "num_envs",
    ("env", "num_train_levels"): "num_train_levels",
    ("env", "distribution_mode"): "distribution_mode",
    ("env", "arm_means"): "arm_means",
    ("env", "deterministic_rewards"): "deterministic_rewards",
    ("train", "total_steps"): "total_timesteps",
    ("train", "rollout_length"): "unroll_length",
    ("train", "batch_size"): "batch_size",
    ("train", "num_minibatches"): "num_minibatches",
    ("train", "update_epochs"): "update_epochs",
    ("train", "gamma"): "gamma",
    ("train", "gae_lambda"): "gae_lambda",
    ("train", "clip_coef"): "clip_eps",
    ("train", "ent_coef"): "entropy_cost",
    ("train", "vf_coef"): "vf_cost",
    ("train", "learning_rate"): "learning_rate",
    ("train", "anneal_lr"): "anneal_lr",
    ("train", "max_grad_norm"): "max_grad_norm",
    ("train", "adam_eps"): "adam_eps",
    ("train", "normalize_rewards"): "normalize_rewards",
    ("agent", "sota_init"): "sota_init",
    ("agent", "experiment_name"): "experiment_name",
    ("agent", "policy_init_logit_bias"): "policy_init_logit_bias",
    ("agent", "policy_hidden_layer_sizes"): "policy_hidden_layer_sizes",
    ("agent", "value_hidden_layer_sizes"): "value_hidden_layer_sizes",
    ("train", "eval_every"): "eval_every",
    ("seed",): "seed",
}

ALGO_MODULES = {
    "ppo": "backprop_algorithms.ppo",
    "trpo": "backprop_algorithms.trpo",
    "reinforce": "backprop_algorithms.reinforce",
    "cleanba_ppo": "backprop_algorithms.cleanba_ppo",
    "pc_reinforce": "pc_algorithms.pc_reinforce",
    "pc_actor_critic": "pc_algorithms.pc_actor_critic",
}


def _deep_get(cfg, path):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _apply_overrides(cfg, overrides):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        path, value = item.split("=", 1)
        keys = path.split(".")
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass
        cur = cfg
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
    return cfg


def _override_algo_config(AlgoConfig, cfg):
    """Set attributes on the algorithm's inline Config class from the YAML dict."""
    for yaml_path, attr in KEY_MAP.items():
        value = _deep_get(cfg, yaml_path)
        if value is None:
            continue
        if not hasattr(AlgoConfig, attr):
            continue
        setattr(AlgoConfig, attr, value)

    network = _deep_get(cfg, ("agent", "network"))
    if network is not None and hasattr(AlgoConfig, "use_cnn"):
        AlgoConfig.use_cnn = "cnn" in str(network).lower()


def _setup_wandb(cfg, run_name):
    wandb_cfg = cfg.get("wandb") or {}
    mode = wandb_cfg.get("mode", "disabled")
    if mode == "disabled":
        return None
    import wandb
    wandb.init(
        project=wandb_cfg.get("project", "pcpg"),
        group=wandb_cfg.get("group"),
        name=run_name,
        mode=mode,
        config=cfg,
    )
    # Hook stdlib logging.info(dict) -> wandb.log so we don't have to edit
    # the algorithm files. Each algo logs metric dicts via logging.info(...).
    import logging
    class _WandbHandler(logging.Handler):
        _last_step = 0  # eval dicts have no total_steps; reuse the latest train step
        def emit(self, record):
            msg = record.msg
            if isinstance(msg, dict):
                if "training/total_steps" in msg:
                    _WandbHandler._last_step = int(msg["training/total_steps"])
                wandb.log({k: v for k, v in msg.items() if k != "training/total_steps"},
                          step=_WandbHandler._last_step)
    logging.getLogger().addHandler(_WandbHandler())
    # The algo's logging.basicConfig is a no-op once a handler exists, so without
    # this the run trains but writes nothing to stdout / the per-run log file.
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(stream)
    logging.getLogger().setLevel(logging.INFO)
    return wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Dotted-path overrides, e.g. agent.algorithm=trpo seed=7",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "checkpoints"),
        help="Where to save final model params.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable checkpoint saving (default: enabled).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = _apply_overrides(cfg, args.overrides)

    algo = (cfg.get("agent") or {}).get("algorithm", "ppo").lower()
    if algo not in ALGO_MODULES:
        raise ValueError(f"Unknown algorithm: {algo}. Choices: {list(ALGO_MODULES)}")

    algo_module = importlib.import_module(ALGO_MODULES[algo])
    AlgoConfig = algo_module.Config

    _override_algo_config(AlgoConfig, cfg)
    AlgoConfig.save_model = not args.no_save
    AlgoConfig.checkpoint_dir = args.checkpoint_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    run_name = f"Exp_{AlgoConfig.experiment_name}__{AlgoConfig.env_name}__{AlgoConfig.seed}"
    _setup_wandb(cfg, run_name)

    print(f"[run_train] algorithm={algo} env={AlgoConfig.env_name} "
          f"num_envs={AlgoConfig.num_envs} total_steps={AlgoConfig.total_timesteps} "
          f"seed={AlgoConfig.seed} use_cnn={getattr(AlgoConfig, 'use_cnn', None)}")
    algo_module.main(None)


if __name__ == "__main__":
    main()
