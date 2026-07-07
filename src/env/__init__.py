from env.env import (
    State,
    ProcgenVecEnv,
    ProcgenEvalEnv,
    make_envs,
    observe,
    step,
)
from env.bandit import BanditVecEnv, BanditEvalEnv
from env.mujoco import MujocoVecEnv, MujocoEvalEnv

Transition = None  # imported from algorithm modules, not env

BANDIT_ENV_NAMES = ("bandit",)
MUJOCO_ENV_NAMES = (
    "ant", "halfcheetah", "hopper", "humanoid", "humanoidstandup",
    "inverted_pendulum", "inverted_double_pendulum", "pusher", "reacher",
    "swimmer", "walker2d",
)


def make_vec_env(cfg, evaluate=False):
    if cfg.env_name in BANDIT_ENV_NAMES:
        return BanditEvalEnv(cfg) if evaluate else BanditVecEnv(cfg)
    if cfg.env_name in MUJOCO_ENV_NAMES:
        return MujocoEvalEnv(cfg) if evaluate else MujocoVecEnv(cfg)
    return ProcgenEvalEnv(cfg) if evaluate else ProcgenVecEnv(cfg)


def make_env(cfg, evaluate=False, **kwargs):
    """Factory matching the PolicyGradientsJax make_env signature."""
    return make_vec_env(cfg, evaluate=evaluate)


def has_discrete_action_space(env):
    return True  # Procgen and bandit are both discrete
