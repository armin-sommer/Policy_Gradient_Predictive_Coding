from env.env import (
    State,
    ProcgenVecEnv,
    ProcgenEvalEnv,
    make_envs,
    observe,
    step,
)
from env.bandit import BanditVecEnv, BanditEvalEnv

Transition = None  # imported from algorithm modules, not env

BANDIT_ENV_NAMES = ("bandit",)


def make_vec_env(cfg, evaluate=False):
    """Factory dispatching on env_name: 'bandit' vs Procgen."""
    if cfg.env_name in BANDIT_ENV_NAMES:
        return BanditEvalEnv(cfg) if evaluate else BanditVecEnv(cfg)
    return ProcgenEvalEnv(cfg) if evaluate else ProcgenVecEnv(cfg)


def make_env(cfg, evaluate=False, **kwargs):
    """Factory matching the PolicyGradientsJax make_env signature."""
    return make_vec_env(cfg, evaluate=evaluate)


def has_discrete_action_space(env):
    return True  # Procgen and bandit are both discrete
