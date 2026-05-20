from env.env import (
    State,
    ProcgenVecEnv,
    ProcgenEvalEnv,
    make_envs,
    observe,
    step,
)

Transition = None  # imported from algorithm modules, not env

def make_env(cfg, evaluate=False, **kwargs):
    """Factory matching the PolicyGradientsJax make_env signature."""
    if evaluate:
        return ProcgenEvalEnv(cfg)
    return ProcgenVecEnv(cfg)

def has_discrete_action_space(env):
    return True  # Procgen is always discrete
