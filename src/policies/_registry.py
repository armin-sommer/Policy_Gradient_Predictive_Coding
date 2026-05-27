"""Policy registry: string name -> factory.

Each factory takes (env, **kwargs) and decides which env attributes it needs:
    tiny_mlp     -> env.obs_dim (scalar), env.n_actions
    nature_cnn   -> env.obs_shape (H, W, C), env.n_actions
"""
from policies.tiny_mlp import make_tiny_policy
from policies.nature_cnn import make_nature_cnn_policy


def _tiny_mlp_factory(env, *, hidden, fixed_obs=None):
    return make_tiny_policy(obs_dim=env.obs_dim, hidden=hidden,
                            n_actions=env.n_actions, fixed_obs=fixed_obs)


def _nature_cnn_factory(env, *, hidden=512, fixed_obs=None):
    return make_nature_cnn_policy(obs_shape=env.obs_shape,
                                  n_actions=env.n_actions,
                                  fixed_obs=fixed_obs, hidden=hidden)


_REGISTRY = {
    "tiny_mlp":   _tiny_mlp_factory,
    "nature_cnn": _nature_cnn_factory,
}


def make_policy(name: str, env, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown policy '{name}'. registered: {list(_REGISTRY)}")
    return _REGISTRY[name](env, **kwargs)
