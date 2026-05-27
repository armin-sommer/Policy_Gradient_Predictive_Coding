"""NatureCNN actor-critic policy for Procgen-style (H, W, C) uint8 inputs.

Ported from src/networks/networks.py (the original "impala_cnn" config key
actually corresponds to NatureCNN here; we keep the implementation name
honest). Two separate trunks: one outputs n_actions logits, one outputs a
scalar value. params is {"actor": ..., "critic": ...}.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn

from policies._protocol import Policy


class _NatureCNN(nn.Module):
    """NatureCNN trunk + head. head_std is the orthogonal init scale on the
    *final* Dense (the head). Per CleanRL JAX convention:
        actor head  -> head_std = 0.01
        critic head -> head_std = 1.0
    Conv layers and the 512-trunk Dense always use sqrt(2).
    """
    output_size: int
    head_std: float
    squeeze_output: bool = False

    @nn.compact
    def __call__(self, x):
        kinit = jax.nn.initializers.orthogonal(jnp.sqrt(2))
        binit = jax.nn.initializers.constant(0.0)
        h = x.astype(jnp.float32) / 255.0
        h = nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID",
                    kernel_init=kinit, bias_init=binit)(h)
        h = nn.relu(h)
        h = nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID",
                    kernel_init=kinit, bias_init=binit)(h)
        h = nn.relu(h)
        h = nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID",
                    kernel_init=kinit, bias_init=binit)(h)
        h = nn.relu(h)
        h = h.reshape(h.shape[:-3] + (-1,))
        h = nn.Dense(512, kernel_init=kinit, bias_init=binit)(h)
        h = nn.relu(h)
        head_init = jax.nn.initializers.orthogonal(self.head_std)
        h = nn.Dense(self.output_size, kernel_init=head_init,
                     bias_init=binit)(h)
        if self.squeeze_output:
            h = jnp.squeeze(h, axis=-1)
        return h


def make_nature_cnn_policy(*, obs_shape, n_actions: int,
                           fixed_obs=None, hidden: int = 512) -> Policy:
    """obs_shape = (H, W, C). hidden is unused (NatureCNN has its own
    head size of 512); kept for signature parity with tiny_mlp."""
    del hidden  # NatureCNN's hidden head is fixed at 512

    actor_module = _NatureCNN(output_size=n_actions, head_std=0.01,
                              squeeze_output=False)
    critic_module = _NatureCNN(output_size=1, head_std=1.0,
                               squeeze_output=True)
    dummy = jnp.zeros((1,) + tuple(obs_shape), dtype=jnp.uint8)

    if fixed_obs is None:
        fixed_obs = jnp.zeros(tuple(obs_shape), dtype=jnp.uint8)

    def init(key):
        k_actor, k_critic = jax.random.split(key)
        return {
            "actor":  actor_module.init(k_actor, dummy),
            "critic": critic_module.init(k_critic, dummy),
        }

    def apply(params, x):
        return actor_module.apply(params["actor"], x)

    def value_apply(params, x):
        return critic_module.apply(params["critic"], x)

    def logits_fn(params):
        return actor_module.apply(params["actor"], fixed_obs[None])[0]

    return Policy(init=init, logits_fn=logits_fn, apply=apply,
                  fixed_obs=fixed_obs,
                  obs_dim=int(jnp.prod(jnp.array(obs_shape))),
                  hidden=512, n_actions=n_actions,
                  value_apply=value_apply)
