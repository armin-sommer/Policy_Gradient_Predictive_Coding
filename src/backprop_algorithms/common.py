"""Shared helpers used by REINFORCE, PPO, and TRPO."""

from typing import Any, Mapping, NamedTuple, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from networks.policy import Policy
from networks.networks import (
    FeedForwardNetwork,
    ActivationFn,
    make_policy_network,
    make_value_network,
    make_cnn_policy_network,
    make_cnn_value_network,
    make_sota_policy_network,
    make_sota_value_network,
)
from networks.distributions import (
    NormalTanhDistribution,
    ParametricDistribution,
    DiscreteDistribution,
)


Metrics = Mapping[str, jnp.ndarray]

PMAP_AXIS_NAME = 'i'


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray
    extras: jnp.ndarray = ()


def unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)
    return jax.tree_util.tree_map(f, tree)


@flax.struct.dataclass
class NetworkParams:
    policy: Any
    value: Any


@flax.struct.dataclass
class Networks:
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: Union[ParametricDistribution, DiscreteDistribution]


@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: NetworkParams
    env_steps: jnp.ndarray


def make_inference_fn(agent_networks: Networks):
    """Creates an inference function for the agent."""

    def make_policy(params: Any, deterministic: bool = False) -> Policy:
        policy_network = agent_networks.policy_network
        parametric_action_distribution = agent_networks.parametric_action_distribution

        @jax.jit
        def policy(observations: jnp.ndarray,
                   key_sample: jnp.ndarray) -> Tuple[jnp.ndarray, Mapping[str, Any]]:
            logits = policy_network.apply(params, observations)
            if deterministic:
                return agent_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample)
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions,
            }

        return policy

    return make_policy


def apply_policy_init_logit_bias(policy_params, logit_bias):
    """Set final MLP policy bias (and zero kernel) for a fixed initial policy."""
    logit_bias = jnp.asarray(logit_bias, dtype=jnp.float32)
    params = flax.core.unfreeze(policy_params)
    layers = params["params"]
    hidden_keys = sorted(
        (k for k in layers if k.startswith("hidden_")),
        key=lambda k: int(k.split("_")[1]),
    )
    if not hidden_keys:
        raise ValueError(
            "policy_init_logit_bias only supports the MLP policy (use_cnn=False)")
    final = layers[hidden_keys[-1]]
    if final["bias"].shape != logit_bias.shape:
        raise ValueError(
            f"logit_bias shape {logit_bias.shape} != action logits shape {final['bias'].shape}")
    final["kernel"] = jnp.zeros_like(final["kernel"])
    final["bias"] = logit_bias
    return flax.core.freeze(params) if isinstance(policy_params, flax.core.FrozenDict) else params


def make_networks(
        observation_size,
        action_size: int,
        policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
        value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
        activation: ActivationFn = nn.swish,
        discrete_policy: bool = True,
        use_cnn: bool = False,
        sota_init: bool = False,
    ) -> Networks:
    """Build the shared policy/value networks and action distribution.

    sota_init=True selects the CleanRL/Engstrom continuous-control stack
    (orthogonal init, tanh, state-independent log_std with exp std). It only
    applies to the MLP continuous path (Gaussian, non-CNN); the discrete and
    CNN paths ignore it.
    """
    sota = sota_init and not discrete_policy and not use_cnn
    if discrete_policy:
        parametric_action_distribution = DiscreteDistribution(param_size=action_size)
    else:
        parametric_action_distribution = NormalTanhDistribution(
            event_size=action_size, exp_std=sota)
    if use_cnn:
        policy_network = make_cnn_policy_network(
            parametric_action_distribution.param_size,
            observation_size,
            hidden_layer_sizes=policy_hidden_layer_sizes,
            activation=activation)
        value_network = make_cnn_value_network(
            observation_size,
            hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation)
    elif sota:
        # tanh is baked in (part of the canonical spec)
        policy_network = make_sota_policy_network(
            action_size, observation_size,
            hidden_layer_sizes=policy_hidden_layer_sizes, activation=nn.tanh)
        value_network = make_sota_value_network(
            observation_size,
            hidden_layer_sizes=value_hidden_layer_sizes, activation=nn.tanh)
    else:
        policy_network = make_policy_network(
            parametric_action_distribution.param_size,
            observation_size,
            hidden_layer_sizes=policy_hidden_layer_sizes,
            activation=activation)
        value_network = make_value_network(
            observation_size,
            hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation)

    return Networks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution)


def compute_gae(truncation: jnp.ndarray,
                termination: jnp.ndarray,
                rewards: jnp.ndarray,
                values: jnp.ndarray,
                bootstrap_value: jnp.ndarray,
                lambda_: float = 1.0,
                discount: float = 0.99):
    """Generalized Advantage Estimation."""
    truncation_mask = 1 - truncation
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs, (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True)
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate(
        [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (rewards + discount *
                  (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)
