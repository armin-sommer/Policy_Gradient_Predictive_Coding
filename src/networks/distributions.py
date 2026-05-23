"""Probability distributions using distrax."""

import abc
import distrax
import jax
import jax.numpy as jnp


class ParametricDistribution(abc.ABC):
    """Abstract class for parametric (action) distribution."""

    def __init__(self, param_size, postprocessor, event_ndims, reparametrizable):
        self._param_size = param_size
        self._postprocessor = postprocessor
        self._event_ndims = event_ndims
        self._reparametrizable = reparametrizable
        assert event_ndims in [0, 1]

    @abc.abstractmethod
    def create_dist(self, parameters) -> distrax.Normal:
        pass

    @property
    def param_size(self):
        return self._param_size

    @property
    def reparametrizable(self):
        return self._reparametrizable

    def postprocess(self, event):
        if self._postprocessor is not None:
            return self._postprocessor.forward(event)
        return event

    def inverse_postprocess(self, event):
        if self._postprocessor is not None:
            return self._postprocessor.inverse(event)
        return event

    def sample_no_postprocessing(self, parameters, seed):
        return self.create_dist(parameters).sample(seed=seed)

    def sample(self, parameters, seed):
        """Returns a sample from the postprocessed distribution."""
        return self.postprocess(self.sample_no_postprocessing(parameters, seed))

    def mode(self, parameters):
        """Returns the mode of the postprocessed distribution."""
        return self.postprocess(self.create_dist(parameters).mode())

    def log_prob(self, parameters, actions):
        """Compute the log probability of actions."""
        dist = self.create_dist(parameters)
        log_probs = dist.log_prob(actions)
        if self._postprocessor is not None:
            log_probs -= self._postprocessor.forward_log_det_jacobian(actions)
        if self._event_ndims == 1:
            log_probs = jnp.sum(log_probs, axis=-1)
        return log_probs

    def entropy(self, parameters, seed):
        """Return the entropy of the given distribution."""
        dist = self.create_dist(parameters)
        entropy = dist.entropy()
        if self._postprocessor is not None:
            entropy += self._postprocessor.forward_log_det_jacobian(
                dist.sample(seed=seed))
        if self._event_ndims == 1:
            entropy = jnp.sum(entropy, axis=-1)
        return entropy

    def kl_divergence(self, p_parameters, q_parameters):
        """Return the KL divergence of the given distributions."""
        p_dist = self.create_dist(p_parameters)
        q_dist = self.create_dist(q_parameters)

        diff_log_scale = jnp.log(p_dist.scale) - jnp.log(q_dist.scale)
        return (
            0.5 * jnp.square(p_dist.loc / q_dist.scale - q_dist.loc / q_dist.scale) +
            0.5 * (jnp.exp(2. * diff_log_scale) - 1) -
            diff_log_scale)

    def kl_divergence_mu(self, p_parameters, q_parameters):
        """Return the decoupled KL divergence for the mean of the given distributions."""
        p_dist = self.create_dist(p_parameters)
        q_dist = self.create_dist(q_parameters)

        diff_loc = q_dist.loc - p_dist.loc
        return 0.5 * jnp.sum(diff_loc / p_dist.scale * diff_loc, axis=-1)

    def kl_divergence_sigma(self, p_parameters, q_parameters):
        """Return the decoupled KL divergence for the covariance of the given distributions."""
        p_dist = self.create_dist(p_parameters)
        q_dist = self.create_dist(q_parameters)

        return 0.5 * (jnp.sum(p_dist.scale / q_dist.scale, axis=-1) -
                      q_dist.scale.shape[-1] +
                      jnp.prod(q_dist.scale, axis=-1) / jnp.prod(p_dist.scale, axis=-1))


class NormalTanhDistribution(ParametricDistribution):
    """Normal distribution followed by tanh."""

    def __init__(self, event_size, min_std=0.001):
        super().__init__(
            param_size=2 * event_size,
            postprocessor=distrax.Tanh(),
            event_ndims=1,
            reparametrizable=True)
        self._min_std = min_std

    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        scale = jax.nn.softplus(scale) + self._min_std
        return distrax.Normal(loc=loc, scale=scale)


class PolicyNormalDistribution(ParametricDistribution):
    """Normal distribution without postprocessing."""

    def __init__(self, event_size, min_std=0.001):
        super().__init__(
            param_size=2 * event_size,
            postprocessor=None,
            event_ndims=1,
            reparametrizable=True)
        self._min_std = min_std

    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        scale = jax.nn.softplus(scale) + self._min_std
        return distrax.Normal(loc=loc, scale=scale)


class DiscreteDistribution(abc.ABC):
    """Discrete (action) distribution using distrax.Categorical."""

    def __init__(self, param_size):
        self._param_size = param_size
        self._event_ndims = 1
        self._reparametrizable = False

    @property
    def param_size(self):
        return self._param_size

    @property
    def reparametrizable(self):
        return self._reparametrizable

    def postprocess(self, event):
        return event

    def inverse_postprocess(self, event):
        return event

    def sample_no_postprocessing(self, parameters, seed):
        dist = distrax.Categorical(logits=parameters)
        return dist.sample(seed=seed)

    def sample(self, parameters, seed):
        """Returns a sample from the postprocessed distribution."""
        return self.postprocess(self.sample_no_postprocessing(parameters, seed))

    def mode(self, parameters):
        """Returns the mode of the discrete distribution."""
        return distrax.Categorical(logits=parameters).mode()

    def log_prob(self, parameters, actions):
        """Compute the log probability of actions."""
        dist = distrax.Categorical(logits=parameters)
        return dist.log_prob(actions.astype(jnp.int32))

    def entropy(self, parameters, seed):
        """Return the entropy of the given distribution."""
        dist = distrax.Categorical(logits=parameters)
        return dist.entropy()

    def kl_divergence(self, p_parameters, q_parameters):
        """Return the KL divergence of the given distributions."""
        p_dist = distrax.Categorical(logits=p_parameters)
        q_dist = distrax.Categorical(logits=q_parameters)
        return p_dist.kl_divergence(q_dist)
