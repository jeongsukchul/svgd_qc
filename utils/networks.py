import functools
from typing import Any, Optional, Sequence, Type

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability


tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)



class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    swap : bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = self.activations(nn.LayerNorm()(x)) if self.swap else nn.LayerNorm()(self.activations(x))
                else:
                    x = self.activations(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution, validate_args: bool = False):
        super().__init__(
            distribution=distribution,
            bijector=tfb.Tanh(),
            validate_args=validate_args,
        )

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class Normal(nn.Module):
    """Gaussian policy head built on top of an arbitrary base network."""

    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    learnable_log_std_multiplier: Optional[float] = None
    learnable_log_std_offset: Optional[float] = None
    state_dependent_std: bool = True
    squash_tanh: bool = False
    fixed_log_std: Optional[float] = None
    encoder: nn.Module = None

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        if self.encoder is not None:
            inputs = self.encoder(inputs)
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim,
            kernel_init=default_init(),
            name="OutputDenseMean",
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim,
                kernel_init=default_init(),
                name="OutputDenseLogStd",
            )(x)
        else:
            log_stds = self.param(
                "OutputLogStd",
                nn.initializers.zeros,
                (self.action_dim,),
                jnp.float32,
            )

        if self.learnable_log_std_multiplier is not None:
            log_stds *= self.param(
                "LogStdMul",
                nn.initializers.constant(self.learnable_log_std_multiplier),
                (),
                jnp.float32,
            )
        if self.learnable_log_std_offset is not None:
            log_stds += self.param(
                "LogStdOffset",
                nn.initializers.constant(self.learnable_log_std_offset),
                (),
                jnp.float32,
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        if self.fixed_log_std is not None:
            log_stds = jnp.ones_like(log_stds) * self.fixed_log_std

        distribution = tfd.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds),
        )
        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    swap : bool = False
    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm, swap=self.swap)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class NonNegativeDense(nn.Module):
    """Dense layer with non-negative kernel weights."""

    features: int
    use_bias: bool = False
    kernel_init: Any = default_init()
    bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (x.shape[-1], self.features),
            jnp.float32,
        )
        y = jnp.matmul(x, jnp.square(kernel))
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), jnp.float32)
            y = y + bias
        return y


class ICNN(nn.Module):
    """Conditional input convex neural network scalar potential.

    The output is convex in `actions` for each fixed observation/context. Context
    inputs may enter through unrestricted affine terms; hidden-to-hidden convex
    connections are constrained to be non-negative.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    activations: Any = nn.softplus
    input_quadratic_scale: float = 1.0

    def setup(self) -> None:
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        context = observations
        if times is not None:
            if self.use_fourier_features:
                times = self.ff(times)
            context = jnp.concatenate([context, times], axis=-1)
        if self.layer_norm:
            context = nn.LayerNorm(name="context_layer_norm")(context)

        z = None
        for i, size in enumerate(self.hidden_dims):
            hidden = nn.Dense(size, name=f"input_dense_{i}")(actions)
            hidden += nn.Dense(size, name=f"context_dense_{i}")(context)
            if z is not None:
                hidden += NonNegativeDense(
                    size,
                    use_bias=False,
                    name=f"hidden_dense_{i}",
                )(z)
            z = self.activations(hidden)

        potential = nn.Dense(1, name="input_output_dense")(actions)
        potential += nn.Dense(1, name="context_output_dense")(context)
        if z is not None:
            potential += NonNegativeDense(
                1,
                use_bias=False,
                name="hidden_output_dense",
            )(z)
        potential = potential.squeeze(-1)

        if self.input_quadratic_scale > 0.0:
            potential += 0.5 * self.input_quadratic_scale * jnp.sum(
                jnp.square(actions),
                axis=-1,
            )
        return potential


class ActorICNN(nn.Module):
    """Drift actor given by the input gradient of an ICNN potential."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    input_quadratic_scale: float = 1.0
    output_scale: float = 1.0

    def setup(self) -> None:
        self.potential = ICNN(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            encoder=self.encoder,
            use_fourier_features=self.use_fourier_features,
            fourier_feature_dim=self.fourier_feature_dim,
            input_quadratic_scale=self.input_quadratic_scale,
        )

    def __call__(self, observations, actions, times=None, is_encoded=False):
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, got {actions.shape[-1]}"
            )

        def potential_sum(convex_inputs):
            potential = self.potential(
                observations,
                convex_inputs,
                times=times,
                is_encoded=is_encoded,
            )
            return potential.sum()

        return self.output_scale * jax.grad(potential_sum)(actions)


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    swap : bool =False

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm, swap=self.swap)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v
