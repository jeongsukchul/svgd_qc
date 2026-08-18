"""Geometry-aware, value-free ANQ with a stochastic drift policy.

This agent keeps ANQSTDFP's refinement actor in action space, but replaces its
Euclidean delta penalty with the local metric induced by the drift generator.
For ``a = G_s(z)`` and ``J = dG_s(z) / dz``, the refinement penalty is

    delta.T @ (J @ J.T + manifold_ridge * I)^-1 @ delta.

The ridge makes the metric positive definite and softly permits movement in
directions to which the generator is locally insensitive.
"""

import jax
import jax.numpy as jnp

from agents.anq_stdfp import (
    ANQSTDFPAgent,
    get_config as get_anq_stdfp_config,
    improvement_penalty_weight,
)


def generator_covariance(jacobians):
    """Return ``J J^T`` for a (possibly batched) generator Jacobian."""
    jacobians = jnp.nan_to_num(
        jacobians, nan=0.0, posinf=0.0, neginf=0.0
    )
    return jnp.einsum("...ik,...jk->...ij", jacobians, jacobians)


def manifold_quadratic(delta, jacobians, ridge):
    """Compute ``delta^T (J J^T + ridge I)^-1 delta``.

    Args:
        delta: Action-space displacement with shape ``(..., action_dim)``.
        jacobians: Generator Jacobian with shape
            ``(..., action_dim, latent_dim)``.
        ridge: Positive scalar metric regularizer.
    """
    covariance = generator_covariance(jacobians)
    identity = jnp.eye(covariance.shape[-1], dtype=covariance.dtype)
    metric_inverse_rhs = jnp.linalg.solve(
        covariance + ridge * identity, delta[..., None]
    )[..., 0]
    quadratic = jnp.sum(delta * metric_inverse_rhs, axis=-1)
    return jnp.maximum(quadratic, 0.0)


class ManiSTDFPAgent(ANQSTDFPAgent):
    """ANQ-STDFP whose action refiner follows drift-generator geometry."""

    def generator_jacobians(self, observations, noises):
        """Evaluate ``d G_s(z) / dz`` independently for each batch item."""

        def generate(observation, noise):
            # Differentiate the smooth learned generator itself. Action clipping
            # is an environment-safety operation and would create artificial
            # zero Jacobians at the box boundary.
            return self.network.select("actor_drift")(observation, noise)

        return jax.vmap(jax.jacrev(generate, argnums=1))(
            observations, noises
        )

    def refine_actor_loss(self, batch, grad_params, rng):
        observations = batch["observations"]
        base, noises = self._refine_base_actions(observations, rng)
        refined, delta = self._refine(
            observations, base, params=grad_params
        )
        qs = self.network.select("critic")(observations, actions=refined)
        q = self._aggregate_q(qs, mode=self.config["refine_q_agg"])

        # Geometry is evaluated at the behavior action's own latent coordinate.
        # The drift generator is deliberately frozen for this loss: it describes
        # the neighborhood, while the refinement actor learns within it.
        jacobians = jax.lax.stop_gradient(
            self.generator_jacobians(observations, noises)
        )
        jacobians = jnp.nan_to_num(
            jacobians, nan=0.0, posinf=0.0, neginf=0.0
        )
        metric_delta_sq = manifold_quadratic(
            delta, jacobians, self.config["manifold_ridge"]
        )

        valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]
        q_objective = (q * valid).sum() 
        penalty = (
            metric_delta_sq * valid
        ).sum()
        euclidean_delta_sq = jnp.sum(jnp.square(delta), axis=-1)
        euclidean_penalty = (euclidean_delta_sq * valid).sum() 
        generator_variance = (
            jnp.sum(jnp.square(jacobians), axis=(-2, -1)) * valid
        ).sum() / (denom * jacobians.shape[-2])

        norm_q = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        loss = - norm_q * q_objective + self.config["lam"] * penalty
        return loss, {
            "refine_actor_loss": loss,
            "refine_q": q_objective,
            "manifold_penalty": penalty,
            "euclidean_penalty": euclidean_penalty,
            "delta_rms": jnp.sqrt(jnp.mean(jnp.square(delta))),
            "generator_variance": generator_variance,
        }

    @staticmethod
    def _validate_config(config):
        ANQSTDFPAgent._validate_config(config)
        if config["manifold_ridge"] <= 0.0:
            raise ValueError("manifold_ridge must be positive")
        if config["lam"] < 0.0:
            raise ValueError("lam must be non-negative")
        if config["refine_action_scale"] < 0.0:
            raise ValueError("refine_action_scale must be non-negative")
        if config["refine_weight_min"] <= 0.0:
            raise ValueError("refine_weight_min must be positive")
        if config["refine_weight_min"] > config["refine_weight_max"]:
            raise ValueError(
                "refine_weight_min cannot exceed refine_weight_max"
            )


def get_config():
    config = get_anq_stdfp_config()
    config.agent_name = "mani_stdfp"
    config.manifold_ridge = 1e-2
    return config
