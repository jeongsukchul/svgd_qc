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


def manifold_quadratic(delta, jacobians, ridge, normalize=False):
    """Compute ``delta^T (J J^T + ridge I)^-1 delta``.

    Args:
        delta: Action-space displacement with shape ``(..., action_dim)``.
        jacobians: Generator Jacobian with shape
            ``(..., action_dim, latent_dim)``.
        ridge: Positive scalar metric regularizer.
        normalize: Rescale the metric to unit mean eigenvalue.  ``J J^T`` grows
            as the decoder sharpens -- measured 33x -> 12.7x over 65k steps --
            so without this the trust region silently loosens during training
            and ``lam`` has no stable meaning.  Normalising leaves the metric
            encoding only *which* directions are cheap, and makes the penalty
            reduce exactly to ``||delta||^2`` for any isotropic metric, so
            ``lam`` carries the same meaning as in anq_stdfp.
    """
    covariance = generator_covariance(jacobians)
    identity = jnp.eye(covariance.shape[-1], dtype=covariance.dtype)
    metric = covariance + ridge * identity
    metric_inverse_rhs = jnp.linalg.solve(metric, delta[..., None])[..., 0]
    quadratic = jnp.sum(delta * metric_inverse_rhs, axis=-1)
    if normalize:
        dim = covariance.shape[-1]
        mean_eig = jnp.trace(metric, axis1=-2, axis2=-1) / dim
        quadratic = quadratic * mean_eig
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
        # Which displacement the metric trust region is measured on.  Anchoring
        # on ``delta`` (the drift sample) makes the penalty pure shrinkage toward
        # the decoder's own output: with M^-1 eigenvalues of 6-100 it dominates
        # the Q gradient ~7x and collapses delta to ~5e-4, i.e. plain BC.
        # Anchoring on the dataset action instead gives ReBRAC's BC penalty under
        # an anisotropic metric -- deviate where the behavior varies, not where
        # it never moves -- and reduces to anq_stdfp exactly when M = I.
        if self.config["refine_anchor"] == "data":
            offset = refined - self._batch_actions(batch)
        else:
            offset = delta
        metric_offset_sq = manifold_quadratic(
            offset, jacobians, self.config["manifold_ridge"],
            normalize=self.config["metric_normalize"],
        )

        valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]
        # Means, not sums, so ``lam`` and the logged penalties are directly
        # comparable with anq_stdfp's.
        q_objective = (q * valid).mean()
        penalty = (metric_offset_sq * valid).mean()
        euclidean_penalty = (
            jnp.sum(jnp.square(offset), axis=-1) * valid
        ).mean()
        generator_variance = (
            jnp.sum(jnp.square(jacobians), axis=(-2, -1)) * valid
        ).mean()

        norm_q = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        loss = - norm_q * q_objective + self.config["lam"] * penalty
        return loss, {
            "refine_actor_loss": loss,
            "refine_q": q_objective,
            "manifold_penalty": penalty,
            "euclidean_penalty": euclidean_penalty,
            "delta_rms": jnp.sqrt(jnp.mean(jnp.square(delta))),
            "offset_rms": jnp.sqrt(jnp.mean(jnp.square(offset))),
            "generator_variance": generator_variance,
        }

    @staticmethod
    def _validate_config(config):
        ANQSTDFPAgent._validate_config(config)
        if config["manifold_ridge"] <= 0.0:
            raise ValueError("manifold_ridge must be positive")
        if config["lam"] < 0.0:
            raise ValueError("lam must be non-negative")

def get_config():
    config = get_anq_stdfp_config()
    config.agent_name = "mani_stdfp"
    config.manifold_ridge = 1e-2
    # Scale-free metric: shape only, with ``lam`` alone setting the strength.
    config.metric_normalize = False
    return config
