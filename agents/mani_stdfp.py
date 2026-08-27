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
        base, noises = self._refine_base_actions(observations, rng, train=True)
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
        # Variance stabilisation: JJ^T from a single z-draw is a 1-sample
        # estimate of E_z[JJ^T].  Averaging over k draws (implemented by
        # stacking jacobians along the latent axis with 1/sqrt(k) scaling,
        # which yields exactly the averaged covariance) shrinks noise
        # directions toward isotropy while consistent signal survives.
        k = int(self.config.get("metric_z_samples", 1))
        if k > 1:
            zs_rng = jax.random.fold_in(rng, 11)
            extra = []
            for j in range(k - 1):
                zj = jax.random.normal(
                    jax.random.fold_in(zs_rng, j), noises.shape, dtype=noises.dtype
                )
                Jj = jax.lax.stop_gradient(
                    self.generator_jacobians(observations, zj)
                )
                extra.append(jnp.nan_to_num(Jj, nan=0.0, posinf=0.0, neginf=0.0))
            jacobians = jnp.concatenate([jacobians] + extra, axis=-1) / jnp.sqrt(k)
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
        # Relative ridge: ridge_eff = rel * mean_eig(JJ^T) + absolute floor.
        # Caps the metric's anisotropy at the decoder's signal-to-mean ratio,
        # so a z-impotent decoder (antmaze: JJ^T ~ noise) self-degenerates to
        # an isotropic anchor while a z-potent one (HM/cube) keeps its shape.
        ridge = self.config["manifold_ridge"]
        rel = self.config.get("manifold_ridge_rel", 0.0)
        if rel:
            cov = generator_covariance(jacobians)
            mean_eig = jax.lax.stop_gradient(
                jnp.trace(cov, axis1=-2, axis2=-1) / cov.shape[-1]
            )
            ridge = ridge + rel * mean_eig[..., None, None]
        alpha = float(self.config.get("metric_power", 1.0))
        if alpha != 1.0 and not self.config.get("metric_invert", False):
            # Matrix-power smoothing: (JJ^T + ridge I)^-alpha interpolates
            # Euclidean (alpha=0) <-> full metric (alpha=1); anisotropy ratio
            # r is compressed to r^alpha.
            cov = generator_covariance(jacobians)
            covr = cov + ridge * jnp.eye(cov.shape[-1], dtype=cov.dtype) \
                if jnp.ndim(ridge) == 0 else cov + ridge[..., None] * jnp.eye(cov.shape[-1], dtype=cov.dtype)
            w, v = jnp.linalg.eigh(jax.lax.stop_gradient(covr))
            inv_pow = jnp.clip(w, 1e-8) ** (-alpha)
            if self.config["metric_normalize"]:
                inv_pow = inv_pow / jnp.clip(inv_pow.mean(axis=-1, keepdims=True), 1e-8)
            proj = jnp.einsum("...ij,...i->...j", v, offset) if False else jnp.einsum("...ji,...j->...i", v, offset)
            metric_offset_sq = jnp.einsum("...i,...i,...i->...", proj, inv_pow, proj)
        elif self.config.get("metric_invert", False):
            # Inverted metric: penalise TANGENT deviation, allow normal
            # deviation -- tests the hypothesis that antmaze's useful
            # refinements point off the behaviour manifold.
            cov = generator_covariance(jacobians)
            covr = cov + ridge * jnp.eye(cov.shape[-1], dtype=cov.dtype)
            if self.config["metric_normalize"]:
                covr = covr / jnp.clip(
                    jnp.trace(covr, axis1=-2, axis2=-1)[..., None, None]
                    / covr.shape[-1], 1e-8)
            metric_offset_sq = jnp.einsum(
                "...i,...ij,...j->...", offset, jax.lax.stop_gradient(covr), offset)
        else:
            metric_offset_sq = manifold_quadratic(
                offset, jacobians, ridge,
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
        lam = self.config["lam"]
        warm = self.config.get("lam_warmup_steps", 0)
        if warm:
            lam = lam * jnp.clip(self.network.step / warm, 0.0, 1.0)
        if self.config.get("delta_budget", 0.0) > 0.0:
            # hinge trust region on the (metric-weighted) per-dim offset:
            # penalty-free inside the budget, stiff wall beyond it.
            dim = offset.shape[-1]
            viol = jnp.maximum(metric_offset_sq / dim - self.config["delta_budget"], 0.0)
            loss = - norm_q * q_objective + self.config.get("budget_lam", 10.0) * (viol * batch["valid"][..., -1]).mean()
        else:
            loss = - norm_q * q_objective + lam * penalty
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
    config.manifold_ridge_rel = 0.0
    config.metric_z_samples = 1
    config.metric_invert = False
    config.metric_power = 1.0  # (JJ^T+ridge I)^-alpha; 0=Euclidean, 1=full metric  # penalise tangent instead of normal deviation  # >1: average JJ^T over k z-draws (variance stabilisation)  # >0: ridge += rel*mean_eig(JJ^T) (self-degenerating metric)
    # Scale-free metric: shape only, with ``lam`` alone setting the strength.
    config.metric_normalize = False
    return config
