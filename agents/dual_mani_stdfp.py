"""Primal-dual manifold ANQ-STDFP with an automatic refinement multiplier.

The refinement actor solves the constrained problem

    maximize   E[Q(s, a_refined)]
    subject to E[w(s, a) * d_G(a_refined, a_base)^2] <= epsilon,

where ``w`` is ANQ's stopped improvement weight and ``d_G`` is the local
generator-induced metric from :mod:`agents.mani_stdfp`.  The nonnegative
Lagrange multiplier is updated by projected dual ascent, independently of the
network optimizer.
"""

from typing import Any

import jax
import jax.numpy as jnp

from agents.mani_stdfp import (
    ManiSTDFPAgent,
    get_config as get_mani_stdfp_config,
    manifold_quadratic,
)
from agents.anq_stdfp import improvement_penalty_weight


class DualManiSTDFPAgent(ManiSTDFPAgent):
    """Manifold STDFP with a projected dual update for the refine budget."""

    lam: Any
    constraint_ema: Any

    def refine_actor_loss(self, batch, grad_params, rng):
        observations = batch["observations"]
        base, noises = self._refine_base_actions(observations, rng)
        refined, delta = self._refine(
            observations, base, params=grad_params
        )
        qs = self.network.select("critic")(observations, actions=refined)
        q = self._aggregate_q(qs, mode=self.config["refine_q_agg"])

        target_base_qs = self.network.select("target_critic")(
            observations, actions=base
        )
        target_refined_qs = self.network.select("target_critic")(
            observations, actions=refined
        )
        target_base_q = self._aggregate_q(
            target_base_qs, mode=self.config["base_q_agg"]
        )
        target_refined_q = self._aggregate_q(
            target_refined_qs, mode=self.config["improvement_q_agg"]
        )
        target_improvement = target_refined_q - target_base_q
        improvement_weight = improvement_penalty_weight(
            target_improvement,
            alpha=self.config["alpha"],
            improvement_clip=self.config["improvement_clip"],
            weight_min=self.config["refine_weight_min"],
            weight_max=self.config["refine_weight_max"],
        )

        # The drift generator describes the constraint geometry; the dual and
        # refiner updates must not train it through this Jacobian.
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
        denom = jnp.maximum(valid.sum(), 1.0)
        q_objective = (q * valid).sum() / denom
        unweighted_penalty = (metric_delta_sq * valid).sum() / denom
        constraint_value = (
            improvement_weight * metric_delta_sq * valid
        ).sum() / denom
        constraint_violation = constraint_value - self.config["epsilon"]

        # ``lam`` is agent state, not part of grad_params.  Thus this term only
        # supplies the primal refiner gradient; projected ascent is done after
        # the network update in ``_finish_update``.
        loss = -q_objective + self.lam * constraint_violation

        euclidean_delta_sq = jnp.sum(jnp.square(delta), axis=-1)
        euclidean_penalty = (euclidean_delta_sq * valid).sum() / denom
        generator_variance = (
            jnp.sum(jnp.square(jacobians), axis=(-2, -1)) * valid
        ).sum() / (denom * jacobians.shape[-2])

        return loss, {
            "refine_actor_loss": loss,
            "refine_q": q_objective,
            "target_base_q": (target_base_q * valid).sum() / denom,
            "target_refined_q": (target_refined_q * valid).sum() / denom,
            "improvement": (target_improvement * valid).sum() / denom,
            "improvement_weight": (
                (improvement_weight * valid).sum() / denom
            ),
            "penalty": constraint_value,
            "unweighted_penalty": unweighted_penalty,
            "manifold_penalty": constraint_value,
            "euclidean_penalty": euclidean_penalty,
            "constraint_value": constraint_value,
            "constraint_violation": constraint_violation,
            "epsilon": jnp.asarray(self.config["epsilon"]),
            "lam": self.lam,
            "delta_rms": jnp.sqrt(jnp.mean(jnp.square(delta))),
            "generator_variance": generator_variance,
        }

    def dual_update(self, constraint_value):
        """Return the projected multiplier and smoothed constraint estimate."""
        ema_coef = self.config["dual_ema_coef"]
        new_constraint_ema = (
            (1.0 - ema_coef) * self.constraint_ema
            + ema_coef * jax.lax.stop_gradient(constraint_value)
        )
        violation = new_constraint_ema - self.config["epsilon"]
        new_lam = jnp.clip(
            self.lam + self.config["dual_lr"] * violation,
            self.config["lam_min"],
            self.config["lam_max"],
        )
        return new_lam, new_constraint_ema, violation

    @staticmethod
    def _finish_update(agent, new_network, info, new_rng):
        source = new_network.params["modules_critic"]
        target = agent.network.params["modules_target_critic"]
        new_network.params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda p, tp: agent.config["tau"] * p
            + (1.0 - agent.config["tau"]) * tp,
            source,
            target,
        )

        constraint_value = info["actor/constraint_value"]
        new_lam, new_constraint_ema, violation = agent.dual_update(
            constraint_value
        )
        info["dual/lam"] = new_lam
        info["dual/constraint_value"] = constraint_value
        info["dual/constraint_ema"] = new_constraint_ema
        info["dual/epsilon"] = jnp.asarray(agent.config["epsilon"])
        info["dual/violation"] = violation
        info["dual/at_min"] = (
            new_lam <= agent.config["lam_min"]
        ).astype(jnp.float32)
        info["dual/at_max"] = (
            new_lam >= agent.config["lam_max"]
        ).astype(jnp.float32)

        return agent.replace(
            network=new_network,
            rng=new_rng,
            lam=new_lam,
            constraint_ema=new_constraint_ema,
        ), info

    @staticmethod
    def _update(agent, batch):
        new_rng, loss_rng = jax.random.split(agent.rng)
        new_network, info = agent.network.apply_loss_fn(
            lambda params: agent.total_loss(batch, params, loss_rng)
        )
        return DualManiSTDFPAgent._finish_update(
            agent, new_network, info, new_rng
        )

    @staticmethod
    def _update_frozen_bc(agent, batch):
        new_rng, loss_rng = jax.random.split(agent.rng)
        new_network, info = agent.network.apply_loss_fn_with_frozen_modules(
            loss_fn=lambda params: agent.total_loss(
                batch, params, loss_rng
            ),
            frozen_module_keys=agent.frozen_bc_module_keys(),
        )
        info["bc_frozen"] = jnp.asarray(1.0)
        return DualManiSTDFPAgent._finish_update(
            agent, new_network, info, new_rng
        )

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        cls._validate_config(config)
        base_agent = ManiSTDFPAgent.create(
            seed, ex_observations, ex_actions, config
        )
        dtype = ex_actions.dtype
        return cls(
            rng=base_agent.rng,
            network=base_agent.network,
            config=base_agent.config,
            lam=jnp.asarray(config["lam"], dtype=dtype),
            constraint_ema=jnp.asarray(config["epsilon"], dtype=dtype),
        )

    @staticmethod
    def _validate_config(config):
        ManiSTDFPAgent._validate_config(config)
        if config["epsilon"] < 0.0:
            raise ValueError("epsilon must be non-negative")
        if config["dual_lr"] <= 0.0:
            raise ValueError("dual_lr must be positive")
        if not 0.0 < config["dual_ema_coef"] <= 1.0:
            raise ValueError("dual_ema_coef must be in (0, 1]")
        if config["lam_min"] < 0.0:
            raise ValueError("lam_min must be non-negative")
        if config["lam_max"] < config["lam_min"]:
            raise ValueError("lam_max must be at least lam_min")
        if not config["lam_min"] <= config["lam"] <= config["lam_max"]:
            raise ValueError("initial lam must lie in [lam_min, lam_max]")


def get_config():
    config = get_mani_stdfp_config()
    config.agent_name = "dual_mani_stdfp"
    config.epsilon = 0.2
    config.dual_lr = 1e-3
    config.dual_ema_coef = 0.05
    config.lam_min = 0.0
    config.lam_max = 1e4
    return config
