"""Critic-only ANQ with a learned latent-noise drift policy.

ANQ-STDFP uses the same explicit bounded action refinement as ANQ-DFP, but
retains STDFP's learned stochastic/deterministic actor in the drift decoder's
latent-noise space.  The drift decoder remains behavior-cloned; the latent
actor chooses a behavior mode and is optimized through the Q value of the
refined decoded action.
"""

from functools import partial

import jax
import jax.numpy as jnp

from agents.anq_dfp import (
    aggregate_qs,
    refine_actions,
    select_best,
    td_expectile_loss,
    validate_refinement_config,
)
from agents.stdfp import STDFPAgent, get_config as get_stdfp_config


class ANQSTDFPAgent(STDFPAgent):
    """ANQ action refinement plus an STDFP latent-noise actor."""

    def critic_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)
        next_observations = batch["next_observations"][..., -1, :]
        next_actions, target_delta = self._sample_refined_actions(
            next_observations, rng, use_target_critic=True
        )

        next_qs = self.network.select("target_critic")(
            next_observations, actions=next_actions
        )
        next_q = aggregate_qs(next_qs, self.config)
        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        qs = self.network.select("critic")(
            batch["observations"], batch_actions, params=grad_params
        )
        td_error = target_q[None, ...] - qs
        valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]
        loss_values = td_expectile_loss(
            td_error, self.config["critic_expectile"]
        )
        valid = jnp.broadcast_to(valid[None, ...], loss_values.shape)
        critic_loss = (loss_values * valid).sum() / jnp.maximum(valid.sum(), 1.0)

        return critic_loss, {
            "total_loss": critic_loss,
            "critic_loss": critic_loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q_mean": target_q.mean(),
            "target_delta_rms": jnp.sqrt(jnp.mean(jnp.square(target_delta))),
        }

    @partial(jax.jit, static_argnames=("use_target_latent",))
    def sample_drift_actions(
        self,
        observations,
        noises,
        use_target_latent=False,
        rng=None,
    ):
        model_name = (
            "target_actor_drift" if use_target_latent else "actor_drift"
        )
        base_actions = self.network.select(model_name)(observations, noises)
        base_actions = self._add_actor_output_noise(base_actions, rng)
        critic_name = "target_critic" if use_target_latent else "critic"
        refined_actions, _ = refine_actions(
            self,
            observations,
            base_actions,
            critic_name=critic_name,
            straight_through=True,
        )
        return self._safe_clip(refined_actions)

    def _sample_refined_actions(
        self, observations, rng, use_target_critic=False
    ):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"]
            if self.config["action_chunking"]
            else 1
        )
        critic_name = "target_critic" if use_target_critic else "critic"
        latent_rng, output_rng = jax.random.split(rng)

        if self._noise_actor_type() == "ddpg":
            noises = self.network.select("noise_actor")(observations)
            exploration = jnp.clip(
                jax.random.normal(latent_rng, noises.shape)
                * self.config["actor_noise"],
                -self.config["actor_noise_clip"],
                self.config["actor_noise_clip"],
            )
            model_name = (
                "target_actor_drift"
                if self.config["use_target_latent"]
                else "actor_drift"
            )
            base_actions = self.network.select(model_name)(
                observations, noises + exploration
            )
            base_actions = self._add_actor_output_noise(base_actions, output_rng)
            return refine_actions(
                self,
                observations,
                base_actions,
                critic_name=critic_name,
            )

        num_samples = self.config["best_of_n"]
        repeated_observations = jnp.repeat(
            observations[..., None, :], num_samples, axis=-2
        )
        dist = self.network.select("noise_actor")(repeated_observations)
        noises = dist.sample(seed=latent_rng) * self.config["latent_noise_scale"]
        model_name = (
            "target_actor_drift"
            if self.config["use_target_latent"]
            else "actor_drift"
        )
        base_actions = self.network.select(model_name)(
            repeated_observations, noises
        )
        base_actions = self._add_actor_output_noise(base_actions, output_rng)
        refined_actions, deltas = refine_actions(
            self,
            repeated_observations,
            base_actions,
            critic_name=critic_name,
        )
        scores = aggregate_qs(
            self.network.select(critic_name)(
                repeated_observations, actions=refined_actions
            ),
            self.config,
            mode=self.config["sample_q_agg"],
        )
        return select_best(refined_actions, scores), select_best(deltas, scores)

    @partial(jax.jit, static_argnames=("use_target_critic",))
    def sample_actions(self, observations, rng=None, use_target_critic=False):
        actions, _ = self._sample_refined_actions(
            observations, rng, use_target_critic=use_target_critic
        )
        return self._safe_clip(actions)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        q_modes = ("min", "mean", "pessimistic")
        validate_refinement_config(config, q_modes=q_modes)
        for key in ("actor_q_agg", "sample_q_agg"):
            if config[key] not in q_modes:
                raise ValueError(f"{key} must be one of {q_modes}")
        return super().create(seed, ex_observations, ex_actions, config)


def get_config():
    config = get_stdfp_config()
    config.agent_name = "anq_stdfp"
    config.action_chunking = False
    config.num_qs = 4
    config.q_agg = "min"
    config.actor_q_agg = "mean"
    config.sample_q_agg = "min"
    config.noise_scale = 0.0
    config.critic_expectile = 0.7
    config.refine_steps = 3
    config.refine_step_size = 0.05
    config.refine_radius = 0.2
    config.normalize_refine_grad = True
    config.refine_eps = 1e-6
    return config
