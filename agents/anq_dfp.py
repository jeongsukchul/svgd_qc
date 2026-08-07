"""Critic-only ANQ with a drift behavior policy.

This variant removes ANQ's learned value function, auxiliary delta actor, and
final weighted-regression actor.  A DFP model is trained only by drift behavior
cloning.  At target/action time, its decoded behavior action is refined by a
small number of bounded Q-gradient steps, which explicitly implements ANQ's
inner neighborhood optimization without another learned network.  The Q
ensemble is trained with an expectile TD loss.
"""

from functools import partial

import jax
import jax.numpy as jnp

from agents.dfp import DFPAgent, get_config as get_dfp_config


def td_expectile_loss(td_error, expectile):
    """Asymmetric squared TD error used by the critic-only variants."""
    weight = jnp.where(td_error > 0.0, expectile, 1.0 - expectile)
    return weight * jnp.square(td_error)


def aggregate_qs(qs, config, mode=None):
    """Aggregate the leading critic-ensemble dimension."""
    mode = config["q_agg"] if mode is None else mode
    if mode == "min":
        return qs.min(axis=0)
    if mode == "mean":
        return qs.mean(axis=0)
    if mode == "pessimistic":
        return qs.mean(axis=0) - config["rho"] * qs.std(axis=0)
    raise ValueError(f"Unsupported Q aggregation: {mode}")


def select_best(actions, scores):
    """Select one candidate from the penultimate action dimension."""
    indices = jnp.argmax(scores, axis=-1)
    batch_shape = indices.shape
    flat_indices = indices.reshape(-1)
    flat_actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    selected = flat_actions[jnp.arange(flat_indices.size), flat_indices]
    return selected.reshape(batch_shape + (actions.shape[-1],))


def refine_actions(
    agent,
    observations,
    base_actions,
    critic_name="critic",
    straight_through=False,
):
    """Perform projected Q-gradient ascent around drift-decoded actions.

    The total displacement is projected into an L2 ball of radius
    ``refine_radius`` around the behavior action.  With ``straight_through``,
    the numerical refinement is detached while retaining an identity gradient
    through the base action; this lets ANQ-STDFP train its latent policy without
    expensive second-order derivatives through the inner optimization.
    """
    base_actions = jnp.clip(base_actions, -1.0, 1.0)
    radius = agent.config["refine_radius"]
    if radius == 0.0:
        return base_actions, jnp.zeros_like(base_actions)

    refined = base_actions
    for _ in range(agent.config["refine_steps"]):

        def q_objective(candidate_actions):
            candidate_actions = jnp.clip(candidate_actions, -1.0, 1.0)
            qs = agent.network.select(critic_name)(
                observations, actions=candidate_actions
            )
            return aggregate_qs(qs, agent.config).sum()

        action_grad = jax.grad(q_objective)(refined)
        action_grad = jnp.nan_to_num(action_grad)
        if agent.config["normalize_refine_grad"]:
            grad_norm = jnp.linalg.norm(action_grad, axis=-1, keepdims=True)
            action_grad = action_grad / jnp.maximum(
                grad_norm, agent.config["refine_eps"]
            )

        refined = jnp.clip(
            refined + agent.config["refine_step_size"] * action_grad,
            -1.0,
            1.0,
        )
        delta = refined - base_actions
        delta_norm = jnp.linalg.norm(delta, axis=-1, keepdims=True)
        projection = jnp.minimum(
            1.0,
            radius / jnp.maximum(delta_norm, agent.config["refine_eps"]),
        )
        refined = jnp.clip(base_actions + projection * delta, -1.0, 1.0)

    numerical_refined = refined
    delta = numerical_refined - base_actions
    if straight_through:
        refined = base_actions + jax.lax.stop_gradient(delta)
    return refined, jax.lax.stop_gradient(delta)


def validate_refinement_config(config, q_modes=("min", "mean")):
    if not 0.0 < config["critic_expectile"] < 1.0:
        raise ValueError("critic_expectile must be in (0, 1)")
    if config["q_agg"] not in q_modes:
        raise ValueError(f"q_agg must be one of {q_modes}")
    if config["refine_steps"] < 1:
        raise ValueError("refine_steps must be positive")
    if config["refine_step_size"] <= 0.0:
        raise ValueError("refine_step_size must be positive")
    if config["refine_radius"] < 0.0:
        raise ValueError("refine_radius must be non-negative")
    if config["refine_eps"] <= 0.0:
        raise ValueError("refine_eps must be positive")


class ANQDFPAgent(DFPAgent):
    """ANQ neighborhood refinement over a behavior-cloned DFP decoder."""

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def critic_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)
        next_observations = batch["next_observations"][..., -1, :]
        next_actions, target_delta = self._sample_refined_actions(
            next_observations,
            rng,
            use_q_bon=True,
            critic_name="target_critic",
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
            batch["observations"], actions=batch_actions, params=grad_params
        )
        td_error = target_q[None, ...] - qs
        valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]
        loss_values = td_expectile_loss(
            td_error, self.config["critic_expectile"]
        )
        valid = jnp.broadcast_to(valid[None, ...], loss_values.shape)
        critic_loss = (loss_values * valid).sum() / jnp.maximum(valid.sum(), 1.0)

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q_mean": target_q.mean(),
            "target_delta_rms": jnp.sqrt(jnp.mean(jnp.square(target_delta))),
        }

    def _sample_refined_actions(
        self,
        observations,
        rng,
        use_q_bon=False,
        critic_name="critic",
    ):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"]
            if self.config["action_chunking"]
            else 1
        )
        use_bon = self.config["actor_type"] == "best-of-n" or use_q_bon
        num_samples = self.config["actor_num_samples"] if use_bon else 1
        latent_rng, output_rng = jax.random.split(rng)

        if use_bon:
            noises = jax.random.normal(
                latent_rng,
                observations.shape[:-1] + (num_samples, action_dim),
            )
            repeated_observations = jnp.repeat(
                observations[..., None, :], num_samples, axis=-2
            )
            base_actions = self.network.select("actor_drift")(
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
            )
            return select_best(refined_actions, scores), select_best(deltas, scores)

        noises = jax.random.normal(
            latent_rng, observations.shape[:-1] + (action_dim,)
        )
        base_actions = self.network.select("actor_drift")(observations, noises)
        base_actions = self._add_actor_output_noise(base_actions, output_rng)
        return refine_actions(
            self, observations, base_actions, critic_name=critic_name
        )

    @partial(
        jax.jit,
        static_argnames=("use_q_bon", "critic_name"),
    )
    def sample_actions(
        self,
        observations,
        rng=None,
        use_q_bon=False,
        critic_name="critic",
    ):
        actions, _ = self._sample_refined_actions(
            observations,
            rng,
            use_q_bon=use_q_bon,
            critic_name=critic_name,
        )
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        validate_refinement_config(config)
        return super().create(seed, ex_observations, ex_actions, config)


def get_config():
    config = get_dfp_config()
    config.agent_name = "anq_dfp"
    config.action_chunking = False
    config.num_qs = 4
    config.q_agg = "min"
    config.actor_type = "best-of-n"
    config.noise_scale = 0.0
    config.critic_expectile = 0.7
    config.refine_steps = 3
    config.refine_step_size = 0.05
    config.refine_radius = 0.2
    config.normalize_refine_grad = True
    config.refine_eps = 1e-6
    return config
