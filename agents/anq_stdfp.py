"""Standalone value-free ANQ with a stochastic latent drift policy.

Networks: latent-noise actor, drift decoder, refinement actor, critic ensemble,
temperature, and target critic.  Only the critic has an EMA target.
"""

import copy
import math
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections

from utils.drift_loss import drift_loss
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    Actor,
    ActorVectorField,
    LogParam,
    MLP,
    TanhNormal,
    Value,
)
from utils.optimizers import make_optimizer


def td_expectile_loss(td_error, expectile):
    weight = jnp.where(td_error > 0.0, expectile, 1.0 - expectile)
    return weight * jnp.square(td_error)


def select_best(actions, scores):
    indices = jnp.argmax(scores, axis=-1)
    batch_shape = indices.shape
    flat_indices = indices.reshape(-1)
    flat_actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    selected = flat_actions[jnp.arange(flat_indices.size), flat_indices]
    return selected.reshape(batch_shape + (actions.shape[-1],))


class ANQSTDFPAgent(flax.struct.PyTreeNode):
    """Independent latent actor, drift BC, refiner, and expectile critic."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def _aggregate_q(self, qs, mode=None):
        mode = self.config["q_agg"] if mode is None else mode
        if mode == "min":
            return qs.min(axis=0)
        if mode == "mean":
            return qs.mean(axis=0)
        if mode == "pessimistic":
            return qs.mean(axis=0) - self.config["rho"] * qs.std(axis=0)
        raise ValueError(f"Unsupported Q aggregation: {mode}")

    def _safe_clip(self, actions):
        actions = jnp.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        return jnp.clip(actions, -1.0, 1.0)

    def _add_output_noise(self, actions, rng):
        scale = self.config["noise_scale"]
        if scale == 0.0 or rng is None:
            return actions
        return actions + scale * jax.random.normal(
            rng, actions.shape, dtype=actions.dtype
        )

    def _refine(self, observations, base_actions, params=None):
        base_actions = self._safe_clip(base_actions)
        inputs = jnp.concatenate([observations, base_actions], axis=-1)
        raw_delta = self.network.select("refine_actor")(
            inputs, params=params
        ).mode()
        norm = jnp.linalg.norm(raw_delta, axis=-1, keepdims=True)
        projection = jnp.minimum(
            1.0, 1.0 / jnp.maximum(norm, self.config["refine_eps"])
        )
        delta = self.config["refine_radius"] * projection * raw_delta
        refined = self._safe_clip(base_actions + delta)
        return refined, refined - base_actions

    def _decode(self, observations, noises, output_rng=None):
        base = self.network.select("actor_drift")(observations, noises)
        base = self._add_output_noise(base, output_rng)
        return self._refine(observations, base)

    def critic_loss(self, batch, grad_params, rng):
        next_observations = batch["next_observations"][..., -1, :]
        next_actions, target_delta = self._sample_refined_actions(
            next_observations, rng, critic_name="target_critic"
        )
        next_qs = self.network.select("target_critic")(
            next_observations, actions=next_actions
        )
        next_q = self._aggregate_q(next_qs)
        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        qs = self.network.select("critic")(
            batch["observations"],
            actions=self._batch_actions(batch),
            params=grad_params,
        )
        losses = td_expectile_loss(
            target_q[None, ...] - qs,
            self.config["critic_expectile"],
        )
        valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]
        valid = jnp.broadcast_to(valid[None, ...], losses.shape)
        loss = (losses * valid).sum() / jnp.maximum(valid.sum(), 1.0)
        return loss, {
            "critic_loss": loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q_mean": target_q.mean(),
            "target_delta_rms": jnp.sqrt(jnp.mean(jnp.square(target_delta))),
        }

    def drift_bc_loss(self, batch, grad_params, rng):
        actions = self._batch_actions(batch)
        batch_size, action_dim = actions.shape
        observations = jnp.repeat(
            batch["observations"], self.config["gen_per_label"], axis=0
        )
        noises = jax.random.normal(
            rng,
            (batch_size * self.config["gen_per_label"], action_dim),
        )
        generated = self.network.select("actor_drift")(
            observations, noises, params=grad_params
        ).reshape(batch_size, self.config["gen_per_label"], action_dim)
        losses, drift_info = drift_loss(
            gen=generated,
            fixed_pos=actions[:, None, :],
            R_list=tuple(self.config["drift_temps"]),
        )
        loss = losses.mean()
        info = {
            "actor_drift_loss": loss,
            "drift_scale": drift_info.get("scale", 0.0),
            "generated_to_data_mse": jnp.mean(
                jnp.square(generated - actions[:, None, :])
            ),
        }
        for key, value in drift_info.items():
            if key.startswith("loss_"):
                info[f"drift_{key}"] = value
        return loss, info

    def refine_actor_loss(self, batch, grad_params, rng):
        observations = batch["observations"]
        noises = jax.random.normal(
            rng,
            observations.shape[:-1] + (self.config["full_action_dim"],),
        )
        base = self.network.select("actor_drift")(observations, noises)
        base = jax.lax.stop_gradient(self._safe_clip(base))
        refined, delta = self._refine(
            observations, base, params=grad_params
        )
        qs = self.network.select("critic")(observations, actions=refined)
        q = self._aggregate_q(qs, mode=self.config["refine_q_agg"])

        valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]
        denom = jnp.maximum(valid.sum(), 1.0)
        q_objective = (q * valid).sum() / denom
        delta_sq = jnp.sum(jnp.square(delta), axis=-1)
        penalty = (delta_sq * valid).sum() / denom
        q_scale = jax.lax.stop_gradient(
            1.0 / jnp.maximum(jnp.abs(q).mean(), self.config["refine_q_eps"])
        )
        loss = -q_scale * q_objective + self.config["refine_lambda"] * penalty
        return loss, {
            "refine_actor_loss": loss,
            "refine_q": q_objective,
            "refine_penalty": penalty,
            "refine_delta_rms": jnp.sqrt(jnp.mean(jnp.square(delta))),
            "refine_delta_norm": jnp.linalg.norm(delta, axis=-1).mean(),
        }

    def latent_actor_loss(self, batch, grad_params, rng):
        observations = batch["observations"]
        dist = self.network.select("noise_actor")(
            observations, params=grad_params
        )
        raw_noises = dist.sample(seed=rng)
        scale = self.config["latent_noise_scale"]
        noises = raw_noises * scale
        log_probs = dist.log_prob(raw_noises) - self.config[
            "full_action_dim"
        ] * jnp.log(scale)
        actions, _ = self._decode(observations, noises)
        qs = self.network.select("critic")(observations, actions=actions)
        q = self._aggregate_q(qs, mode=self.config["actor_q_agg"])

        alpha = self.network.select("noise_alpha")()
        train_alpha = self.network.select("noise_alpha")(params=grad_params)
        entropy = -log_probs
        if self.config["noise_regularizer"] == "entropy":
            policy_loss = (alpha * log_probs - q).mean()
            alpha_loss = (
                train_alpha
                * (
                    jax.lax.stop_gradient(entropy)
                    - self.config["noise_target_entropy"]
                )
            ).mean()
            kl = jnp.zeros_like(log_probs)
        else:
            log_prior = -0.5 * (
                jnp.square(noises) + math.log(2.0 * math.pi)
            ).sum(axis=-1)
            kl = log_probs - log_prior
            policy_loss = (alpha * kl - q).mean()
            alpha_loss = (
                train_alpha
                * (
                    self.config["noise_target_kl"]
                    - jax.lax.stop_gradient(kl)
                )
            ).mean()

        return policy_loss + alpha_loss, {
            "latent_policy_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "alpha": train_alpha,
            "latent_q": q.mean(),
            "latent_entropy": entropy.mean(),
            "latent_kl": kl.mean(),
            "noise_abs_mean": jnp.abs(noises).mean(),
            "noise_std": noises.std(),
        }

    def actor_loss(self, batch, grad_params, rng):
        drift_rng, refine_rng, latent_rng = jax.random.split(rng, 3)
        bc_loss, info = self.drift_bc_loss(batch, grad_params, drift_rng)
        refine_loss, refine_info = self.refine_actor_loss(
            batch, grad_params, refine_rng
        )
        latent_loss, latent_info = self.latent_actor_loss(
            batch, grad_params, latent_rng
        )
        loss = bc_loss + refine_loss + latent_loss
        info.update(refine_info)
        info.update(latent_info)
        info["total_loss"] = loss
        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        actor_rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(
            batch, grad_params, critic_rng
        )
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info = {"total_loss": critic_loss + actor_loss}
        info.update({f"critic/{k}": v for k, v in critic_info.items()})
        info.update({f"actor/{k}": v for k, v in actor_info.items()})
        return critic_loss + actor_loss, info

    @staticmethod
    def _update(agent, batch):
        new_rng, loss_rng = jax.random.split(agent.rng)
        new_network, info = agent.network.apply_loss_fn(
            lambda params: agent.total_loss(batch, params, loss_rng)
        )
        source = new_network.params["modules_critic"]
        target = agent.network.params["modules_target_critic"]
        new_network.params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda p, tp: agent.config["tau"] * p
            + (1.0 - agent.config["tau"]) * tp,
            source,
            target,
        )
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def _sample_refined_actions(self, observations, rng, critic_name="critic"):
        rng = jax.random.PRNGKey(0) if rng is None else rng
        latent_rng, output_rng = jax.random.split(rng)
        observations = jnp.repeat(
            observations[..., None, :], self.config["best_of_n"], axis=-2
        )
        dist = self.network.select("noise_actor")(observations)
        noises = (
            dist.sample(seed=latent_rng) * self.config["latent_noise_scale"]
        )
        base = self.network.select("actor_drift")(observations, noises)
        base = self._add_output_noise(base, output_rng)
        refined, delta = self._refine(observations, base)
        scores = self._aggregate_q(
            self.network.select(critic_name)(observations, actions=refined),
            mode=self.config["sample_q_agg"],
        )
        return select_best(refined, scores), select_best(delta, scores)

    @partial(jax.jit, static_argnames=("critic_name",))
    def sample_actions(self, observations, rng=None, critic_name="critic"):
        actions, _ = self._sample_refined_actions(
            observations, rng, critic_name=critic_name
        )
        return self._safe_clip(actions)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        cls._validate_config(config)
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        action_dim = ex_actions.shape[-1]
        full_actions = (
            jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
            if config["action_chunking"]
            else ex_actions
        )
        full_action_dim = full_actions.shape[-1]
        refine_inputs = jnp.concatenate([ex_observations, full_actions], axis=-1)
        if config["noise_target_entropy"] is None:
            config["noise_target_entropy"] = (
                config["target_multiplier"] * full_action_dim
            )
        if config["noise_target_kl"] is None:
            config["noise_target_kl"] = (
                config["target_multiplier"] * full_action_dim
            )

        encoders = {}
        if config["encoder"] is not None:
            encoder = encoder_modules[config["encoder"]]
            encoders = {
                "critic": encoder(),
                "noise_actor": encoder(),
                "actor_drift": encoder(),
            }
        critic = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
        )
        noise_actor_base = partial(
            MLP,
            hidden_dims=config["actor_hidden_dims"],
            activate_final=True,
            layer_norm=config["actor_layer_norm"],
        )
        noise_actor = TanhNormal(
            noise_actor_base,
            full_action_dim,
            state_dependent_std=config["noise_state_dependent_std"],
            encoder=encoders.get("noise_actor"),
        )
        actor_drift = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_drift"),
        )
        refine_actor = Actor(
            hidden_dims=config["refine_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["refine_layer_norm"],
            tanh_squash=True,
            final_fc_init_scale=config["refine_fc_scale"],
        )
        definitions = {
            "critic": (critic, (ex_observations, full_actions)),
            "target_critic": (
                copy.deepcopy(critic),
                (ex_observations, full_actions),
            ),
            "noise_actor": (noise_actor, (ex_observations,)),
            "noise_alpha": (
                LogParam(init_value=config["noise_init_temp"]),
                (),
            ),
            "actor_drift": (actor_drift, (ex_observations, full_actions)),
            "refine_actor": (refine_actor, (refine_inputs,)),
        }
        network_def = ModuleDict({k: v[0] for k, v in definitions.items()})
        params = network_def.init(
            init_rng, **{k: v[1] for k, v in definitions.items()}
        )["params"]
        params["modules_target_critic"] = params["modules_critic"]
        network = TrainState.create(
            network_def,
            params,
            tx=make_optimizer(config["optimizer"], config["lr"]),
        )
        config["ob_dims"] = ex_observations.shape
        config["action_dim"] = action_dim
        config["full_action_dim"] = full_action_dim
        return cls(rng, network, flax.core.FrozenDict(**config))

    @staticmethod
    def _validate_config(config):
        q_modes = ("min", "mean", "pessimistic")
        for key in ("q_agg", "refine_q_agg", "actor_q_agg", "sample_q_agg"):
            if config[key] not in q_modes:
                raise ValueError(f"{key} must be one of {q_modes}")
        if not 0.0 < config["critic_expectile"] < 1.0:
            raise ValueError("critic_expectile must be in (0, 1)")
        if config["noise_regularizer"] not in ("entropy", "kl"):
            raise ValueError("noise_regularizer must be 'entropy' or 'kl'")
        if config["num_qs"] < 1 or config["best_of_n"] < 1:
            raise ValueError("num_qs and best_of_n must be positive")
        if config["gen_per_label"] < 2:
            raise ValueError("gen_per_label must be at least 2")
        if config["latent_noise_scale"] <= 0.0:
            raise ValueError("latent_noise_scale must be positive")
        if config["noise_scale"] < 0.0:
            raise ValueError("noise_scale must be non-negative")
        if config["refine_radius"] < 0.0:
            raise ValueError("refine_radius must be non-negative")
        if config["refine_lambda"] < 0.0:
            raise ValueError("refine_lambda must be non-negative")
        if config["refine_eps"] <= 0.0 or config["refine_q_eps"] <= 0.0:
            raise ValueError("refinement epsilons must be positive")


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="anq_stdfp",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            full_action_dim=ml_collections.config_dict.placeholder(int),
            optimizer="adam",
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            refine_hidden_dims=(512, 512),
            actor_layer_norm=False,
            layer_norm=True,
            refine_layer_norm=False,
            refine_fc_scale=0.01,
            discount=0.99,
            tau=0.005,
            num_qs=4,
            q_agg="min",
            refine_q_agg="min",
            actor_q_agg="mean",
            sample_q_agg="min",
            rho=0.5,
            critic_expectile=0.7,
            refine_radius=0.2,
            refine_lambda=5.0,
            refine_eps=1e-6,
            refine_q_eps=1e-6,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,
            best_of_n=4,
            gen_per_label=8,
            drift_temps=(0.1,),
            noise_regularizer="kl",
            noise_state_dependent_std=False,
            noise_target_entropy=None,
            noise_target_kl=None,
            target_multiplier=0.5,
            noise_init_temp=1.0,
            noise_scale=0.0,
            latent_noise_scale=1.0,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
