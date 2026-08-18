"""Standalone value-free ANQ with drift behavior cloning.

Networks: drift decoder, refinement actor, critic ensemble, and target critic.
Only the critic has an EMA target.  The drift decoder is trained strictly by
behavior cloning; the refiner predicts one bounded Q-improving action delta.
"""

import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections

from utils.drift_loss import drift_loss
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, ActorVectorField, Value
from utils.optimizers import make_optimizer


def td_expectile_loss(td_error, expectile):
    weight = jnp.where(td_error > 0.0, expectile, 1.0 - expectile)
    return weight * jnp.square(td_error)


def aggregate_qs(qs, config, mode=None):
    mode = config["q_agg"] if mode is None else mode
    if mode == "min":
        return qs.min(axis=0)
    if mode == "mean":
        return qs.mean(axis=0)
    if mode == "pessimistic":
        return qs.mean(axis=0) - config["rho"] * qs.std(axis=0)
    raise ValueError(f"Unsupported Q aggregation: {mode}")


def select_best(actions, scores):
    indices = jnp.argmax(scores, axis=-1)
    batch_shape = indices.shape
    flat_indices = indices.reshape(-1)
    flat_actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    selected = flat_actions[jnp.arange(flat_indices.size), flat_indices]
    return selected.reshape(batch_shape + (actions.shape[-1],))


def refine_actions(agent, observations, base_actions, params=None):
    """Apply one learned delta, projected into an L2 behavior neighborhood."""
    base_actions = jnp.clip(base_actions, -1.0, 1.0)
    inputs = jnp.concatenate([observations, base_actions], axis=-1)
    raw_delta = agent.network.select("refine_actor")(
        inputs, params=params
    ).mode()
    raw_norm = jnp.linalg.norm(raw_delta, axis=-1, keepdims=True)
    projection = jnp.minimum(
        1.0,
        1.0 / jnp.maximum(raw_norm, agent.config["refine_eps"]),
    )
    delta = agent.config["refine_radius"] * projection * raw_delta
    refined = jnp.clip(base_actions + delta, -1.0, 1.0)
    return refined, refined - base_actions


def validate_config(config, q_modes=("min", "mean")):
    if not 0.0 < config["critic_expectile"] < 1.0:
        raise ValueError("critic_expectile must be in (0, 1)")
    for key in ("q_agg", "refine_q_agg"):
        if config[key] not in q_modes:
            raise ValueError(f"{key} must be one of {q_modes}")
    if config["num_qs"] < 1:
        raise ValueError("num_qs must be positive")
    if config["refine_radius"] < 0.0:
        raise ValueError("refine_radius must be non-negative")
    if config["refine_lambda"] < 0.0:
        raise ValueError("refine_lambda must be non-negative")
    if config["refine_eps"] <= 0.0 or config["refine_q_eps"] <= 0.0:
        raise ValueError("refinement epsilons must be positive")
    if config["refine_fc_scale"] <= 0.0:
        raise ValueError("refine_fc_scale must be positive")
    if config["gen_per_label"] < 2:
        raise ValueError("gen_per_label must be at least 2")
    if config["actor_num_samples"] < 1:
        raise ValueError("actor_num_samples must be positive")
    if config["noise_scale"] < 0.0:
        raise ValueError("noise_scale must be non-negative")


class ANQDFPAgent(flax.struct.PyTreeNode):
    """Independent drift-BC plus learned-refinement agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def _add_output_noise(self, actions, rng):
        scale = self.config["noise_scale"]
        if scale == 0.0:
            return actions
        return actions + scale * jax.random.normal(
            rng, actions.shape, dtype=actions.dtype
        )

    def critic_loss(self, batch, grad_params, rng):
        next_observations = batch["next_observations"][..., -1, :]
        next_actions, target_delta = self._sample_refined_actions(
            next_observations,
            rng,
            force_best_of_n=True,
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
        action_dim = self.config["full_action_dim"]
        noises = jax.random.normal(
            rng, observations.shape[:-1] + (action_dim,)
        )
        base_actions = self.network.select("actor_drift")(observations, noises)
        base_actions = jax.lax.stop_gradient(jnp.clip(base_actions, -1.0, 1.0))
        refined, delta = refine_actions(
            self, observations, base_actions, params=grad_params
        )
        qs = self.network.select("critic")(observations, actions=refined)
        q = aggregate_qs(qs, self.config, mode=self.config["refine_q_agg"])

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

    def actor_loss(self, batch, grad_params, rng):
        drift_rng, refine_rng = jax.random.split(rng)
        bc_loss, info = self.drift_bc_loss(batch, grad_params, drift_rng)
        refine_loss, refine_info = self.refine_actor_loss(
            batch, grad_params, refine_rng
        )
        loss = bc_loss + refine_loss
        info.update(refine_info)
        info["actor_loss"] = loss
        return loss, info

    def actor_loss_frozen_bc(self, batch, grad_params, rng):
        loss, info = self.refine_actor_loss(batch, grad_params, rng)
        info["actor_loss"] = loss
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

    @jax.jit
    def total_loss_frozen_bc(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        actor_rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(
            batch, grad_params, critic_rng
        )
        actor_loss, actor_info = self.actor_loss_frozen_bc(
            batch, grad_params, actor_rng
        )
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

    def pretrain_bc_loss(self, batch, grad_params, rng):
        """Compute only the drift behavior-cloning loss."""
        loss, info = self.drift_bc_loss(batch, grad_params, rng)
        info["bc_pretrain_loss"] = loss
        return loss, info

    @staticmethod
    def _pretrain_bc_update(agent, batch):
        new_rng, loss_rng = jax.random.split(agent.rng)
        new_network, info = agent.network.apply_loss_fn(
            lambda params: agent.pretrain_bc_loss(batch, params, loss_rng)
        )
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def pretrain_bc_update(self, batch):
        return self._pretrain_bc_update(self, batch)

    def frozen_bc_module_keys(self):
        return tuple(
            key
            for key in ("modules_actor_drift", "modules_target_actor_drift")
            if key in self.network.params
        )

    @staticmethod
    def _update_frozen_bc(agent, batch):
        new_rng, loss_rng = jax.random.split(agent.rng)
        new_network, info = agent.network.apply_loss_fn_with_frozen_modules(
            loss_fn=lambda params: agent.total_loss_frozen_bc(
                batch, params, loss_rng
            ),
            frozen_module_keys=agent.frozen_bc_module_keys(),
        )
        source = new_network.params["modules_critic"]
        target = agent.network.params["modules_target_critic"]
        new_network.params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda p, tp: agent.config["tau"] * p
            + (1.0 - agent.config["tau"]) * tp,
            source,
            target,
        )
        info["bc_frozen"] = jnp.asarray(1.0)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_frozen_bc(self, batch):
        return self._update_frozen_bc(self, batch)

    @jax.jit
    def batch_update_frozen_bc(self, batch):
        agent, infos = jax.lax.scan(self._update_frozen_bc, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def _sample_refined_actions(
        self,
        observations,
        rng,
        force_best_of_n=False,
        critic_name="critic",
    ):
        rng = jax.random.PRNGKey(0) if rng is None else rng
        use_best_of_n = self.config["actor_type"] == "best-of-n" or force_best_of_n
        num_samples = self.config["actor_num_samples"] if use_best_of_n else 1
        latent_rng, output_rng = jax.random.split(rng)

        if use_best_of_n:
            observations = jnp.repeat(
                observations[..., None, :], num_samples, axis=-2
            )
            noises = jax.random.normal(
                latent_rng,
                observations.shape[:-1] + (self.config["full_action_dim"],),
            )
            base = self.network.select("actor_drift")(observations, noises)
            base = self._add_output_noise(base, output_rng)
            refined, delta = refine_actions(self, observations, base)
            scores = aggregate_qs(
                self.network.select(critic_name)(observations, actions=refined),
                self.config,
            )
            return select_best(refined, scores), select_best(delta, scores)

        noises = jax.random.normal(
            latent_rng,
            observations.shape[:-1] + (self.config["full_action_dim"],),
        )
        base = self.network.select("actor_drift")(observations, noises)
        base = self._add_output_noise(base, output_rng)
        return refine_actions(self, observations, base)

    @partial(
        jax.jit,
        static_argnames=("force_best_of_n", "critic_name"),
    )
    def sample_actions(
        self,
        observations,
        rng=None,
        force_best_of_n=False,
        critic_name="critic",
    ):
        actions, _ = self._sample_refined_actions(
            observations,
            rng,
            force_best_of_n=force_best_of_n,
            critic_name=critic_name,
        )
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        validate_config(config)
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        action_dim = ex_actions.shape[-1]
        full_actions = (
            jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
            if config["action_chunking"]
            else ex_actions
        )
        full_action_dim = full_actions.shape[-1]
        refine_inputs = jnp.concatenate([ex_observations, full_actions], axis=-1)

        encoders = {}
        if config["encoder"] is not None:
            encoder = encoder_modules[config["encoder"]]
            encoders = {"critic": encoder(), "actor_drift": encoder()}
        critic = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
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


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="anq_dfp",
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
            critic_expectile=0.7,
            refine_radius=0.2,
            refine_lambda=5.0,
            refine_eps=1e-6,
            refine_q_eps=1e-6,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,
            actor_type="best-of-n",
            actor_num_samples=16,
            gen_per_label=8,
            drift_temps=(0.1,),
            noise_scale=0.0,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
