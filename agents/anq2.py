"""Standalone value-free ANQ2.

ANQ2 removes ANQ's V network and value loss.  It keeps an ensemble critic, a
local refinement actor, and a distilled execution actor.  The critic performs
expectile TD regression toward target-Q values of refined next dataset actions.
The refiner uses a target-Q-improvement-weighted displacement penalty, while
policy weights use the critic-only local improvement Q(refined) - Q(data).
Only the critic has an EMA target.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


def expectile_loss(diff, expectile):
    weight = jnp.where(diff > 0.0, expectile, 1.0 - expectile)
    return weight * jnp.square(diff)


def improvement_penalty_weight(
    improvement,
    alpha,
    improvement_clip,
    weight_min,
    weight_max,
):
    """Return a stopped exponential weight for a target-Q improvement."""
    improvement = jnp.clip(
        improvement, -improvement_clip, improvement_clip
    )
    weight = jnp.exp(-alpha * improvement)
    weight = jnp.clip(weight, weight_min, weight_max)
    return jax.lax.stop_gradient(weight)


class ANQ2Agent(flax.struct.PyTreeNode):
    """ANQ neighborhood learning using Q only, without a value function."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def _next_base_actions(self, batch, next_observations):
        """Use next dataset actions when available, otherwise the policy."""
        if "next_actions" in batch:
            if self.config["action_chunking"]:
                return jnp.reshape(
                    batch["next_actions"],
                    (batch["next_actions"].shape[0], -1),
                )
            return batch["next_actions"][..., -1, :]
        return self.network.select("actor")(next_observations).mode()

    def _action_mask(self, batch):
        if self.config["action_chunking"]:
            return jnp.repeat(batch["valid"], self.config["action_dim"], axis=-1)
        return jnp.ones_like(self._batch_actions(batch))

    @staticmethod
    def _masked_mean(values, mask):
        mask = jnp.broadcast_to(mask, values.shape)
        return (values * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    def _aggregate_qs(self, qs, mode=None):
        mode = self.config["q_agg"] if mode is None else mode
        if mode == "min":
            return qs.min(axis=0)
        if mode == "mean":
            return qs.mean(axis=0)
        raise ValueError(f"Unsupported Q aggregation: {mode}")

    def _refine_actions(self, observations, actions, params=None):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        delta = self.network.select("aux_actor")(
            inputs, params=params
        ).mode()
        delta = self.config["aux_action_scale"] * delta
        refined = jnp.clip(actions + delta, -1.0, 1.0)
        return refined, refined - actions

    def critic_loss(self, batch, grad_params):
        observations = batch["observations"]
        next_observations = batch["next_observations"][..., -1, :]
        valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]

        next_base = self._next_base_actions(batch, next_observations)
        next_refined, next_delta = self._refine_actions(
            next_observations, next_base
        )
        next_qs = self.network.select("target_critic")(
            next_observations, actions=next_refined
        )
        next_q = self._aggregate_qs(next_qs)
        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        qs = self.network.select("critic")(
            observations,
            actions=self._batch_actions(batch),
            params=grad_params,
        )
        losses = expectile_loss(
            target_q[None, ...] - qs,
            self.config["critic_expectile"],
        )
        loss = jnp.sum(
            jax.vmap(self._masked_mean, in_axes=(0, None))(losses, valid)
        )
        return loss, {
            "loss": loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q": target_q.mean(),
            "target_delta_rms": jnp.sqrt(jnp.mean(jnp.square(next_delta))),
        }

    def auxiliary_actor_loss(self, batch, grad_params):
        observations = batch["observations"]
        actions = self._batch_actions(batch)
        valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]
        action_mask = self._action_mask(batch) * valid[..., None]

        data_qs = self.network.select("target_critic")(
            observations, actions=actions
        )
        data_q = self._aggregate_qs(
            data_qs, mode=self.config["data_q_agg"]
        )
        refined, delta = self._refine_actions(
            observations, actions, params=grad_params
        )
        refined_qs = self.network.select("critic")(
            observations, actions=refined
        )
        refined_q = self._aggregate_qs(
            refined_qs, mode=self.config["refine_q_agg"]
        )
        target_refined_qs = self.network.select("target_critic")(
            observations, actions=refined
        )
        target_base_q = self._aggregate_qs(
            data_qs, mode=self.config["improvement_q_agg"]
        )
        target_refined_q = self._aggregate_qs(
            target_refined_qs, mode=self.config["improvement_q_agg"]
        )
        target_improvement = target_refined_q - target_base_q
        improvement_weight = improvement_penalty_weight(
            target_improvement,
            alpha=self.config["alpha"],
            improvement_clip=self.config["improvement_clip"],
            weight_min=self.config["aux_weight_min"],
            weight_max=self.config["aux_weight_max"],
        )
        q_objective = self._masked_mean(refined_q, valid)
        unweighted_penalty = self._masked_mean(
            jnp.square(delta), action_mask
        )
        penalty = self._masked_mean(
            improvement_weight[..., None] * jnp.square(delta), action_mask
        )
        loss = - q_objective + self.config["lam"] * penalty
        return loss, {
            "loss": loss,
            "q_objective": q_objective,
            "data_q": self._masked_mean(data_q, valid),
            "target_base_q": self._masked_mean(target_base_q, valid),
            "target_refined_q": self._masked_mean(
                target_refined_q, valid
            ),
            "improvement": self._masked_mean(target_improvement, valid),
            "improvement_weight": self._masked_mean(
                improvement_weight, valid
            ),
            "penalty": penalty,
            "unweighted_penalty": unweighted_penalty,
            "delta_rms": jnp.sqrt(
                self._masked_mean(jnp.square(delta), action_mask)
            ),
        }

    def actor_loss(self, batch, grad_params):
        observations = batch["observations"]
        actions = self._batch_actions(batch)
        valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]
        action_mask = self._action_mask(batch) * valid[..., None]

        refined, _ = self._refine_actions(observations, actions)
        refined = jax.lax.stop_gradient(refined)
        data_qs = self.network.select("target_critic")(
            observations, actions=actions
        )
        refined_qs = self.network.select("target_critic")(
            observations, actions=refined
        )
        data_q = self._aggregate_qs(
            data_qs, mode=self.config["data_q_agg"]
        )
        refined_q = self._aggregate_qs(
            refined_qs, mode=self.config["refine_q_agg"]
        )
        improvement = refined_q - data_q
        weight = jnp.exp(self.config["beta"] * improvement)
        weight = jnp.clip(
            weight,
            self.config["actor_weight_min"],
            self.config["actor_weight_max"],
        )
        weight = jax.lax.stop_gradient(weight)

        policy_actions = self.network.select("actor")(
            observations, params=grad_params
        ).mode()
        loss = self._masked_mean(
            weight[..., None] * jnp.square(policy_actions - refined),
            action_mask,
        )
        return loss, {
            "loss": loss,
            "weight": self._masked_mean(weight, valid),
            "improvement": self._masked_mean(improvement, valid),
            "target_action_rms": jnp.sqrt(
                self._masked_mean(jnp.square(refined), action_mask)
            ),
            "policy_action_rms": jnp.sqrt(
                self._masked_mean(jnp.square(policy_actions), action_mask)
            ),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        del rng
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        aux_loss, aux_info = self.auxiliary_actor_loss(batch, grad_params)
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        total_loss = critic_loss + aux_loss + actor_loss
        info = {"total_loss": total_loss}
        for prefix, values in (
            ("critic", critic_info),
            ("aux_actor", aux_info),
            ("actor", actor_info),
        ):
            info.update({f"{prefix}/{key}": value for key, value in values.items()})
        return total_loss, info

    @staticmethod
    def _update(agent, batch):
        new_rng, loss_rng = jax.random.split(agent.rng)
        new_network, info = agent.network.apply_loss_fn(
            lambda params: agent.total_loss(batch, params, loss_rng)
        )

        policy_update = agent.network.step % agent.config["policy_freq"] == 0
        target_rate = jnp.where(policy_update, agent.config["tau"], 0.0)
        source = new_network.params["modules_critic"]
        target = agent.network.params["modules_target_critic"]
        new_network.params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda p, tp: target_rate * p + (1.0 - target_rate) * tp,
            source,
            target,
        )
        info["policy_update"] = policy_update.astype(jnp.float32)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        del rng
        actions = self.network.select("actor")(observations).mode()
        actions = jnp.clip(actions, -1.0, 1.0)
        if self.config["action_chunking"]:
            actions = jnp.reshape(
                actions,
                actions.shape[:-1]
                + (self.config["horizon_length"], self.config["action_dim"]),
            )
        return actions

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
        aux_inputs = jnp.concatenate([ex_observations, full_actions], axis=-1)

        critic = Value(
            hidden_dims=config["critic_hidden_dims"],
            layer_norm=config["critic_layer_norm"],
            num_ensembles=config["num_qs"],
        )
        actor = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=True,
            final_fc_init_scale=config["actor_fc_scale"],
        )
        aux_actor = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=True,
            final_fc_init_scale=config["actor_fc_scale"],
        )
        definitions = {
            "actor": (actor, (ex_observations,)),
            "aux_actor": (aux_actor, (aux_inputs,)),
            "critic": (critic, (ex_observations, full_actions)),
            "target_critic": (
                copy.deepcopy(critic),
                (ex_observations, full_actions),
            ),
        }
        network_def = ModuleDict({k: v[0] for k, v in definitions.items()})
        params = network_def.init(
            init_rng, **{k: v[1] for k, v in definitions.items()}
        )["params"]
        params["modules_target_critic"] = params["modules_critic"]

        actor_lr = (
            optax.cosine_decay_schedule(
                config["actor_lr"], config["actor_decay_steps"]
            )
            if config["use_actor_lr_schedule"]
            else config["actor_lr"]
        )
        transforms = {
            "actor": optax.conditionally_mask(
                optax.adam(actor_lr),
                lambda step: (step + 1) % config["policy_freq"] == 0,
            ),
            "critic": optax.adam(config["critic_lr"]),
            "frozen": optax.set_to_zero(),
        }
        labels = {}
        for key, module_params in params.items():
            if key in ("modules_actor", "modules_aux_actor"):
                label = "actor"
            elif key == "modules_critic":
                label = "critic"
            else:
                label = "frozen"
            labels[key] = jax.tree_util.tree_map(lambda _: label, module_params)
        network = TrainState.create(
            network_def,
            params,
            tx=optax.multi_transform(transforms, labels),
        )
        config["ob_dims"] = ex_observations.shape
        config["action_dim"] = action_dim
        config["full_action_dim"] = full_action_dim
        return cls(rng, network, flax.core.FrozenDict(**config))

    @staticmethod
    def _validate_config(config):
        if not 0.0 < config["critic_expectile"] < 1.0:
            raise ValueError("critic_expectile must be in (0, 1)")
        for key in (
            "q_agg",
            "data_q_agg",
            "refine_q_agg",
            "improvement_q_agg",
        ):
            if config[key] not in ("min", "mean"):
                raise ValueError(f"{key} must be 'min' or 'mean'")
        if config["num_qs"] < 1:
            raise ValueError("num_qs must be positive")
        if config["lam"] < 0.0:
            raise ValueError("lam must be non-negative")
        if config["alpha"] < 0.0:
            raise ValueError("alpha must be non-negative")
        if config["improvement_clip"] <= 0.0:
            raise ValueError("improvement_clip must be positive")
        if config["beta"] < 0.0:
            raise ValueError("beta must be non-negative")
        if config["policy_freq"] < 1:
            raise ValueError("policy_freq must be positive")
        if config["aux_action_scale"] <= 0.0:
            raise ValueError("aux_action_scale must be positive")
        if config["q_eps"] <= 0.0:
            raise ValueError("q_eps must be positive")
        if config["use_actor_lr_schedule"] and config["actor_decay_steps"] < 1:
            raise ValueError("actor_decay_steps must be positive when scheduling")
        if not (
            0.0 <= config["actor_weight_min"] <= config["actor_weight_max"]
        ):
            raise ValueError("invalid actor weight clipping range")
        if not (
            0.0 < config["aux_weight_min"] <= config["aux_weight_max"]
        ):
            raise ValueError("invalid auxiliary weight clipping range")


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="anq2",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            full_action_dim=ml_collections.config_dict.placeholder(int),
            actor_hidden_dims=(512, 512, 512, 512),
            critic_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            critic_layer_norm=True,
            actor_fc_scale=0.01,
            critic_lr=3e-4,
            actor_lr=3e-4,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            num_qs=2,
            q_agg="mean",
            data_q_agg="mean",
            refine_q_agg="mean",
            improvement_q_agg="min",
            critic_expectile=0.5,
            policy_freq=1,
            use_actor_lr_schedule=False,
            actor_decay_steps=500000,
            lam=10,
            alpha=1.0,
            improvement_clip=10.0,
            aux_weight_min=0.01,
            aux_weight_max=10.0,
            beta=1.0,
            actor_weight_min=0.0,
            actor_weight_max=100.0,
            aux_action_scale=1.0,
            q_eps=1e-6,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,
        )
    )
