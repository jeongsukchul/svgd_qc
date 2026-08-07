"""Adaptive Neighborhood-constrained Q learning (ANQ).

This is a JAX/Flax port of the NeurIPS 2025 reference implementation:
https://github.com/thu-rllab/ANQ

The agent follows the four objectives in the paper:

* expectile regression for V on actions refined by the auxiliary actor;
* Bellman regression for an ensemble Q function;
* neighborhood optimization with an advantage-adaptive penalty; and
* advantage-weighted regression of the final policy toward refined actions.

OGBench stores task completion in ``masks`` separately from dataset trajectory
termination.  The Q target deliberately uses ``masks`` and therefore does not
bootstrap through a successful state, while it may bootstrap across an
arbitrary dataset trajectory boundary exactly as the OGBench API prescribes.
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
    """Return the asymmetric squared loss used by ANQ's value network."""
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    return weight * jnp.square(diff)


class ANQAgent(flax.struct.PyTreeNode):
    """Adaptive Neighborhood-constrained Q-learning agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def _action_mask(self, batch):
        """Mask padded actions when ANQ is used experimentally with chunks."""
        if self.config["action_chunking"]:
            return jnp.repeat(batch["valid"], self.config["action_dim"], axis=-1)
        return jnp.ones_like(self._batch_actions(batch))

    @staticmethod
    def _masked_mean(values, mask):
        mask = jnp.broadcast_to(mask, values.shape)
        return (values * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    def _aggregate_qs(self, qs):
        """Aggregate the leading critic-ensemble dimension."""
        if self.config["q_agg"] == "min":
            return qs.min(axis=0)
        return qs.mean(axis=0)

    def _aux_delta(self, observations, actions, params=None):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        delta = self.network.select("aux_actor")(inputs, params=params).mode()
        return self.config["aux_action_scale"] * delta

    def _refine_actions(self, observations, actions, params=None):
        delta = self._aux_delta(observations, actions, params=params)
        return jnp.clip(actions + delta, -1.0, 1.0), delta

    def value_loss(self, batch, grad_params):
        """Equation (14): expectile regression over optimized neighborhoods."""
        observations = batch["observations"]
        actions = self._batch_actions(batch)
        valid = batch["valid"][..., -1]

        refined_actions, _ = self._refine_actions(observations, actions)
        target_qs = self.network.select("target_critic")(
            observations, actions=refined_actions
        )
        target_q = jax.lax.stop_gradient(self._aggregate_qs(target_qs))
        value = self.network.select("value")(
            observations, params=grad_params
        )
        diff = target_q - value
        loss = self._masked_mean(
            expectile_loss(diff, self.config["expectile"]), valid
        )
        return loss, {
            "loss": loss,
            "mean": value.mean(),
            "target_q": target_q.mean(),
            "advantage": diff.mean(),
        }

    def critic_loss(self, batch, grad_params):
        """Equation (15): masked n-step Bellman regression."""
        observations = batch["observations"]
        actions = self._batch_actions(batch)
        next_observations = batch["next_observations"][..., -1, :]
        valid = batch["valid"][..., -1]

        next_value = self.network.select("value")(next_observations)
        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_value
        target_q = jax.lax.stop_gradient(target_q)

        qs = self.network.select("critic")(
            observations, actions=actions, params=grad_params
        )
        per_critic_loss = jnp.square(qs - target_q[None, :])
        # The released implementation sums the four ensemble MSEs rather than
        # averaging them.  Keep that convention for optimizer fidelity.
        loss = jnp.sum(
            jax.vmap(self._masked_mean, in_axes=(0, None))(
                per_critic_loss, valid
            )
        )
        return loss, {
            "loss": loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q": target_q.mean(),
        }

    def auxiliary_actor_loss(self, batch, grad_params):
        """Equation (13), matching the released squared-displacement penalty."""
        observations = batch["observations"]
        actions = self._batch_actions(batch)
        valid = batch["valid"][..., -1]
        action_mask = self._action_mask(batch) * valid[..., None]

        value = self.network.select("value")(observations)
        data_qs = self.network.select("target_critic")(
            observations, actions=actions
        )
        data_q = self._aggregate_qs(data_qs)
        radius_weight = jnp.exp(
            self.config["alpha"] * (data_q - value)
        )
        radius_weight = jnp.clip(
            radius_weight,
            self.config["aux_weight_min"],
            self.config["aux_weight_max"],
        )
        radius_weight = jax.lax.stop_gradient(radius_weight)

        refined_actions, delta = self._refine_actions(
            observations, actions, params=grad_params
        )
        refined_qs = self.network.select("critic")(
            observations, actions=refined_actions
        )
        refined_q = self._aggregate_qs(refined_qs)
        q_scale = jax.lax.stop_gradient(
            1.0 / jnp.maximum(jnp.abs(refined_q).mean(), self.config["q_eps"])
        )

        q_objective = self._masked_mean(refined_q, valid)
        penalty = self._masked_mean(
            radius_weight[..., None] * jnp.square(delta), action_mask
        )
        loss = -q_scale * q_objective + self.config["lam"] * penalty
        return loss, {
            "loss": loss,
            "q_objective": q_objective,
            "q_scale": q_scale,
            "penalty": penalty,
            "delta_rms": jnp.sqrt(self._masked_mean(jnp.square(delta), action_mask)),
            "radius_weight": self._masked_mean(radius_weight, valid),
        }

    def actor_loss(self, batch, grad_params):
        """Equation (17): weighted regression toward neighborhood optima."""
        observations = batch["observations"]
        actions = self._batch_actions(batch)
        valid = batch["valid"][..., -1]
        action_mask = self._action_mask(batch) * valid[..., None]

        refined_actions, _ = self._refine_actions(observations, actions)
        refined_actions = jax.lax.stop_gradient(refined_actions)
        refined_qs = self.network.select("target_critic")(
            observations, actions=refined_actions
        )
        refined_q = self._aggregate_qs(refined_qs)
        value = self.network.select("value")(observations)
        actor_weight = jnp.exp(
            self.config["beta"] * (refined_q - value)
        )
        actor_weight = jnp.clip(
            actor_weight,
            self.config["actor_weight_min"],
            self.config["actor_weight_max"],
        )
        actor_weight = jax.lax.stop_gradient(actor_weight)

        policy_actions = self.network.select("actor")(
            observations, params=grad_params
        ).mode()
        loss = self._masked_mean(
            actor_weight[..., None]
            * jnp.square(policy_actions - refined_actions),
            action_mask,
        )
        return loss, {
            "loss": loss,
            "weight": self._masked_mean(actor_weight, valid),
            "target_action_rms": jnp.sqrt(
                self._masked_mean(jnp.square(refined_actions), action_mask)
            ),
            "policy_action_rms": jnp.sqrt(
                self._masked_mean(jnp.square(policy_actions), action_mask)
            ),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        del rng
        value_loss, value_info = self.value_loss(batch, grad_params)
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        aux_loss, aux_info = self.auxiliary_actor_loss(batch, grad_params)
        actor_loss, actor_info = self.actor_loss(batch, grad_params)

        total_loss = value_loss + critic_loss + aux_loss + actor_loss
        info = {"total_loss": total_loss}
        for prefix, values in (
            ("value", value_info),
            ("critic", critic_info),
            ("aux_actor", aux_info),
            ("actor", actor_info),
        ):
            info.update({f"{prefix}/{key}": value for key, value in values.items()})
        return total_loss, info

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)

        # The released ANQ code delays both policy optimization and target
        # updates.  The optimizer masks actor updates on the same schedule.
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
        del rng  # ANQ extracts a deterministic policy.
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
        if not 0.0 < config["expectile"] < 1.0:
            raise ValueError("expectile must be in (0, 1)")
        if config["lam"] < 0.0:
            raise ValueError("lam must be non-negative")
        if config["alpha"] < 0.0 or config["beta"] < 0.0:
            raise ValueError("alpha and beta must be non-negative")
        if config["policy_freq"] < 1:
            raise ValueError("policy_freq must be positive")
        if config["aux_action_scale"] <= 0.0:
            raise ValueError("aux_action_scale must be positive")
        if config["q_eps"] <= 0.0:
            raise ValueError("q_eps must be positive")
        if config["num_qs"] < 1:
            raise ValueError("num_qs must be positive")
        if config["q_agg"] not in ("min", "mean"):
            raise ValueError("q_agg must be 'min' or 'mean'")
        if config["use_actor_lr_schedule"] and config["actor_decay_steps"] < 1:
            raise ValueError("actor_decay_steps must be positive when scheduling")
        if not (
            0.0 <= config["aux_weight_min"] <= config["aux_weight_max"]
        ):
            raise ValueError("invalid auxiliary weight clipping range")
        if not (
            0.0 <= config["actor_weight_min"] <= config["actor_weight_max"]
        ):
            raise ValueError("invalid actor weight clipping range")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate(
                [ex_actions] * config["horizon_length"], axis=-1
            )
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]
        aux_inputs = jnp.concatenate([ex_observations, full_actions], axis=-1)

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=config["num_qs"],
        )
        value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=1,
        )
        actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=True,
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=config["actor_fc_scale"],
        )
        aux_actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=True,
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=config["actor_fc_scale"],
        )

        network_info = {
            "actor": (actor_def, (ex_observations,)),
            "aux_actor": (aux_actor_def, (aux_inputs,)),
            "critic": (critic_def, (ex_observations, full_actions)),
            "target_critic": (
                copy.deepcopy(critic_def),
                (ex_observations, full_actions),
            ),
            "value": (value_def, (ex_observations,)),
        }
        network_def = ModuleDict({key: value[0] for key, value in network_info.items()})
        network_args = {key: value[1] for key, value in network_info.items()}
        network_params = network_def.init(init_rng, **network_args)["params"]
        network_params["modules_target_critic"] = network_params["modules_critic"]

        if config["use_actor_lr_schedule"]:
            actor_lr = optax.cosine_decay_schedule(
                config["actor_lr"], config["actor_decay_steps"]
            )
        else:
            actor_lr = config["actor_lr"]
        actor_optimizer = optax.conditionally_mask(
            optax.adam(actor_lr),
            lambda step: (step + 1) % config["policy_freq"] == 0,
        )
        transforms = {
            "actor": actor_optimizer,
            "critic": optax.adam(config["critic_lr"]),
            "frozen": optax.set_to_zero(),
        }
        labels = {}
        for key, params in network_params.items():
            if key in ("modules_actor", "modules_aux_actor"):
                label = "actor"
            elif key in ("modules_critic", "modules_value"):
                label = "critic"
            else:
                label = "frozen"
            labels[key] = jax.tree_util.tree_map(lambda _: label, params)
        network_tx = optax.multi_transform(transforms, labels)
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        config["full_action_dim"] = full_action_dim
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return OGBench-oriented defaults with paper-faithful ANQ objectives."""
    return ml_collections.ConfigDict(
        dict(
            agent_name="anq",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            full_action_dim=ml_collections.config_dict.placeholder(int),
            # OGBench uses wider networks than the 256x2 D4RL reference.
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_layer_norm=True,
            actor_fc_scale=0.01,
            critic_lr=3e-4,
            actor_lr=3e-4,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            num_qs=4,
            q_agg="min",
            policy_freq=2,
            use_actor_lr_schedule=True,
            actor_decay_steps=500000,
            # Main ANQ hyperparameters (paper defaults).
            lam=5.0,
            alpha=1.0,
            expectile=0.7,
            beta=3.0,
            aux_weight_min=0.01,
            aux_weight_max=30.0,
            actor_weight_min=0.0,
            actor_weight_max=3.0,
            aux_action_scale=2.0,
            q_eps=1e-6,
            # Faithful single-step ANQ by default.  Setting both options below
            # enables the same objective in Q-chunk action space.
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,
        )
    )
