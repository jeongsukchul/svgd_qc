"""
Drift Policy 

- Using 1-step drift model (no multi-step flow)
- No distillation loss
- Drift loss with Q-learning
"""

import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.log_kde_loss import grouped_log_kde_loss
from utils.networks import ActorVectorField, Value
from utils.drift_loss import drift_loss
from utils.optimizers import make_optimizer


def dfp_log_kde_loss(generated, positives, bandwidth):
    """Return the Gaussian leave-one-out log-KDE loss for each group."""
    return grouped_log_kde_loss(generated, positives, bandwidth)


class DFPAgent(flax.struct.PyTreeNode):
    """Drift Policy agent with drift model."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the Drift critic loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)
        next_obs = batch["next_observations"][..., -1, :]
        next_actions = self.sample_actions(next_obs, sample_rng, use_q_bon=True)

        next_qs = self.network.select("target_critic")(next_obs, actions=next_actions)
        if self.config["q_agg"] == "min":
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_q

        q = self.network.select("critic")(
            batch["observations"], actions=batch_actions, params=grad_params
        )
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        info = {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "tgt_q_mean": next_q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }
        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute actor loss with drift model."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_size, action_dim = batch_actions.shape

        output_noise_rng, drift_rng = jax.random.split(rng)

        # Generate multiple samples per observation.
        gen_per_label = self.config.get("gen_per_label", 8)
        obs_repeated = jnp.repeat(batch["observations"], gen_per_label, axis=0)
        drift_noises = jax.random.normal(
            drift_rng, (batch_size * gen_per_label, action_dim)
        )

        # Get actions from drift model
        drift_actions_all = self.network.select("actor_drift")(
            obs_repeated, drift_noises, params=grad_params
        )
        # drift_actions_all = self._add_actor_output_noise(
        #     drift_actions_all, output_noise_rng
        # )
        # drift_actions_all = jnp.clip(drift_actions_all, -1, 1)
        # Reshape to [B, gen_per_label, action_dim]
        gen_samples = drift_actions_all.reshape(batch_size, gen_per_label, action_dim)

        # Positive samples: dataset actions [B, 1, action_dim]
        pos_samples = jnp.expand_dims(batch_actions, axis=1)

        drift_backend = self.config.get("drift_backend", "drift_loss")
        if drift_backend == "log_kde":
            behavior_loss_val, behavior_info = dfp_log_kde_loss(
                generated=gen_samples,
                positives=pos_samples,
                bandwidth=self.config["log_kde_bandwidth"],
            )
        else:
            # drift_loss already normalizes internally by scale_inputs.
            drift_temps = self.config.get("drift_temps", (0.1,))
            if isinstance(drift_temps, (int, float)):
                drift_temps = (float(drift_temps),)
            else:
                drift_temps = tuple(drift_temps)
            behavior_loss_val, behavior_info = drift_loss(
                gen=gen_samples,
                fixed_pos=pos_samples,
                R_list=drift_temps,
            )

        # Apply alpha to match Q loss magnitude.
        alpha = self.config.get("alpha", 1.0)
        actor_drift_loss = alpha * behavior_loss_val.mean()

        # Total loss (no distillation)
        actor_loss = actor_drift_loss #+ q_loss

        info = dict(
            actor_loss=actor_loss,
            actor_drift_loss=actor_drift_loss,
        )
        if drift_backend == "log_kde":
            info["log_kde_loss"] = behavior_loss_val.mean()
            info["log_kde_log_p"] = behavior_info["per_group_log_p"].mean()
            info["log_kde_log_q"] = behavior_info["per_group_log_q"].mean()
        else:
            info["drift_scale"] = behavior_info.get("scale", 0.0)
            # Add per-temperature losses.
            for key, val in behavior_info.items():
                if key.startswith("loss_"):
                    info[f"drift_{key}"] = val
        return actor_loss, info


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    @jax.jit
    def total_loss_frozen_bc(self, batch, grad_params, rng=None):
        """Compute offline RL loss without updating or evaluating BC loss."""
        rng = rng if rng is not None else self.rng
        critic_loss, critic_info = self.critic_loss(batch, grad_params, rng)
        info = {"total_loss": critic_loss}
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v
        return critic_loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params


    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def pretrain_bc_loss(self, batch, grad_params, rng):
        """Compute the behavior-cloning-only drift loss."""
        loss, info = self.actor_loss(batch, grad_params, rng)
        info["bc_pretrain_loss"] = loss
        return loss, info

    @staticmethod
    def _pretrain_bc_update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.pretrain_bc_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def pretrain_bc_update(self, batch):
        return self._pretrain_bc_update(self, batch)

    def frozen_bc_module_keys(self):
        return tuple(
            key
            for key in (
                "modules_actor_drift",
                "modules_target_actor_drift",
                "modules_actor_dfm",
                "modules_target_actor_dfm",
            )
            if key in self.network.params
        )

    @staticmethod
    def _update_frozen_bc(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss_frozen_bc(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn_with_frozen_modules(
            loss_fn=loss_fn,
            frozen_module_keys=agent.frozen_bc_module_keys(),
        )
        agent.target_update(new_network, "critic")
        info["bc_frozen"] = jnp.asarray(1.0)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_frozen_bc(self, batch):
        return self._update_frozen_bc(self, batch)

    @jax.jit
    def batch_update_frozen_bc(self, batch):
        agent, infos = jax.lax.scan(self._update_frozen_bc, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def _add_actor_output_noise(self, actions, rng):
        noise_scale = self.config.get("noise_scale", 0.0)
        if noise_scale == 0.0:
            return actions
        return actions + jax.random.normal(
            rng, actions.shape, dtype=actions.dtype
        ) * noise_scale

    def sample_noises(self, obs, rng):
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        return jax.random.normal(
            rng,
            (
                *obs.shape[: -len(self.config["ob_dims"])],
                full_action_dim,
            ),
        )

    def _score_actions(self, observations, actions):
        qs = self.network.select("critic")(observations, actions)
        if self.config["q_agg"] == "mean":
            return qs.mean(axis=0)
        return qs.min(axis=0)

    def _select_best_bon_action(self, actions, q_values):
        indices = jnp.argmax(q_values, axis=-1)
        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        return jnp.reshape(actions, (-1, actions.shape[-2], actions.shape[-1]))[
            jnp.arange(bsize), indices, :
        ].reshape(bshape + (actions.shape[-1],))

    @partial(jax.jit, static_argnames=("use_q_bon",))
    def sample_actions(self, observations, rng=None, use_q_bon=False):
        """Sample actions with either direct actor or best-of-n search."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        actor_type = self.config.get("actor_type", "distill-ddpg")
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )

        if actor_type == "best-of-n":
            num_samples = self.config['actor_num_samples']
            rng, init_noise_rng = jax.random.split(rng)
            noises = jax.random.normal(
                init_noise_rng,
                (
                    *observations.shape[: -len(self.config["ob_dims"])],
                    num_samples,
                    full_action_dim,
                ),
            )
            obs_rep = jnp.repeat(observations[..., None, :], num_samples, axis=-2)
            actions = self.network.select("actor_drift")(obs_rep, noises)
            actions = self._add_actor_output_noise(actions, rng)
            actions = jnp.clip(actions, -1, 1)

            q = self._score_actions(obs_rep, actions)
            return self._select_best_bon_action(actions, q)

        input_noise_rng = rng
        output_noise_rng = rng
        if self.config.get("noise_scale", 0.0) != 0.0:
            input_noise_rng, output_noise_rng = jax.random.split(rng)
        noises = self.sample_noises(observations, input_noise_rng)
        actions = self.network.select("actor_drift")(observations, noises)
        actions = self._add_actor_output_noise(actions, output_noise_rng)
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        drift_backend = config.get("drift_backend", "drift_loss")
        if drift_backend not in ("drift_loss", "log_kde"):
            raise ValueError("drift_backend must be 'drift_loss' or 'log_kde'")
        if config.get("noise_scale", 0.0) < 0.0:
            raise ValueError("noise_scale must be non-negative")
        if drift_backend == "log_kde":
            if config["log_kde_bandwidth"] <= 0.0:
                raise ValueError("log_kde_bandwidth must be positive")
            if config["gen_per_label"] < 2:
                raise ValueError(
                    "gen_per_label must be at least 2 for leave-one-out log-KDE"
                )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_drift"] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config.get("num_qs", 2),
            encoder=encoders.get("critic"),
        )
        actor_drift_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_drift"),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_drift=(actor_drift_def, (ex_observations, full_actions)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = make_optimizer(config["optimizer"], config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="dfp",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            optimizer="adam",
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg="mean",
            num_qs=2,
            alpha=1.,
            alpha_pos=ml_collections.config_dict.placeholder(float),    # online: weight for top-a pos drift loss (default: alpha)
            alpha_target=ml_collections.config_dict.placeholder(float), # online: weight for dataset target drift loss (default: alpha)
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            actor_type="best-of-n",
            actor_num_samples=16,
            drift_backend="drift_loss", # or "log_kde"
            drift_temps=(0.1,),
            log_kde_bandwidth=0.4,
            gen_per_label=8,
            noise_scale=0.2,
        )
    )
    return config
