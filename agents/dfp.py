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
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from utils.drift_loss import drift_loss


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
        valid = batch["valid"][..., -1] if "valid" in batch else jnp.ones_like(target_q)
        critic_loss = (jnp.square(q - target_q) * valid).mean()

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

        _, drift_rng = jax.random.split(rng)

        # Generate multiple samples per observation.
        gen_per_label = self.config.get("gen_per_label", 8)
        obs_repeated = jnp.repeat(batch["observations"], gen_per_label, axis=0)
        drift_noises = jax.random.normal(drift_rng, (batch_size * gen_per_label, action_dim))

        # Get actions from drift model
        drift_actions_all = self.network.select("actor_drift")(
            obs_repeated, drift_noises, params=grad_params
        )
        drift_actions_all = jnp.clip(drift_actions_all, -1, 1)
        # Reshape to [B, gen_per_label, action_dim]
        gen_samples = drift_actions_all.reshape(batch_size, gen_per_label, action_dim)

        # Positive samples: dataset actions [B, 1, action_dim]
        pos_samples = jnp.expand_dims(batch_actions, axis=1)

        # Compute drift loss (per sample in batch)
        # Note: drift_loss already normalizes internally by scale_inputs
        drift_loss_val, drift_info = drift_loss(
            gen=gen_samples,
            fixed_pos=pos_samples,
            R_list=tuple(self.config.get("drift_temps", [0.1])),
        )

        # Apply alpha to match Q loss magnitude.
        alpha = self.config.get("alpha", 1.0)
        actor_drift_loss = alpha * drift_loss_val.mean()

        # Total loss (no distillation)
        actor_loss = actor_drift_loss #+ q_loss

        info = dict(
            actor_loss=actor_loss,
            actor_drift_loss=actor_drift_loss,
            drift_scale=drift_info.get("scale", 0.0),
        )
        # Add per-temperature losses
        for key, val in drift_info.items():
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

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    def actor_ema_update(self, network):
        """Polyak update for the actor EMA (online-stage only)."""
        tau = self.config.get("actor_ema_tau")
        new_ema_params = jax.tree_util.tree_map(
            lambda p, ep: p * tau + ep * (1 - tau),
            self.network.params["modules_actor_drift"],
            self.network.params["modules_actor_drift_ema"],
        )
        network.params["modules_actor_drift_ema"] = new_ema_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        if agent.config.get("use_actor_ema", False):  
            agent.actor_ema_update(new_network)      
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

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
            actions = jnp.clip(actions, -1, 1)

            q = self._score_actions(obs_rep, actions)
            return self._select_best_bon_action(actions, q)

        noises = self.sample_noises(observations, rng)
        actions = self.network.select("actor_drift")(observations, noises)
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
            actor_drift_ema=(copy.deepcopy(actor_drift_def), (ex_observations, full_actions)),  # ← 추가
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        # EMA mirrors actor_drift; only updated after switch_config_to_online.
        params["modules_actor_drift_ema"] = params["modules_actor_drift"]  

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="dfp",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
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
            drift_temps=(0.1,),
            gen_per_label=8,
            # Actor EMA (for N-sample pool selection in online stage)
            use_actor_ema=False,            # Set True automatically by switch_config_to_online
            actor_ema_tau=ml_collections.config_dict.placeholder(float),  # If None, falls back to `tau`
        )
    )
    return config