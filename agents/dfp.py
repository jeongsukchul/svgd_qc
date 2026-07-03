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
from utils.flax_utils import (
    ModuleDict,
    TrainState,
    nonpytree_field,
    restore_partial_modules,
)
from utils.networks import ActorVectorField, Value
from utils.drift_loss import drift_loss


class DFPAgent(flax.struct.PyTreeNode):
    """Drift Policy agent with drift model."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    def _masked_action_mse(self, squared_error, batch):
        if self.config["action_chunking"]:
            squared_error = jnp.reshape(
                squared_error,
                (
                    squared_error.shape[0],
                    squared_error.shape[1],
                    self.config["horizon_length"],
                    self.config["action_dim"],
                ),
            )
            return jnp.mean(squared_error * batch["valid"][:, None, :, None])
        return jnp.mean(squared_error)
    def _get_bon_samples(self, use_q_bon: bool):
        key = "q_bon" if use_q_bon else "eval_bon"
        n = self.config[key]
        if n <= 0:
            n = self.config["actor_num_samples"]
        return n

    def _aggregate_q(self, qs):
        mode = self.config["q_agg"]
        if mode == "min":
            return qs.min(axis=0)
        if mode == "mean":
            return qs.mean(axis=0)
        if mode == "pessimistic":
            return qs.mean(axis=0) - self.config["rho"] * qs.std(axis=0)
        raise ValueError(f"Unsupported q_agg: {mode}")
    # def critic_loss(self, batch, grad_params, rng):
    #     if self.config["action_chunking"]:
    #         batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
    #     else:
    #         batch_actions = batch["actions"][..., 0, :]

    #     next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=rng)
    #     next_actions = jnp.clip(next_actions, -1, 1)
    #     next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
    #     next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

    #     target_q = batch['rewards'][..., -1] + \
    #         (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

    #     q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
    #     critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

    #     total_loss = critic_loss
    #     return total_loss, {
    #         'critic_loss': critic_loss,
    #         'q_mean': q.mean(),
    #         'q_max': q.max(),
    #         'q_min': q.min(),
    #     }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the Drift critic loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)
        next_obs = batch["next_observations"][..., -1, :]
        batch_size = next_obs.shape[0]
        action_dim = batch_actions.shape[-1]
        # flow_noise = jax.random.normal(
        #         sample_rng, (batch_size, action_dim)
        #     )
        # next_actions = self.compute_flow_actions(
        #         next_obs, noises=flow_noise
        #     )
        next_actions = self.sample_actions(next_obs, sample_rng, use_q_bon=True)

        next_qs = self.network.select("target_critic")(next_obs, actions=next_actions)
        next_q = self._aggregate_q(next_qs)

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

        # drift loss part
        # drift_mode = self.config.get("drift_mode", "none")
        # if drift_mode in ("hard", "soft"):
        #     return self._actor_loss_online(batch, batch_actions, grad_params, rng, drift_mode)

        flow_rng, drift_rng, tr_rng = jax.random.split(rng, 3)

        # Generate multiple samples per observation.
        gen_per_label = self.config.get("gen_per_label", 8)
        # bc_pos_samples = self.config.get("bc_pos_samples", 8)
        obs_repeated = jnp.repeat(batch["observations"], gen_per_label, axis=0)
        drift_noises = jax.random.normal(drift_rng, (batch_size * gen_per_label, action_dim))
        # Get actions from drift model
        drift_actions_all = self.network.select("actor_drift")(
            obs_repeated, drift_noises, params=grad_params
        )
        # drift_actions_all = jnp.clip(drift_actions_all, -1, 1)
        # Reshape to [B, gen_per_label, action_dim]
        gen_samples = drift_actions_all.reshape(batch_size, gen_per_label, action_dim)

        # Reuse the same generated actions for Q loss over all generated samples.
        qs_all = self.network.select("critic")(
            obs_repeated, actions=drift_actions_all
        )
        q_all = self._aggregate_q(qs_all).reshape(batch_size, gen_per_label)

        def q_fn(x):
            return self._aggregate_q(self.network.select("critic")(obs_repeated, x)).mean()

        score = jax.grad(q_fn)(drift_actions_all) * self.config.get("q_score_coeff", 0.0)
        score = score.reshape(batch_size, gen_per_label, action_dim)
        q_loss = -q_all.mean()
        if self.config["normalize_q_loss"]:
            lam = jax.lax.stop_gradient(1 / (jnp.abs(q_all).mean()))
            q_loss = lam * q_loss
        q_loss = self.config["q_alpha"] * q_loss

        pos_samples = jnp.expand_dims(batch_actions, axis=1)
        total_pos_samples = pos_samples
        drift_loss_val, drift_info = drift_loss(
            gen=gen_samples,
            fixed_pos=total_pos_samples,
            R_list=tuple(self.config.get("drift_temps", [0.1])),
        )
        # Apply alpha to match Q loss magnitude.
        alpha = self.config.get("alpha", 1.0)
        actor_drift_loss = alpha * drift_loss_val.mean()

        # Trust Region Drift
        # noise_old = jax.random.normal(tr_rng, (batch_size * gen_per_label, action_dim))
        # sel_actor_name = "actor_drift_ema" if self.config.get("use_actor_ema", False) else "actor_drift"
        # sel_acts = self.network.select(sel_actor_name)(obs_repeated, noise_old)
        # sel_acts = jnp.clip(sel_acts, -1, 1)
        # sel_pool = jax.lax.stop_gradient(
        #     sel_acts.reshape(batch_size, gen_per_label, action_dim)
        # ) 
        # tr_drift_loss_val, tr_drift_info = drift_loss(
        #     gen=gen_samples,
        #     fixed_pos=sel_pool,
        #     score = score,
        #     R_list=tuple(self.config.get("drift_temps", [0.1])),
        # )
        # tr_drift_loss = self.config.get("alpha_target") * tr_drift_loss_val.mean()
        # Total loss (no distillation)
        actor_loss = actor_drift_loss + q_loss #+ tr_drift_loss

        info = dict(
            actor_loss=actor_loss,
            actor_drift_loss=actor_drift_loss,
            drift_scale=drift_info.get("scale", 0.0),
            q_loss=q_loss,
            generated_to_data_mse=self._masked_action_mse(
                jnp.square(gen_samples - batch_actions[:, None, :]),
                batch,
            ),
        )
        # Add per-temperature losses
        for key, val in drift_info.items():
            if key.startswith("loss_"):
                info[f"drift_{key}"] = val
        info["attraction_norm"] = drift_info.get("attraction_norm", 0.0)
        info["repulsion_norm"] = drift_info.get("repulsion_norm", 0.0)
        info["diff_from_theory"] = drift_info.get("diff_from_theory", 0.0)
        info["drift_norm"] = drift_info.get("drift_norm", 0.0)

        # info["tr_attraction_norm"] = tr_drift_info.get("attraction_norm", 0.0)
        # info["tr_repulsion_norm"] = tr_drift_info.get("repulsion_norm", 0.0)
        # info["tr_drift_norm"] = tr_drift_info.get("drift_norm", 0.0)
        # info["tr_score_norm"] = tr_drift_info.get("score_norm", 0.0)
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
        return self._aggregate_q(qs)

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
            num_samples = self._get_bon_samples(use_q_bon=use_q_bon)
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

        ex_times = ex_actions[..., :1]
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
            encoders["actor_bc_flow"] = encoder_module()

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
            # actor_drift_ema=(copy.deepcopy(actor_drift_def), (ex_observations, full_actions)),  # ← 추가
        )
        if encoders.get("actor_bc_flow") is not None:
            network_info["actor_bc_flow_encoder"] = (encoders.get("actor_bc_flow"), (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        # EMA mirrors actor_drift; only updated after switch_config_to_online.
        # params["modules_actor_drift_ema"] = params["modules_actor_drift"]  

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        if config.get("actor_type") is None:
            config["actor_type"] = "distill-ddpg"
        if config.get("actor_num_samples") is None:
            config["actor_num_samples"] = 32
        agent = cls(rng, network=network, config=flax.core.FrozenDict(**config))

        return agent


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
            q_agg="pessimistic",
            rho=0.5,
            num_qs=2,
            q_score_coeff=1.0,  
            alpha=1.0,
            alpha_target=0.0,   # online: weight for dataset target drift loss (default: alpha)
            normalize_q_loss=False,
            q_alpha=0.0,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            actor_type="best-of-n",
            actor_num_samples=1,
            q_bon=16,
            eval_bon=16,
            drift_temps=[0.1],
            # bc_pos_samples=8,          # Number of BC flow samples to use as positives in drift loss (rest of N are from drift model)
            gen_per_label=8, # Number of generated samples per data point for drift loss
            flow_steps=10,
            use_fourier_features=False,
            fourier_feature_dim=64,
            # Actor EMA (for N-sample pool selection in online stage)
            # use_actor_ema=True,            # Set True automatically by switch_config_to_online
            actor_ema_tau=0.0001,  # If None, falls back to `tau`
        )
    )
    return config
