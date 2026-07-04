"""
Stochastic Drift Field Policy.

- Uses DFP's drift loss for the action decoder.
- Trains a stochastic policy over the drift-policy input noise space.
- Optimizes the noise policy through critic gradients on decoded actions.
"""

import copy
import math
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.drift_loss import drift_loss
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, LogParam, MLP, Normal, TanhNormal, Value




class STDFPAgent(flax.struct.PyTreeNode):
    """DFP actor-drift decoder trained directly through the critic."""

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
        raise ValueError(f"Unsupported q_agg: {mode}")

    def _safe_clip(self, x, low=-1.0, high=1.0):
        x = jnp.nan_to_num(x, nan=0.0, posinf=high, neginf=low)
        return jnp.clip(x, low, high)

    def _safe_tanh_value(self, x, eps=1e-6):
        x = jnp.nan_to_num(x, nan=0.0, posinf=1.0 - eps, neginf=-1.0 + eps)
        return jnp.clip(x, -1.0 + eps, 1.0 - eps)

    def _safe_atanh_value(self, x, eps=1e-6):
        x = self._safe_tanh_value(x, eps=eps)
        return jnp.arctanh(x)

    def _unit_normal_log_prob(self, x):
        return -0.5 * (jnp.square(x) + math.log(2.0 * math.pi)).sum(axis=-1)

    def _noise_actor_type(self):
        actor_type = self.config["actor_type"] if "actor_type" in self.config else "sac"
        if actor_type in ("sac", "stochastic"):
            return "sac"
        if actor_type in ("ddpg", "deterministic"):
            return "ddpg"
        raise ValueError(f"Unsupported actor_type: {actor_type}")

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
            if "valid" in batch:
                squared_error = squared_error * batch["valid"][:, None, :, None]
        return jnp.mean(squared_error)

    def critic_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)

        rng, sample_rng = jax.random.split(rng)
        next_obs = batch["next_observations"][..., -1, :]
        next_actions = self.sample_actions(next_obs, sample_rng)
        next_actions = self._safe_clip(next_actions)

        next_qs = self.network.select("target_critic")(next_obs, next_actions)
        next_q = self._aggregate_q(next_qs)

        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_q

        q = self.network.select("critic")(
            batch["observations"],
            batch_actions,
            params=grad_params,
        )
        valid = batch["valid"][..., -1] if "valid" in batch else jnp.ones_like(target_q)
        critic_loss = (jnp.square(q - target_q) * valid).mean()

        return critic_loss, {
            "total_loss": critic_loss,
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "tgt_q_mean": next_q.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)
        batch_size, action_dim = batch_actions.shape

        drift_rng, actor_rng = jax.random.split(rng)

        gen_per_label = self.config["gen_per_label"]
        obs_repeated = jnp.repeat(batch["observations"], gen_per_label, axis=0)
        drift_noises = jax.random.normal(
            drift_rng,
            (batch_size * gen_per_label, action_dim),
        )
        drift_actions = self.network.select("actor_drift")(
            obs_repeated,
            drift_noises,
            params=grad_params,
        )
        # drift_actions = self._safe_clip(drift_actions)
        gen_samples = drift_actions.reshape(batch_size, gen_per_label, action_dim)

        drift_loss_val, drift_info = drift_loss(
            gen=gen_samples,
            fixed_pos=batch_actions[:, None, :],
            R_list=tuple(self.config["drift_temps"]),
        )
        actor_drift_loss = drift_loss_val.mean()

        if self._noise_actor_type() == "sac":
            dist = self.network.select("noise_actor")(
                batch["observations"],
                params=grad_params,
            )

            noises = dist.sample(seed=rng)
            log_probs = dist.log_prob(noises)
            decoded_actions = self.network.select("actor_drift")(
                batch["observations"],
                noises * self.config['noise_scale'],
            )
            decoded_actions = self._safe_clip(decoded_actions)
            critic_qs = self.network.select("critic")(
                batch["observations"],
                decoded_actions,
            )
            actor_q = self._aggregate_q(critic_qs, mode=self.config["actor_q_agg"])

            alpha = self.network.select("noise_alpha")()
            alpha_train = self.network.select("noise_alpha")(params=grad_params)
            entropy = -log_probs
            log_prior = self._unit_normal_log_prob(noises)
            kl = log_probs - log_prior

            if self.config["noise_regularizer"] == "entropy":
                reg = log_probs
                policy_loss = (alpha * reg - actor_q).mean()
                alpha_loss = (
                    alpha_train
                    * (
                        jax.lax.stop_gradient(entropy)
                        - self.config["noise_target_entropy"]
                    )
                ).mean()
                target_entropy = self.config["noise_target_entropy"]
                target_kl = jnp.zeros(())
            elif self.config["noise_regularizer"] == "kl":
                reg = kl
                policy_loss = (alpha * reg - actor_q).mean()
                alpha_loss = (
                    alpha_train
                    * (
                        self.config["noise_target_kl"]
                        - jax.lax.stop_gradient(kl)
                    )
                ).mean()
                target_entropy = jnp.zeros(())
                target_kl = self.config["noise_target_kl"]
            else:
                raise ValueError(
                    f"Unsupported noise_regularizer: {self.config['noise_regularizer']}"
                )
            alpha = alpha_train
        else:
            raw_noises = self.network.select("noise_actor")(
                batch["observations"],
                params=grad_params,
            )
            sq_sum = jnp.sum(jnp.square(raw_noises), axis=-1, keepdims=True)
            norm = jnp.sqrt(sq_sum + 1e-6) 
            noises = raw_noises / norm * jnp.sqrt(action_dim)
            log_probs = jnp.zeros((batch_size,))
            decoded_actions = self.network.select("actor_drift")(
                batch["observations"],
                noises,
            )
            decoded_actions = self._safe_clip(decoded_actions)
            critic_qs = self.network.select("critic")(
                batch["observations"],
                decoded_actions,
            )
            actor_q = self._aggregate_q(critic_qs, mode=self.config["actor_q_agg"])

            alpha = jnp.zeros(())
            entropy = jnp.zeros((batch_size,))
            kl = jnp.zeros((batch_size,))
            alpha_loss = jnp.zeros(())
            policy_loss = -actor_q.mean()
            target_entropy = jnp.zeros(())
            target_kl = jnp.zeros(())

        total_loss = actor_drift_loss + policy_loss + alpha_loss

        info = {
            "total_loss": total_loss,
            "actor_drift_loss": actor_drift_loss,
            "policy_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
            "entropy": entropy.mean(),
            "target_entropy": target_entropy,
            "kl": kl.mean(),
            "target_kl": target_kl,
            "q": actor_q.mean(),
            "noise_abs_mean": jnp.abs(noises).mean(),
            "noise_std": noises.std(),
            "noise_max": noises.max(),
            "noise_min": noises.min(),
            "log_prob_mean": log_probs.mean(),
            "log_prob_max": log_probs.max(),
            "log_prob_min": log_probs.min(),
            "log_prob_finite": jnp.isfinite(log_probs).mean(),
            "decoded_action_mean": decoded_actions.mean(),
            "drift_scale": drift_info.get("scale", 0.0),
            "generated_to_data_mse": self._masked_action_mse(
                jnp.square(gen_samples - batch_actions[:, None, :]),
                batch,
            ),
        }
        for key, val in drift_info.items():
            if key.startswith("loss_"):
                info[f"drift_{key}"] = val
        info["attraction_norm"] = drift_info.get("attraction_norm", 0.0)
        info["repulsion_norm"] = drift_info.get("repulsion_norm", 0.0)
        info["diff_from_theory"] = drift_info.get("diff_from_theory", 0.0)
        info["drift_norm"] = drift_info.get("drift_norm", 0.0)

        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
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
        agent.target_update(new_network, "actor_drift")

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_drift_actions(self, observations, noises):
        model_name = (
            "target_actor_drift"
            if self.config["use_target_latent"]
            else "actor_drift"
        )
        actions = self.network.select(model_name)(observations, noises)
        return self._safe_clip(actions)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )

        if self._noise_actor_type() == "ddpg":
            raw_noises = self._safe_clip(self.network.select("noise_actor")(observations))
            sq_sum = jnp.sum(jnp.square(raw_noises), axis=-1, keepdims=True)
            norm = jnp.sqrt(sq_sum + 1e-6) 
            noises = raw_noises / norm * jnp.sqrt(action_dim)
            actions = self.sample_drift_actions(observations, noises)
            return self._safe_clip(actions)

        best_of_n = self.config["best_of_n"]
        observations = jnp.repeat(observations[..., None, :], best_of_n, axis=-2)
        dist = self.network.select("noise_actor")(observations)
        noises = dist.sample(seed=rng)
        actions = self.sample_drift_actions(observations, noises * self.config['noise_scale'])

        q = self._aggregate_q(
            self.network.select("critic")(observations, actions),
            mode=self.config["sample_q_agg"],
        )
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, best_of_n, action_dim))[
            jnp.arange(bsize),
            indices,
            :,
        ].reshape(bshape + (action_dim,))

        return self._safe_clip(actions)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate(
                [ex_actions] * config["horizon_length"],
                axis=-1,
            )
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        actor_type = config.get("actor_type", "sac")
        regularizer = config.get("noise_regularizer", "entropy")
        if actor_type in ("sac", "stochastic"):
            if regularizer == "entropy" and config["noise_target_entropy"] is None:

                config["noise_target_entropy"] = (
                    config["noise_normal_target_entropy_multiplier"] * full_action_dim
                )
            if regularizer == "kl" and config["noise_target_kl"] is None:
                raise ValueError("noise_target_kl must be set when noise_regularizer='kl'")
        elif actor_type in ("ddpg", "deterministic"):
            config["noise_target_entropy"] = 0.0
            config["noise_target_kl"] = 0.0
        else:
            raise ValueError(f"Unsupported actor_type: {actor_type}")

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["noise_actor"] = encoder_module()
            encoders["actor_drift"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
        )
        if actor_type in ("sac", "stochastic"):
            noise_actor_base_cls = partial(
                MLP,
                hidden_dims=config["actor_hidden_dims"],
                activate_final=True,
                layer_norm=config["actor_layer_norm"],
            )
            noise_actor_def = TanhNormal(
                noise_actor_base_cls,
                full_action_dim,
                state_dependent_std=config["noise_state_dependent_std"],
                encoder=encoders.get("noise_actor"),
            )
        else:
            noise_actor_def = MLP(
                hidden_dims=(*tuple(config["actor_hidden_dims"]), full_action_dim),
                layer_norm=config["actor_layer_norm"],
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
            noise_actor=(noise_actor_def, (ex_observations, )),
            actor_drift=(actor_drift_def, (ex_observations, full_actions)),
            target_actor_drift=(
                copy.deepcopy(actor_drift_def),
                (ex_observations, full_actions),
            ),
        )
        if actor_type in ("sac", "stochastic"):
            network_info["noise_alpha"] = (
                LogParam(init_value=config["noise_init_temp"]),
                (),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        params["modules_target_actor_drift"] = params["modules_actor_drift"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="stdfp",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            num_qs=2,
            q_agg="pessimistic",
            actor_q_agg="mean",
            sample_q_agg="mean",
            rho=0.5,
            discount=0.99,
            tau=0.005,
            actor_type="sac",
            best_of_n=1,
            drift_temps=[0.1],
            gen_per_label=8,
            noise_regularizer="kl",
            noise_state_dependent_std=False,
            use_target_latent=True,
            noise_target_entropy=ml_collections.config_dict.placeholder(float),
            noise_target_kl=ml_collections.config_dict.placeholder(float),
            noise_normal_target_entropy_multiplier=0.5,
            noise_init_temp=1.0,
            noise_scale=1.5,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
