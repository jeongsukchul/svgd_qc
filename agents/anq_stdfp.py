"""Value-free ANQ with latent-noise and action-refinement actors.

ANQ-STDFP retains STDFP's latent-noise policy and behavior-cloned drift
decoder.  A learned refinement actor then predicts a bounded action delta.
The critic uses expectile Bellman regression; no V function or final policy is
learned.
"""

import copy
from functools import partial

import flax
import jax
import jax.numpy as jnp

from agents.anq_dfp import (
    aggregate_qs,
    refine_actions,
    refine_actor_loss,
    select_best,
    td_expectile_loss,
    validate_refinement_config,
)
from agents.stdfp import STDFPAgent, get_config as get_stdfp_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import (
    Actor,
    ActorVectorField,
    LogParam,
    MLP,
    TanhNormal,
    Value,
)
from utils.optimizers import make_optimizer


class ANQSTDFPAgent(STDFPAgent):
    """Latent behavior-mode selection followed by learned action refinement."""

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

    def actor_loss(self, batch, grad_params, rng):
        latent_drift_rng, refine_rng = jax.random.split(rng)
        base_loss, info = super().actor_loss(
            batch, grad_params, latent_drift_rng
        )
        refinement_loss, refinement_info = refine_actor_loss(
            self, batch, grad_params, refine_rng
        )
        total_loss = base_loss + refinement_loss
        info = dict(info)
        info.update(refinement_info)
        info["total_loss"] = total_loss
        return total_loss, info

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
        refined_actions, _ = refine_actions(
            self,
            observations,
            base_actions,
            target=use_target_latent,
        )
        return self._safe_clip(refined_actions)

    def _sample_refined_actions(
        self, observations, rng, use_target_critic=False
    ):
        if rng is None:
            rng = jax.random.PRNGKey(0)
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
                target=use_target_critic,
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
            target=use_target_critic,
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

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        for module_name in ("critic", "actor_drift", "refine_actor"):
            source = new_network.params[f"modules_{module_name}"]
            target = agent.network.params[f"modules_target_{module_name}"]
            new_network.params[f"modules_target_{module_name}"] = (
                jax.tree_util.tree_map(
                    lambda p, tp: (
                        agent.config["tau"] * p
                        + (1.0 - agent.config["tau"]) * tp
                    ),
                    source,
                    target,
                )
            )
        return agent.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        q_modes = ("min", "mean", "pessimistic")
        validate_refinement_config(config, q_modes=q_modes)
        for key in ("actor_q_agg", "sample_q_agg"):
            if config[key] not in q_modes:
                raise ValueError(f"{key} must be one of {q_modes}")

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
        noises = full_actions
        refine_inputs = jnp.concatenate([ex_observations, full_actions], axis=-1)

        actor_type = config.get("actor_type", "sac")
        regularizer = config.get("noise_regularizer", "entropy")
        if config.get("noise_scale", 0.0) < 0.0:
            raise ValueError("noise_scale must be non-negative")
        if config.get("latent_noise_scale", 1.0) <= 0.0:
            raise ValueError("latent_noise_scale must be positive")
        if actor_type in ("sac", "stochastic"):
            if regularizer == "entropy" and config["noise_target_entropy"] is None:
                config["noise_target_entropy"] = (
                    config["target_multiplier"] * full_action_dim
                )
            if regularizer == "kl" and config["noise_target_kl"] is None:
                config["noise_target_kl"] = (
                    config["target_multiplier"] * full_action_dim
                )
        elif actor_type in ("ddpg", "deterministic"):
            config["noise_target_entropy"] = 0.0
            config["noise_target_kl"] = 0.0
        else:
            raise ValueError(f"Unsupported actor_type: {actor_type}")

        encoders = {}
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
        refine_actor_def = Actor(
            hidden_dims=config["refine_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["refine_layer_norm"],
            tanh_squash=True,
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=config["refine_fc_scale"],
        )

        network_info = {
            "critic": (critic_def, (ex_observations, full_actions)),
            "target_critic": (
                copy.deepcopy(critic_def),
                (ex_observations, full_actions),
            ),
            "noise_actor": (noise_actor_def, (ex_observations,)),
            "actor_drift": (actor_drift_def, (ex_observations, noises)),
            "target_actor_drift": (
                copy.deepcopy(actor_drift_def),
                (ex_observations, noises),
            ),
            "refine_actor": (refine_actor_def, (refine_inputs,)),
            "target_refine_actor": (
                copy.deepcopy(refine_actor_def),
                (refine_inputs,),
            ),
        }
        if actor_type in ("sac", "stochastic"):
            network_info["noise_alpha"] = (
                LogParam(init_value=config["noise_init_temp"]),
                (),
            )

        network_def = ModuleDict(
            {name: definition for name, (definition, _) in network_info.items()}
        )
        network_args = {name: args for name, (_, args) in network_info.items()}
        network_params = network_def.init(init_rng, **network_args)["params"]
        network_params["modules_target_critic"] = network_params["modules_critic"]
        network_params["modules_target_actor_drift"] = network_params[
            "modules_actor_drift"
        ]
        network_params["modules_target_refine_actor"] = network_params[
            "modules_refine_actor"
        ]
        network_tx = make_optimizer(config["optimizer"], config["lr"])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        config["full_action_dim"] = full_action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


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
    config.refine_hidden_dims = (512, 512)
    config.refine_layer_norm = False
    config.refine_fc_scale = 0.01
    config.refine_q_agg = "min"
    config.refine_radius = 0.2
    config.refine_lambda = 5.0
    config.refine_eps = 1e-6
    config.refine_q_eps = 1e-6
    return config
