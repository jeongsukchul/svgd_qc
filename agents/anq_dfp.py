"""Value-free ANQ with a drift behavior policy and learned refiner.

The drift decoder is trained only with its behavior-cloning objective.  A
separate refinement actor predicts a bounded action delta around each decoded
behavior action.  The refiner is optimized through the Q ensemble, and the
critic is trained with an expectile Bellman loss.  There is no learned V
function and no final weighted-regression actor.
"""

import copy
from functools import partial

import flax
import jax
import jax.numpy as jnp

from agents.dfp import DFPAgent, get_config as get_dfp_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import Actor, ActorVectorField, Value
from utils.optimizers import make_optimizer


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


def _bounded_delta(raw_delta, radius, eps):
    """Map a tanh refiner output into an L2 ball of the given radius."""
    raw_norm = jnp.linalg.norm(raw_delta, axis=-1, keepdims=True)
    unit_ball_scale = jnp.minimum(1.0, 1.0 / jnp.maximum(raw_norm, eps))
    return radius * unit_ball_scale * raw_delta


def refine_actions(agent, observations, base_actions, params=None, target=False):
    """Apply the learned, bounded refinement actor in one forward pass."""
    base_actions = jnp.clip(base_actions, -1.0, 1.0)
    model_name = "target_refine_actor" if target else "refine_actor"
    inputs = jnp.concatenate([observations, base_actions], axis=-1)
    raw_delta = agent.network.select(model_name)(inputs, params=params).mode()
    delta = _bounded_delta(
        raw_delta,
        agent.config["refine_radius"],
        agent.config["refine_eps"],
    )
    refined_actions = jnp.clip(base_actions + delta, -1.0, 1.0)
    # Clipping at an action-space boundary can make the applied delta smaller.
    applied_delta = refined_actions - base_actions
    return refined_actions, applied_delta


def refine_actor_loss(agent, batch, grad_params, rng):
    """Optimize the refiner through Q while keeping it near drift behavior."""
    observations = batch["observations"]
    action_dim = agent.config["action_dim"] * (
        agent.config["horizon_length"]
        if agent.config["action_chunking"]
        else 1
    )
    noises = jax.random.normal(rng, observations.shape[:-1] + (action_dim,))

    # The drift decoder remains behavior-cloned: stop refiner gradients from
    # changing it, while retaining gradients through Q into the refiner.
    base_actions = agent.network.select("actor_drift")(observations, noises)
    base_actions = jax.lax.stop_gradient(jnp.clip(base_actions, -1.0, 1.0))
    refined_actions, delta = refine_actions(
        agent,
        observations,
        base_actions,
        params=grad_params,
    )
    qs = agent.network.select("critic")(
        observations, actions=refined_actions
    )
    q = aggregate_qs(qs, agent.config, mode=agent.config["refine_q_agg"])

    valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]
    denom = jnp.maximum(valid.sum(), 1.0)
    q_objective = (q * valid).sum() / denom
    delta_sq = jnp.sum(jnp.square(delta), axis=-1)
    penalty = (delta_sq * valid).sum() / denom
    q_scale = jax.lax.stop_gradient(
        1.0 / jnp.maximum(jnp.abs(q).mean(), agent.config["refine_q_eps"])
    )
    loss = -q_scale * q_objective + agent.config["refine_lambda"] * penalty

    return loss, {
        "refine_actor_loss": loss,
        "refine_q": q_objective,
        "refine_q_scale": q_scale,
        "refine_penalty": penalty,
        "refine_delta_rms": jnp.sqrt(jnp.mean(jnp.square(delta))),
        "refine_delta_norm": jnp.linalg.norm(delta, axis=-1).mean(),
    }


def validate_refinement_config(config, q_modes=("min", "mean")):
    if not 0.0 < config["critic_expectile"] < 1.0:
        raise ValueError("critic_expectile must be in (0, 1)")
    for key in ("q_agg", "refine_q_agg"):
        if config[key] not in q_modes:
            raise ValueError(f"{key} must be one of {q_modes}")
    if config["refine_radius"] < 0.0:
        raise ValueError("refine_radius must be non-negative")
    if config["refine_lambda"] < 0.0:
        raise ValueError("refine_lambda must be non-negative")
    if config["refine_eps"] <= 0.0:
        raise ValueError("refine_eps must be positive")
    if config["refine_q_eps"] <= 0.0:
        raise ValueError("refine_q_eps must be positive")
    if config["refine_fc_scale"] <= 0.0:
        raise ValueError("refine_fc_scale must be positive")


class ANQDFPAgent(DFPAgent):
    """Learned action refinement over a behavior-cloned DFP decoder."""

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
            target_refiner=True,
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

    def actor_loss(self, batch, grad_params, rng):
        drift_rng, refine_rng = jax.random.split(rng)
        drift_loss, info = super().actor_loss(batch, grad_params, drift_rng)
        refinement_loss, refinement_info = refine_actor_loss(
            self, batch, grad_params, refine_rng
        )
        total_loss = drift_loss + refinement_loss
        info = dict(info)
        info.update(refinement_info)
        info["actor_loss"] = total_loss
        return total_loss, info

    def _sample_refined_actions(
        self,
        observations,
        rng,
        use_q_bon=False,
        critic_name="critic",
        target_refiner=False,
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
                target=target_refiner,
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
            self, observations, base_actions, target=target_refiner
        )

    @partial(
        jax.jit,
        static_argnames=("use_q_bon", "critic_name", "target_refiner"),
    )
    def sample_actions(
        self,
        observations,
        rng=None,
        use_q_bon=False,
        critic_name="critic",
        target_refiner=False,
    ):
        actions, _ = self._sample_refined_actions(
            observations,
            rng,
            use_q_bon=use_q_bon,
            critic_name=critic_name,
            target_refiner=target_refiner,
        )
        return jnp.clip(actions, -1.0, 1.0)

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        for module_name in ("critic", "refine_actor"):
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
        validate_refinement_config(config)
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
        refine_inputs = jnp.concatenate([ex_observations, full_actions], axis=-1)

        encoders = {}
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_drift"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
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
            "actor_drift": (actor_drift_def, (ex_observations, full_actions)),
            "refine_actor": (refine_actor_def, (refine_inputs,)),
            "target_refine_actor": (
                copy.deepcopy(refine_actor_def),
                (refine_inputs,),
            ),
        }
        network_def = ModuleDict(
            {name: definition for name, (definition, _) in network_info.items()}
        )
        network_args = {name: args for name, (_, args) in network_info.items()}
        network_params = network_def.init(init_rng, **network_args)["params"]
        network_params["modules_target_critic"] = network_params["modules_critic"]
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
    config = get_dfp_config()
    config.agent_name = "anq_dfp"
    config.action_chunking = False
    config.num_qs = 4
    config.q_agg = "min"
    config.actor_type = "best-of-n"
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
