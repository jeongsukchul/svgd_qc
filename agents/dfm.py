"""Drift Flow Matching policy for best-of-N offline RL."""

import copy
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.dfp import DFPAgent
from utils.dfm_drift import grouped_sinkhorn_drift
from utils.drift_loss import drift_loss
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.log_kde_loss import grouped_log_kde_loss
from utils.networks import ActorVectorField, Value


def sample_time_pairs(
    rng,
    num_pairs,
    num_flow_steps,
    time_grid_ratio,
    dtype=jnp.float32,
):
    """Sample forward time pairs from a uniform/grid mixture.

    Uniform pairs are formed by sorting two independent U[0, 1] samples. Grid
    pairs are adjacent intervals from the same uniform grid used at inference.
    ``time_grid_ratio`` is the exact fraction of pairs drawn from that grid,
    up to rounding to an integer number of pairs.
    """
    uniform_rng, grid_rng, mixture_rng = jax.random.split(rng, 3)

    uniform_times = jax.random.uniform(
        uniform_rng, (num_pairs, 2), dtype=dtype
    )
    uniform_t = jnp.min(uniform_times, axis=-1)
    uniform_r = jnp.max(uniform_times, axis=-1)

    grid_indices = jax.random.randint(
        grid_rng, (num_pairs,), 0, num_flow_steps
    )
    grid_t = grid_indices.astype(dtype) / num_flow_steps
    grid_r = (grid_indices.astype(dtype) + 1.0) / num_flow_steps

    num_grid_pairs = int(round(time_grid_ratio * num_pairs))
    shuffled_indices = jax.random.permutation(
        mixture_rng, jnp.arange(num_pairs)
    )
    use_grid = shuffled_indices < num_grid_pairs
    t = jnp.where(use_grid, grid_t, uniform_t)
    r = jnp.where(use_grid, grid_r, uniform_r)
    return t[:, None], r[:, None]


def dfm_sinkhorn_loss(
    predicted_state,
    target_state,
    temp_pos,
    temp_neg,
    sinkhorn_iters,
):
    """Return the per-group DFM loss using grouped Sinkhorn drift."""
    predicted_for_drift = jax.lax.stop_gradient(predicted_state)
    target_for_drift = jax.lax.stop_gradient(target_state)
    drift, _, _ = grouped_sinkhorn_drift(
        x=predicted_for_drift,
        pos=target_for_drift,
        neg=predicted_for_drift,
        temp_pos=temp_pos,
        temp_neg=temp_neg,
        sinkhorn_iters=sinkhorn_iters,
    )
    regression_target = jax.lax.stop_gradient(predicted_for_drift + drift)
    per_group_loss = jnp.mean(
        jnp.square(predicted_state - regression_target), axis=(-2, -1)
    )
    info = {
        "per_group_drift_rms": jnp.sqrt(
            jnp.mean(jnp.square(drift), axis=(-2, -1))
        ),
        "drift_scale": jnp.asarray(1.0, dtype=predicted_state.dtype),
    }
    return per_group_loss, info


def dfm_drift_loss(
    predicted_state,
    target_state,
    group_valid,
    drift_temps,
):
    """Return the per-group DFM loss using ``utils.drift_loss.py``."""
    num_particles = predicted_state.shape[-2]
    particle_weights = jnp.broadcast_to(
        (group_valid + 1e-8)[:, None],
        (predicted_state.shape[0], num_particles),
    )
    per_group_loss, legacy_info = drift_loss(
        gen=predicted_state,
        fixed_pos=target_state,
        weight_gen=particle_weights,
        weight_pos=particle_weights,
        R_list=tuple(drift_temps),
    )
    info = {
        "per_group_drift_rms": jnp.full_like(
            group_valid,
            jnp.sqrt(jnp.maximum(legacy_info["drift_norm"], 0.0)),
        ),
        "drift_scale": legacy_info["scale"],
    }
    return per_group_loss, info


def dfm_log_kde_loss(predicted_state, target_state, bandwidth):
    """Return the paper's Gaussian log-KDE scalar loss per group."""
    return grouped_log_kde_loss(predicted_state, target_state, bandwidth)


class DFMAgent(DFPAgent):
    """Time-conditioned marginal transport with Q-ranked best-of-N selection."""

    def _expand_observations_for_particles(self, observations, num_particles):
        """Insert a particle axis before the observation dimensions."""
        obs_ndim = len(self.config["ob_dims"])
        batch_shape = observations.shape[:-obs_ndim] if obs_ndim > 0 else observations.shape
        obs_shape = observations.shape[-obs_ndim:] if obs_ndim > 0 else ()
        observations = jnp.expand_dims(observations, axis=len(batch_shape))
        return jnp.broadcast_to(
            observations, batch_shape + (num_particles,) + obs_shape
        )

    def _transport(self, observations, state, start_time, end_time, params=None):
        """Apply T_{t,r}(x) = x + (r - t) u(x, t, r, s)."""
        target_shape = state.shape[:-1] + (1,)
        start_time = jnp.broadcast_to(
            jnp.asarray(start_time, dtype=state.dtype), target_shape
        )
        end_time = jnp.broadcast_to(
            jnp.asarray(end_time, dtype=state.dtype), target_shape
        )
        # The paper's best-performing embedding is (t, t-r), which exposes
        # both absolute location and interval length to the velocity model.
        times = jnp.concatenate((start_time, start_time - end_time), axis=-1)
        velocity = self.network.select("actor_dfm")(
            observations,
            state,
            times,
            params=params,
        )
        increment = (end_time - start_time) * velocity
        return state + increment, increment

    def _one_step_transport(self, observations, source, params=None):
        """Apply the endpoint map; retained as the one-step convenience path."""
        return self._transport(
            observations, source, 0.0, 1.0, params=params
        )

    def actor_loss(self, batch, grad_params, rng):
        """Compute conditional DFM behavior-cloning loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
            # An incomplete chunk contains actions after an episode boundary.
            # Exclude the whole group, matching the critic's chunk-valid rule.
            group_valid = batch["valid"][..., -1].astype(jnp.float32)
        else:
            batch_actions = batch["actions"][..., 0, :]
            group_valid = jnp.ones((batch_actions.shape[0],), dtype=jnp.float32)

        batch_size, action_dim = batch_actions.shape
        num_particles = self.config["gen_per_label"]
        source_rng, time_rng = jax.random.split(rng)
        # For a high-dimensional condition such as an observation, the paper
        # constructs multiple marginal samples from one dataset action by
        # pairing it with independently drawn source samples.
        source = jax.random.normal(
            source_rng, (batch_size, num_particles, action_dim)
        )
        target_endpoint = jnp.broadcast_to(
            batch_actions[:, None, :], (batch_size, num_particles, action_dim)
        )
        start_time, end_time = sample_time_pairs(
            rng=time_rng,
            num_pairs=batch_size,
            num_flow_steps=self.config["num_flow_steps"],
            time_grid_ratio=self.config["time_grid_ratio"],
            dtype=batch_actions.dtype,
        )
        grouped_start_time = start_time[:, None, :]
        grouped_end_time = end_time[:, None, :]
        source_state = (
            (1.0 - grouped_start_time) * source
            + grouped_start_time * target_endpoint
        )
        target_state = (
            (1.0 - grouped_end_time) * source
            + grouped_end_time * target_endpoint
        )
        observations = self._expand_observations_for_particles(
            batch["observations"], num_particles
        )
        predicted_state, increment = self._transport(
            observations,
            source_state,
            grouped_start_time,
            grouped_end_time,
            params=grad_params,
        )

        drift_backend = self.config["drift_backend"]
        if drift_backend == "sinkhorn":
            per_group_loss, drift_info = dfm_sinkhorn_loss(
                predicted_state=predicted_state,
                target_state=target_state,
                temp_pos=self.config["temp_pos"],
                temp_neg=self.config["temp_neg"],
                sinkhorn_iters=self.config["sinkhorn_iters"],
            )
        elif drift_backend == "log_kde":
            per_group_loss, drift_info = dfm_log_kde_loss(
                predicted_state=predicted_state,
                target_state=target_state,
                bandwidth=self.config["log_kde_bandwidth"],
            )
        else:
            per_group_loss, drift_info = dfm_drift_loss(
                predicted_state=predicted_state,
                target_state=target_state,
                group_valid=group_valid,
                drift_temps=self.config["drift_temps"],
            )

        valid_count = jnp.maximum(group_valid.sum(), 1.0)
        dfm_loss = jnp.sum(per_group_loss * group_valid) / valid_count
        actor_loss = self.config["alpha"] * dfm_loss

        info = {
            "actor_loss": actor_loss,
            "dfm_loss": dfm_loss,
            "residual_rms": jnp.sqrt(jnp.mean(jnp.square(increment))),
            "mean_start_time": start_time.mean(),
            "mean_end_time": end_time.mean(),
            "mean_time_delta": (end_time - start_time).mean(),
            "valid_group_fraction": group_valid.mean(),
        }
        if drift_backend == "log_kde":
            info["log_kde_log_p"] = jnp.sum(
                drift_info["per_group_log_p"] * group_valid
            ) / valid_count
            info["log_kde_log_q"] = jnp.sum(
                drift_info["per_group_log_q"] * group_valid
            ) / valid_count
        else:
            info["drift_rms"] = jnp.sum(
                drift_info["per_group_drift_rms"] * group_valid
            ) / valid_count
            info["drift_scale"] = drift_info["drift_scale"]
        return actor_loss, info

    @partial(jax.jit, static_argnames=("use_q_bon",))
    def sample_actions(self, observations, rng=None, use_q_bon=False):
        """Generate DFM candidates and return the critic-ranked best action."""
        del use_q_bon  # This agent intentionally supports best-of-N only.
        if self.config["actor_type"] != "best-of-n":
            raise ValueError("DFMAgent only supports actor_type='best-of-n'")
        if rng is None:
            rng = jax.random.PRNGKey(0)

        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"]
            if self.config["action_chunking"]
            else 1
        )
        num_candidates = self.config["actor_num_samples"]
        source = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                num_candidates,
                full_action_dim,
            ),
        )
        candidate_observations = self._expand_observations_for_particles(
            observations, num_candidates
        )

        def transport_step(step, state):
            start_time = step.astype(state.dtype) / self.config["num_flow_steps"]
            end_time = (step.astype(state.dtype) + 1.0) / self.config[
                "num_flow_steps"
            ]
            next_state, _ = self._transport(
                candidate_observations,
                state,
                start_time,
                end_time,
            )
            return next_state

        actions = jax.lax.fori_loop(
            0, self.config["num_flow_steps"], transport_step, source
        )

        # Clipping intermediate states would alter the learned probability
        # path; only project completed action candidates to the environment.
        actions = jnp.clip(actions, -1.0, 1.0)
        q_values = self._score_actions(candidate_observations, actions)
        return self._select_best_bon_action(actions, q_values)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a DFM agent."""
        if config["actor_type"] != "best-of-n":
            raise ValueError("DFMAgent only supports actor_type='best-of-n'")
        if config["num_flow_steps"] < 1:
            raise ValueError("num_flow_steps must be at least 1")
        if not 0.0 <= config["time_grid_ratio"] <= 1.0:
            raise ValueError("time_grid_ratio must be between 0 and 1")
        if config["actor_num_samples"] < 1:
            raise ValueError("actor_num_samples must be at least 1")
        if config["gen_per_label"] < 2:
            raise ValueError("gen_per_label must be at least 2 for a drift field")
        if config["drift_backend"] not in (
            "drift_loss",
            "log_kde",
            "sinkhorn",
        ):
            raise ValueError(
                "drift_backend must be 'drift_loss', 'log_kde', or 'sinkhorn'"
            )
        if config["drift_backend"] == "sinkhorn":
            if config["temp_pos"] <= 0.0:
                raise ValueError("temp_pos must be positive")
            if config["temp_neg"] <= 0.0:
                raise ValueError("temp_neg must be positive")
            if config["sinkhorn_iters"] < 1 or config["sinkhorn_iters"] % 2 == 0:
                raise ValueError("sinkhorn_iters must be a positive odd integer")
        elif config["drift_backend"] == "log_kde":
            if config["log_kde_bandwidth"] <= 0.0:
                raise ValueError("log_kde_bandwidth must be positive")
        elif not config["drift_temps"] or any(
            temperature <= 0.0 for temperature in config["drift_temps"]
        ):
            raise ValueError("drift_temps must contain positive temperatures")

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

        encoders = {}
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_dfm"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config.get("num_qs", 2),
            encoder=encoders.get("critic"),
        )
        actor_dfm_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_actions.shape[-1],
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_dfm"),
        )
        ex_times = jnp.zeros((2,), dtype=full_actions.dtype)

        network_info = {
            "critic": (critic_def, (ex_observations, full_actions)),
            "target_critic": (
                copy.deepcopy(critic_def),
                (ex_observations, full_actions),
            ),
            "actor_dfm": (
                actor_dfm_def,
                (ex_observations, full_actions, ex_times),
            ),
        }
        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params["modules_target_critic"] = network.params["modules_critic"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="dfm",
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
            alpha=1.0,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            actor_type="best-of-n",
            actor_num_samples=16,
            num_flow_steps=10,
            time_grid_ratio=.5,
            drift_backend="log_kde",
            temp_pos=1.,
            temp_neg=1.,
            sinkhorn_iters=9,
            drift_temps=(0.1,),
            log_kde_bandwidth=0.1,
            gen_per_label=8,
        )
    )
