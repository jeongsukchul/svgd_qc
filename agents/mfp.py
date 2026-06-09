import copy

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.acfql import ACFQLAgent
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import ActorVectorField, Value


class MFPAgent(ACFQLAgent):
    """Mean Flow Policy (MFP) agent with instantaneous velocity constraint."""

    def sample_t_r(self, batch_size, rng):
        """Sample MeanFlow time pairs with t >= r and optional r=t boundary samples."""
        sample_rng, perm_rng = jax.random.split(rng)
        if self.config["time_dist"][0] == "uniform":
            samples = jax.random.uniform(sample_rng, (batch_size, 2), dtype=jnp.float32)
        elif self.config["time_dist"][0] == "lognorm":
            mu = self.config["time_dist"][-2]
            sigma = self.config["time_dist"][-1]
            normal_samples = sigma * jax.random.normal(sample_rng, (batch_size, 2), dtype=jnp.float32) + mu
            samples = jax.nn.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unsupported time_dist: {self.config['time_dist']}")

        t = jnp.max(samples, axis=-1, keepdims=True)
        r = jnp.min(samples, axis=-1, keepdims=True)

        num_selected = int(self.config["flow_ratio"] * batch_size)
        if num_selected > 0:
            indices = jax.random.permutation(perm_rng, batch_size)[:num_selected]
            r = r.at[indices].set(t[indices])

        return t, r

    def _mean_flow_actions(self, observations, noises):
        """Generate one-step actions with the mean-flow policy."""
        r = jnp.zeros((*noises.shape[:-1], 1))
        t = jnp.ones((*noises.shape[:-1], 1))
        times = jnp.concatenate([r, t], axis=-1)

        velocities = self.network.select("actor_mean_flow")(
            observations,
            noises,
            times,
        )
        actions = noises + velocities
        actions = jnp.clip(actions, -1, 1)
        return actions

    def _expand_observations_for_candidates(self, observations, num_candidates):
        """Insert the best-of-N candidate axis before the observation dimensions."""
        obs_ndim = len(self.config["ob_dims"])
        batch_shape = observations.shape[:-obs_ndim] if obs_ndim > 0 else observations.shape
        obs_shape = observations.shape[-obs_ndim:] if obs_ndim > 0 else ()
        observations = jnp.expand_dims(observations, axis=len(batch_shape))
        return jnp.broadcast_to(observations, batch_shape + (num_candidates,) + obs_shape)

    def _sample_actions(self, observations, rng):
        """Sample from the unified MFP policy."""
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )

        if self.config["actor_type"] == "best-of-n":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config["ob_dims"])],
                    self.config["actor_num_samples"],
                    action_dim,
                ),
            )
            observations = self._expand_observations_for_candidates(
                observations,
                self.config["actor_num_samples"],
            )
            actions = self._mean_flow_actions(observations, noises)

            if self.config["q_agg"] == "mean":
                q = self.network.select("critic")(observations, actions).mean(axis=0)
            else:
                q = self.network.select("critic")(observations, actions).min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[
                jnp.arange(bsize), indices, :
            ].reshape(bshape + (action_dim,))
        else:
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config["ob_dims"])],
                    action_dim,
                ),
            )
            actions = self._mean_flow_actions(observations, noises)

        return actions

    def _masked_action_mse(self, squared_error, batch):
        """Average action errors while respecting invalid action-chunk tails."""
        if self.config["action_chunking"]:
            squared_error = jnp.reshape(
                squared_error,
                (squared_error.shape[0], self.config["horizon_length"], self.config["action_dim"]),
            )
            return jnp.mean(squared_error * batch["valid"][..., None])
        return jnp.mean(squared_error)

    def actor_loss(self, batch, grad_params, rng):
        """Compute the MFP policy loss from the paper's mean-flow objective."""
        rng, target_rng, x_rng, t_rng = jax.random.split(rng, 4)

        # Calls without grad_params use stored pre-update parameters, i.e. pi_old for this update.
        target_actions = self.sample_actions(batch["observations"], rng=target_rng)
        target_actions = jax.lax.stop_gradient(target_actions)

        batch_size, action_dim = target_actions.shape
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = target_actions
        vel = x_1 - x_0

        t, r = self.sample_t_r(batch_size, t_rng)
        x_t = (1.0 - t) * x_0 + t * x_1

        def mean_velocity(actions, end_times):
            # MeanFlow identity for u_theta(z_t, r, t); r is held fixed, while
            # the JVP below differentiates through z_t and the end time t.
            times = jnp.concatenate([r, end_times], axis=-1)
            return self.network.select("actor_mean_flow")(
                batch["observations"],
                actions,
                times,
                params=grad_params,
            )

        pred, total_derivative = jax.jvp(
            mean_velocity,
            (x_t, t),
            (vel, jnp.ones_like(t)),
        )
        total_derivative_clip = self.config["total_derivative_clip"]
        total_derivative = jnp.nan_to_num(
            total_derivative,
            nan=0.0,
            posinf=total_derivative_clip,
            neginf=-total_derivative_clip,
        )
        total_derivative = jnp.clip(total_derivative, -total_derivative_clip, total_derivative_clip)
        mf_target = jax.lax.stop_gradient(vel - (t - r) * total_derivative)
        mf_loss = self._masked_action_mse(jnp.square(pred - mf_target), batch)

        ivc_times = jnp.concatenate([t, t], axis=-1)
        ivc_pred = self.network.select("actor_mean_flow")(
            batch["observations"],
            x_t,
            ivc_times,
            params=grad_params,
        )
        ivc_loss = self._masked_action_mse(jnp.square(ivc_pred - vel), batch)

        actor_loss = mf_loss + self.config["ivc_coeff"] * ivc_loss

        return actor_loss, {
            "actor_loss": actor_loss,
            "mf_loss": mf_loss,
            "ivc_loss": ivc_loss,
            "target_action_mean": target_actions.mean(),
            "target_action_max": target_actions.max(),
            "target_action_min": target_actions.min(),
            "total_derivative_abs_max": jnp.max(jnp.abs(total_derivative)),
        }

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        return self._sample_actions(observations, rng)

    @jax.jit
    def compute_mean_flow_actions(
        self,
        observations,
        noises,
    ):
        """Generate one-step MFP actions: a(1) = a(0) + u_theta(a(0), 0, 1, s)."""
        return self._mean_flow_actions(observations, noises)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new MFP agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = jnp.concatenate([ex_actions[..., :1], ex_actions[..., :1]], axis=-1)
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_mean_flow"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
        )

        actor_mean_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_actions.shape[-1],
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_mean_flow"),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        network_info = dict(
            actor_mean_flow=(actor_mean_flow_def, (ex_observations, full_actions, ex_times)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.0:
            network_tx = optax.adamw(learning_rate=config["lr"], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config["lr"])
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
            agent_name="mfp",
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
            ivc_coeff=1.0,
            total_derivative_clip=100.0,
            time_dist=("uniform",),
            flow_ratio=0.0,
            min_time_delta=0.0,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            actor_type="best-of-n",
            actor_num_samples=32,
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.0,
        )
    )
    return config
