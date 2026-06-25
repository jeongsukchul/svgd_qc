import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, LogParam, MLP, TanhNormal, Value


class DSRLAgent(flax.struct.PyTreeNode):
    """DSRL agent with a stochastic policy in the flow noise space."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def _safe_clip(self, x, low=-1.0, high=1.0):
        x = jnp.nan_to_num(x, nan=0.0, posinf=high, neginf=low)
        return jnp.clip(x, low, high)

    def critic_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)

        rng, sample_rng, noise_rng = jax.random.split(rng, 3)
        next_obs = batch["next_observations"][..., -1, :]
        next_actions = self.sample_actions(next_obs, rng=sample_rng)
        next_actions = self._safe_clip(next_actions)

        next_qs = self.network.select("target_critic")(next_obs, next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

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

        noises = jax.random.normal(noise_rng, batch_actions.shape)
        actions = self.sample_flow_actions(batch["observations"], noises=noises)
        actions = self._safe_clip(actions)
        target_qs = self.network.select("critic")(batch["observations"], actions)
        target_qs = jax.lax.stop_gradient(target_qs)
        z_qs = self.network.select("z_critic")(
            batch["observations"],
            noises,
            params=grad_params,
        )
        distill_loss = jnp.mean(jnp.square(z_qs - target_qs))

        total_loss = critic_loss + distill_loss

        return total_loss, {
            "total_loss": total_loss,
            "critic_loss": critic_loss,
            "distill_loss": distill_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "z_q_mean": z_qs.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)
        batch_size, action_dim = batch_actions.shape

        rng, x_rng, t_rng, actor_rng = jax.random.split(rng, 4)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_bc_flow")(
            batch["observations"],
            x_t,
            t,
            params=grad_params,
        )
        valid = batch["valid"][..., -1] if "valid" in batch else jnp.ones((batch_size,))
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * valid)

        dist = self.network.select("actor")(
            batch["observations"],
            params=grad_params,
        )
        raw_noises = dist.sample(seed=actor_rng)
        log_probs = dist.log_prob(raw_noises)
        noises = self._safe_clip(raw_noises) * self.config["noise_scale"]

        qs = self.network.select("z_critic")(batch["observations"], noises)
        q = qs.mean(axis=0)
        alpha = self.network.select("alpha")()
        policy_loss = (alpha * log_probs - q).mean()

        alpha_grad = self.network.select("alpha")(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha_grad * (entropy - self.config["target_entropy"])).mean()

        total_loss = flow_loss + policy_loss + alpha_loss
        action_std = dist.distribution.stddev().mean()

        return total_loss, {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "actor_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha_grad,
            "entropy": -log_probs.mean(),
            "action_std": action_std,
            "q": q.mean(),
        }

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
        agent.target_update(new_network, "actor_bc_flow")

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
        if rng is None:
            rng = jax.random.PRNGKey(0)
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )

        best_of_n = self.config["best_of_n"]
        observations = jnp.repeat(observations[..., None, :], best_of_n, axis=-2)
        dist = self.network.select("actor")(observations)
        noises = dist.sample(seed=rng)
        noises = self._safe_clip(noises) * self.config["noise_scale"]

        actions = self.sample_flow_actions(observations, noises)
        actions = self._safe_clip(actions)

        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, best_of_n, action_dim))[
            jnp.arange(bsize), indices, :
        ].reshape(bshape + (action_dim,))

        return self._safe_clip(actions)

    @jax.jit
    def sample_flow_actions(self, observations, noises):
        actions = noises
        model_name = (
            "target_actor_bc_flow"
            if self.config["use_target_latent"]
            else "actor_bc_flow"
        )
        model = self.network.select(model_name)

        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = model(observations, actions, t)
            actions = self._safe_clip(actions + vels / self.config["flow_steps"])

        actions = self._safe_clip(actions)
        return actions

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

        ex_times = ex_actions[..., :1]
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

        if config["target_entropy"] is None:
            config["target_entropy"] = (
                -config["target_entropy_multiplier"] * full_action_dim
            )

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["z_critic"] = encoder_module()
            encoders["actor"] = encoder_module()
            encoders["actor_bc_flow"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
        )
        z_critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("z_critic"),
        )
        actor_base_cls = partial(
            MLP,
            hidden_dims=config["actor_hidden_dims"],
            activate_final=True,
            layer_norm=config["actor_layer_norm"],
        )
        actor_def = TanhNormal(
            actor_base_cls,
            full_action_dim,
            encoder=encoders.get("actor"),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_bc_flow"),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        alpha_def = LogParam(init_value=config["init_temp"])

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            z_critic=(z_critic_def, (ex_observations, full_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            target_actor_bc_flow=(
                copy.deepcopy(actor_bc_flow_def),
                (ex_observations, full_actions, ex_times),
            ),
            actor=(actor_def, (ex_observations,)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        params["modules_target_actor_bc_flow"] = params["modules_actor_bc_flow"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="dsrl",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),
            value_layer_norm=True,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            num_qs=10,
            rho=0.5,
            discount=0.99,
            tau=0.005,
            flow_steps=10,
            best_of_n=1,
            noise_scale=1.0,
            target_entropy=ml_collections.config_dict.placeholder(float),
            target_entropy_multiplier=0.5,
            init_temp=1.0,
            use_target_latent=True,
            encoder=ml_collections.config_dict.placeholder(str),
            use_fourier_features=False,
            fourier_feature_dim=64,
        )
    )
    return config
