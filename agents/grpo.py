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
from utils.networks import ActorVectorField, Value


class GRPOAgent(flax.struct.PyTreeNode):
    """Offline RL agent with a flow-matching reference policy and Flow-GRPO updates."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def _aggregate_q(self, qs):
        mode = self.config["q_agg"]
        if mode == "min":
            return qs.min(axis=0)
        if mode == "mean":
            return qs.mean(axis=0)
        if mode == "pessimistic":
            return qs.mean(axis=0) - self.config["rho"] * qs.std(axis=0)
        raise ValueError(f"Unsupported q_agg: {mode}")

    def critic_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)

        next_actions = self.sample_actions(
            batch["next_observations"][..., -1, :],
            rng=rng,
            use_q_bfn=True,
        )
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self.network.select("target_critic")(
            batch["next_observations"][..., -1, :],
            next_actions,
        )
        next_q = self._aggregate_q(next_qs)

        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_q

        q = self.network.select("critic")(
            batch["observations"],
            batch_actions,
            params=grad_params,
        )
        critic_loss = (jnp.square(q - target_q) * batch["valid"][..., -1]).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "target_q_mean": target_q.mean(),
        }

    def _flow_matching_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)
        batch_size, action_dim = batch_actions.shape
        x_rng, t_rng = jax.random.split(rng)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1.0 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_ref_flow")(
            batch["observations"],
            x_t,
            t,
            params=grad_params,
        )

        if self.config["action_chunking"]:
            loss = jnp.mean(
                jnp.reshape(
                    jnp.square(pred - vel),
                    (
                        batch_size,
                        self.config["horizon_length"],
                        self.config["action_dim"],
                    ),
                )
                * batch["valid"][..., None]
            )
        else:
            loss = jnp.mean(jnp.square(pred - vel))

        return loss, {
            "ref_flow_loss": loss,
            "ref_flow_mse": jnp.mean(jnp.square(pred - vel)),
        }

    def _transition_mean(self, observations, states, times, module_name, params=None):
        velocities = self.network.select(module_name)(
            observations,
            states,
            times,
            params=params,
        )
        step_size = 1.0 / self.config["flow_steps"]
        return states + step_size * (2.0 * velocities - states / (times + step_size))

    def _transition_stats(
        self,
        observations,
        states,
        next_states,
        times,
        module_name,
        params=None,
    ):
        step_size = 1.0 / self.config["flow_steps"]
        g_t_sq = 2.0 * (1.0 - times + step_size) / (times + step_size)
        std = self.config["sde_noise_scale"] * jnp.sqrt(step_size * g_t_sq)
        mean = self._transition_mean(
            observations,
            states,
            times,
            module_name=module_name,
            params=params,
        )
        normalized = (next_states - mean) / (std + 1e-8)
        log_prob = -0.5 * (
            jnp.square(normalized)
            + 2.0 * jnp.log(std + 1e-8)
            + jnp.log(2.0 * jnp.pi)
        )
        log_prob = log_prob.mean(axis=-1)

        return log_prob, mean, std

    def _rollout_old_policy(self, observations, rng):
        group_size = self.config["grpo_group_size"]
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        batch_size = observations.shape[0]
        obs_group = jnp.repeat(observations[:, None, ...], group_size, axis=1)
        step_rngs = jax.random.split(rng, self.config["flow_steps"] + 1)
        states = jax.random.normal(step_rngs[0], (batch_size, group_size, action_dim))

        xs = []
        next_xs = []
        ts = []
        old_log_probs = []

        for i in range(self.config["flow_steps"]):
            times = jnp.full((*states.shape[:-1], 1), i / self.config["flow_steps"])
            step_noise = jax.random.normal(step_rngs[i + 1], states.shape)
            _, old_mean, old_std = self._transition_stats(
                obs_group,
                states,
                states,
                times,
                module_name="old_actor_tilted_flow",
            )
            next_states = old_mean + old_std * step_noise
            old_log_prob, _, _ = self._transition_stats(
                obs_group,
                states,
                next_states,
                times,
                module_name="old_actor_tilted_flow",
            )

            xs.append(states)
            next_xs.append(next_states)
            ts.append(times)
            old_log_probs.append(old_log_prob)
            states = next_states

        return (
            obs_group,
            jnp.stack(xs, axis=0),
            jnp.stack(next_xs, axis=0),
            jnp.stack(ts, axis=0),
            jnp.stack(old_log_probs, axis=0),
            jnp.clip(states, -1, 1),
        )

    def _group_advantages(self, observations, final_actions):
        batch_size, group_size, action_dim = final_actions.shape
        obs_group = jnp.repeat(observations[:, None, ...], group_size, axis=1)
        qs = self.network.select(
            "target_critic" if self.config["use_target_adv_critic"] else "critic"
        )(obs_group, final_actions)
        q_values = self._aggregate_q(qs)
        q_values = jax.lax.stop_gradient(q_values)

        mean = q_values.mean(axis=-1, keepdims=True)
        std = q_values.std(axis=-1, keepdims=True)
        advantages = (q_values - mean) / jnp.maximum(std, self.config["adv_norm_eps"])
        advantages = jnp.clip(
            advantages,
            -self.config["adv_clip_max"],
            self.config["adv_clip_max"],
        )

        return advantages, q_values

    def actor_loss(self, batch, grad_params, rng):
        rng, ref_rng, rollout_rng = jax.random.split(rng, 3)
        ref_flow_loss, ref_info = self._flow_matching_loss(batch, grad_params, ref_rng)
        (
            obs_group,
            states,
            next_states,
            times,
            old_log_probs,
            final_actions,
        ) = self._rollout_old_policy(batch["observations"], rollout_rng)

        advantages, q_values = self._group_advantages(batch["observations"], final_actions)
        advantages = jnp.broadcast_to(advantages[None, ...], old_log_probs.shape)

        current_log_probs, current_means, stds = self._transition_stats(
            jnp.repeat(obs_group[None, ...], self.config["flow_steps"], axis=0),
            states,
            next_states,
            times,
            module_name="actor_tilted_flow",
            params=grad_params,
        )
        _, ref_means, _ = self._transition_stats(
            jnp.repeat(obs_group[None, ...], self.config["flow_steps"], axis=0),
            states,
            next_states,
            times,
            module_name="actor_ref_flow",
        )

        ratio = jnp.exp(current_log_probs - old_log_probs)
        clipped_ratio = jnp.clip(
            ratio,
            1.0 - self.config["clip_range"],
            1.0 + self.config["clip_range"],
        )
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * clipped_ratio
        policy_loss = jnp.mean(jnp.maximum(unclipped_loss, clipped_loss))

        kl = jnp.square(current_means - ref_means) / (2.0 * jnp.square(stds) + 1e-8)
        kl_loss = jnp.mean(kl)

        total_actor_loss = ref_flow_loss + policy_loss + self.config["beta"] * kl_loss

        info = {
            "actor_loss": total_actor_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "ratio_mean": ratio.mean(),
            "ratio_std": ratio.std(),
            "approx_kl": 0.5 * jnp.mean(jnp.square(current_log_probs - old_log_probs)),
            "clipfrac": jnp.mean(
                (jnp.abs(ratio - 1.0) > self.config["clip_range"]).astype(jnp.float32)
            ),
            "clipfrac_gt_one": jnp.mean(
                (ratio - 1.0 > self.config["clip_range"]).astype(jnp.float32)
            ),
            "clipfrac_lt_one": jnp.mean(
                (1.0 - ratio > self.config["clip_range"]).astype(jnp.float32)
            ),
            "adv_mean": advantages.mean(),
            "adv_std": advantages.std(),
            "q_group_mean": q_values.mean(),
            "q_group_std": q_values.std(),
            "sampled_action_mean": final_actions.mean(),
        }
        info.update(ref_info)
        return total_actor_loss, info

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

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            network.params[f"modules_{module_name}"],
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
        new_network.params["modules_old_actor_tilted_flow"] = jax.tree_util.tree_map(
            lambda x: x + jnp.zeros_like(x),
            new_network.params["modules_actor_tilted_flow"],
        )

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @partial(jax.jit, static_argnames=("module_name",))
    def compute_flow_actions(self, observations, noises, module_name="actor_tilted_flow"):
        actions = noises
        for i in range(self.config["flow_steps"]):
            times = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            velocities = self.network.select(module_name)(observations, actions, times)
            actions = actions + velocities / self.config["flow_steps"]
        return jnp.clip(actions, -1, 1)

    @partial(jax.jit, static_argnames=("use_q_bfn",))
    def sample_actions(self, observations, rng, use_q_bfn=False):
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        num_samples = self.config["q_bfn"] if use_q_bfn else self.config["best_of_n"]
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                num_samples,
                action_dim,
            ),
        )
        observations = jnp.repeat(observations[..., None, :], num_samples, axis=-2)
        actions = self.compute_flow_actions(
            observations,
            noises,
            module_name="actor_tilted_flow",
        )

        qs = self.network.select("critic")(observations, actions)
        q = self._aggregate_q(qs)
        indices = jnp.argmax(q, axis=-1)

        batch_shape = indices.shape
        flat_indices = indices.reshape(-1)
        flat_actions = jnp.reshape(actions, (-1, num_samples, action_dim))
        chosen = flat_actions[jnp.arange(len(flat_indices)), flat_indices, :]
        return chosen.reshape(batch_shape + (action_dim,))

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
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

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_ref_flow"] = encoder_module()
            encoders["actor_tilted_flow"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
        )
        actor_ref_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_ref_flow"),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        actor_tilted_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_tilted_flow"),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_ref_flow=(actor_ref_def, (ex_observations, full_actions, ex_times)),
            actor_tilted_flow=(actor_tilted_def, (ex_observations, full_actions, ex_times)),
            old_actor_tilted_flow=(copy.deepcopy(actor_tilted_def), (ex_observations, full_actions, ex_times)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        if config["clip_grad"]:
            network_tx = optax.chain(
                optax.clip_by_global_norm(max_norm=1.0),
                optax.adam(learning_rate=config["lr"]),
            )
        else:
            network_tx = optax.adam(learning_rate=config["lr"])

        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        params["modules_actor_tilted_flow"] = jax.tree_util.tree_map(
            lambda x: x + jnp.zeros_like(x),
            params["modules_actor_ref_flow"],
        )
        params["modules_old_actor_tilted_flow"] = jax.tree_util.tree_map(
            lambda x: x + jnp.zeros_like(x),
            params["modules_actor_tilted_flow"],
        )

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="grpo",
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
            encoder=ml_collections.config_dict.placeholder(str),
            use_fourier_features=False,
            fourier_feature_dim=64,
            num_qs=2,
            q_agg="pessimistic",
            rho=0.5,
            discount=0.995,
            tau=0.005,
            flow_steps=10,
            best_of_n=1,
            q_bfn=1,
            grpo_group_size=8,
            clip_range=0.2,
            beta=0.01,
            adv_clip_max=5.0,
            adv_norm_eps=1e-6,
            sde_noise_scale=1.0,
            use_target_adv_critic=True,
            clip_grad=True,
        )
    )
