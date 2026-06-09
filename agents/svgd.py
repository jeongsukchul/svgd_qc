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
from jax.scipy.special import logsumexp


class SVGDAgent(ACFQLAgent):
    """Drifting Field Policy agent with the behavior-cloning drift loss."""

    def _apply_actor(self, observations, noises, params=None):
        actions = self.network.select("actor")(observations, noises, params=params)

        return actions

    def _apply_old_actor(self, observations, noises):
        actions = self.network.select("old_actor")(observations, noises)

        return actions

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def _sample_raw_actions(self, observations, rng):
        """Sample directly from pi_theta without the best-of-N execution wrapper."""
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                action_dim,
            ),
        )
        actions = self._apply_actor(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

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

    def _action_mask(self, batch):
        if not self.config["action_chunking"]:
            return None
        return jnp.reshape(
            jnp.repeat(batch["valid"][..., None], self.config["action_dim"], axis=-1),
            (batch["valid"].shape[0], -1),
        )

    def _pairwise_distance(self, x, y):
        diff = x[:, :, None, :] - y[:, None, :, :]
        sq_dist = jnp.sum(jnp.square(diff), axis=-1)
        # sq_dist /= sq_dist.mean()
        #[B, N, N+P]
        N = x.shape[-2]
        P = y.shape[-2] -N
        eye = 1e5 * jnp.pad(jnp.eye(N,), ( (0,0), (0, P)))[None, ...]
        sq_dist = sq_dist + eye
        return sq_dist

    def _score_gain(self):
        if self.config["online_learning"]:
            return self.config["online_score_gain"]
        return self.config["score_gain"]

    def _drifting_field(self, query_actions, positive_actions, q_value, score):
        """Compute drift field with the toy mean-drift kernel normalization."""
        bandwidth = self.config["bandwidth"]
        epsilon = self.config["epsilon"]
        num_gen = query_actions.shape[-2]
        targets = jnp.concatenate([query_actions, positive_actions], axis=-2)
        dist = self._pairwise_distance(query_actions, targets)
        score_gain = self._score_gain()

        # logits = -dist / (2 * bandwidth)
        # [B, N, N+P]
        P = positive_actions.shape[-2]
        N = query_actions.shape[-2] 
        target_diff_pos = positive_actions[:, None, :, :] - query_actions[:, :, None, :]
        self_diff = query_actions[:, None, :, :] - query_actions[:, :, None, :]

        dist_pos = jnp.sum(target_diff_pos ** 2, axis=-1) 
        dist_pos /=jnp.median(dist_pos, axis=(1,2), keepdims=True) + 1e-6
        dist_self = jnp.sum(self_diff ** 2, axis=-1)

        log_kernel_pos = jax.nn.log_softmax(
            (score_gain * q_value[..., None] - dist_pos) / epsilon,
            axis=-2
        ) 

        weights_pos = jnp.exp(log_kernel_pos)
        weights_neg = jnp.exp(-dist_self / (2 * bandwidth))

        drift_pos =  (2 / epsilon) * jnp.sum(weights_pos[..., None] * target_diff_pos, axis=-2)
        repulsion_term = (2/bandwidth) * jnp.sum(weights_neg[..., None] * self_diff, axis=-2)

        target_mass = weights_pos.sum(axis=-1, keepdims=True)
        score_term = weights_neg @ ((score_gain / epsilon) * (target_mass * score))

        attraction_term = weights_neg @ drift_pos 
        # log_kernel_pos = jax.nn.log_softmax((score_gain * q_value[..., None] + target_diff.sum(axis=-1)) / bandwidth, axis=-2) # [B, N, P]
        # log_kernel_neg = logits[..., :num_gen]                              # [B, N, N]
        # use sinkhorn iteration instead
        # for _ in range(5):
        #     log_kernel_pos = jax.nn.log_softmax(log_kernel_pos, axis=-2)
        #     log_kernel_pos = jax.nn.log_softmax(log_kernel_pos, axis=-1)
        #     log_kernel_neg = jax.nn.log_softmax(log_kernel_neg, axis=-2)
        #     log_kernel_neg = jax.nn.log_softmax(log_kernel_neg, axis=-1)
        # log_kernel = 0.5 * jnp.maximum(log_kernel_row + log_kernel_col, jnp.log(1e-6))
        # log_kernel = 0.5 * jnp.maximum(log_kernel_row + log_kernel_col, jnp.log(1e-6))
        # log_kernel_pos = log_kernel[..., num_gen:]
        # log_kernel_neg = log_kernel[..., :num_gen]

        #sinkhorn DRO like update
        # weights_pos = jnp.exp(log_kernel_pos)
        # weights_neg = jnp.exp(log_kernel_neg)

        # weights_pos = weights_pos * (weights_neg.sum(axis=-1, keepdims=True))
        # weights_neg = weights_neg * (weights_pos.sum(axis=-1, keepdims=True))

        # drift_score2 = score_gain * weights_neg.sum(axis=-1, keepdims=True) * score
        original_drift = attraction_term - repulsion_term
        drift = original_drift + score_term # - total_coeff * old_scaled_query_actions
        drift_norm = jnp.sqrt(jnp.clip((drift ** 2).mean(), a_min=1e-6))
        return drift/drift_norm, {
            "bandwidth": bandwidth,
            "score_gain": jnp.asarray(score_gain),
            'loss': jnp.sqrt((drift ** 2).mean()),
            'original_scale' : jnp.sqrt((original_drift ** 2).mean()/ drift_norm),
            'score_drift_scale':  jnp.sqrt((score_term ** 2).mean() / drift_norm),
            'attraction_scale':  jnp.sqrt((attraction_term ** 2).mean() / drift_norm),
            'repulsion_scale':  jnp.sqrt((repulsion_term ** 2).mean() / drift_norm),
            "pos_weight_mean": weights_pos.mean(),
            "pos_weight_max": weights_pos.max(),
            "pos_weight_min": weights_pos.min(),
            "neg_weight_mean": weights_neg.mean(),
            "neg_weight_max": weights_neg.max(),
            "neg_weight_min": weights_neg.min(),
            "pos_dist_mean": dist[..., num_gen:].mean(),
            "neg_dist_mean": dist[..., :num_gen].mean(),

        }

    def _drift_loss(self, query_actions, positive_actions, q_value, score, batch):
        drift, drift_info = self._drifting_field(
            query_actions=query_actions,
            positive_actions=positive_actions,
            q_value=q_value,
            score=score,
        )
        drift_targets = jax.lax.stop_gradient(query_actions + drift)
        squared_error = jnp.square(query_actions - drift_targets)
        loss = squared_error.mean() # self._masked_action_mse(squared_error, batch)

        return loss, {
            # "score" : jnp.sqrt((score **2).mean()),
            # "score_gain": drift_info["score_gain"],
            # "query_actions" : jnp.sqrt((query_actions **2).mean()),
            # "drift_target" : jnp.sqrt((drift_targets **2).mean()),
            "drift_norm": drift_info["loss"],
            "original_drift_norm" : drift_info["original_scale"],
            "attraction_norm" : drift_info["attraction_scale"],
            "repulsion_norm" : drift_info["repulsion_scale"],
            "score_norm" : drift_info["score_drift_scale"],
            # "pos_weight_mean": drift_info["pos_weight_mean"],
            # "pos_weight_max": drift_info["pos_weight_max"],
            # "pos_weight_min": drift_info["pos_weight_min"],
            # "neg_weight_mean": drift_info["neg_weight_mean"],
            # "neg_weight_max": drift_info["neg_weight_max"],
            # "neg_weight_min": drift_info["neg_weight_min"],
            # "pos_dist_mean": drift_info["pos_dist_mean"],
            # "neg_dist_mean": drift_info["neg_dist_mean"],
            # "pos_mass_mean": drift_info["pos_mass_mean"],
            # "neg_mass_mean": drift_info["neg_mass_mean"],
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute svgd actor loss: L_BC."""
        if self.config["action_chunking"]:
            batch_size = batch["actions"].shape[0]
            actor_action_dim = batch["actions"].shape[1] * batch["actions"].shape[2]
            batch_actions = jnp.reshape(batch["actions"], (batch_size, actor_action_dim))
        else:
            batch_actions = batch["actions"][..., 0, :]
            batch_size, actor_action_dim = batch_actions.shape
        positive_actions = batch_actions[:, None, :]

        rng, generated_rng, topk_rng = jax.random.split(rng, 3)

        noises = jax.random.normal(generated_rng, (batch_size, self.config["num_generated_actions"], actor_action_dim))
        observations = jnp.repeat(
            batch["observations"][..., None, :],
            self.config["num_generated_actions"],
            axis=-2,
        )
        generated_actions = self._apply_actor(
            observations,
            noises,
            params=grad_params,
        )
        if self.config["action_q_agg"] == "mean":
            q_fn = lambda x: self.network.select("critic")(observations, x).mean(axis=0).mean()
        else:
            q_fn = lambda x: self.network.select("critic")(observations, x).min(axis=0).mean()
        score = jax.grad(q_fn)(generated_actions)
        q_value = q_fn(generated_actions) 
        # generated_actions = jnp.clip(generated_actions, -1, 1)
        bc_loss, bc_drift_info = self._drift_loss(
            query_actions=generated_actions,            # [D, Q, A] or [D, Q, H * A]
            positive_actions=positive_actions,          # [D, P, A] or [D, P, H * A]
            q_value=q_value,
            score=score,
            batch=batch,
        )
        actor_loss = bc_loss  #+ q_loss # 

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "generated_to_data_mse": self._masked_action_mse(
                jnp.square(generated_actions - batch_actions[:, None, :]),
                batch,
            ),
            "score_mean": jnp.mean(score),
            "valid" : batch["valid"].mean(),
            **{k: v for k, v in bc_drift_info.items()},
            # **{f"topk_{k}": v for k, v in topk_drift_info.items()},
            # **topk_info,
        }

    def update_old_actor(self, network):
        """EMA update: theta_old <- tau_ema * theta + (1 - tau_ema) * theta_old."""
        new_old_actor_params = jax.tree_util.tree_map(
            lambda p, op: self.config["actor_ema_tau"] * p + (1.0 - self.config["actor_ema_tau"]) * op,
            network.params["modules_actor"],
            self.network.params["modules_old_actor"],
        )
        network.params["modules_old_actor"] = new_old_actor_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent, target critic, and EMA old actor."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        agent.update_old_actor(new_network)
        return agent.replace(network=new_network, rng=new_rng), info

    @staticmethod
    def _offline_update(agent, batch):
        """Update the agent during offline training without old-actor EMA."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        if self.config["online_learning"]:
            return self._update(self, batch)
        return self._offline_update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        if self.config["online_learning"]:
            update_fn = self._update
        else:
            update_fn = self._offline_update
        agent, infos = jax.lax.scan(update_fn, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)


    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
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
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self._apply_actor(observations, noises)
            actions = jnp.clip(actions, -1, 1)

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
            actions = self._apply_actor(observations, noises)
            actions = jnp.clip(actions, -1, 1)

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
            encoders["actor"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
        )
        actor_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_actions.shape[-1],
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor"),
        )

        network_info = dict(
            actor=(actor_def, (ex_observations, full_actions)),
            old_actor=(copy.deepcopy(actor_def), (ex_observations, full_actions)),
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
        params["modules_old_actor"] = params["modules_actor"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="svgd",
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
            action_q_agg="mean",
            num_qs=2,
            epsilon=1e-5,
            bandwidth=0.05,
            score_gain = 0.,
            online_score_gain = 0,
            num_generated_actions=8,
            actor_ema_tau=1., #1e-4,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            online_learning=False,
            actor_type="best-of-n",
            actor_num_samples=16,# 16, #for bast of n
            weight_decay=0.0,
        )
    )
    return config
