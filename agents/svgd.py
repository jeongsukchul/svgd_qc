import copy

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from typing import Any

from agents.acfql import ACFQLAgent
from utils.encoders import encoder_modules
from utils.flax_utils import (
    ModuleDict,
    TrainState,
    nonpytree_field,
    restore_partial_modules,
)
from utils.networks import ActorVectorField, Value
from functools import partial

class SVGDAgent(ACFQLAgent):
    """Drifting Field Policy agent with the behavior-cloning drift loss."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    score_gain: jnp.ndarray
    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def _use_iql_update(self):
        return self.config.get("update_flag", "td") == "iql"

    def _aggregate_q(self, qs, mode=None):
        mode = self.config["q_agg"] if mode is None else mode
        if mode == "min":
            return qs.min(axis=0)
        if mode == "mean":
            return qs.mean(axis=0)
        if mode == "pessimistic":
            return qs.mean(axis=0) - self.config["rho"] * qs.std(axis=0)
        raise ValueError(f"Unsupported q_agg: {mode}")

    def _aggregate_action_q(self, qs):
        return self._aggregate_q(qs, mode=self.config["action_q_agg"])

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        batch_actions = self._batch_actions(batch)
        target_qs = self.network.select("target_critic")(
            batch["observations"],
            actions=batch_actions,
        )
        q = self._aggregate_q(target_qs)
        v = self.network.select("value")(batch["observations"], params=grad_params)
        valid = batch["valid"][..., -1] if "valid" in batch else jnp.ones_like(v)
        value_loss = (
            self.expectile_loss(q - v, q - v, self.config["expectile"]) * valid
        ).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def _td_critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=rng, use_q_bfn=True)
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
        next_q = self._aggregate_q(next_qs)

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        total_loss = critic_loss
        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def _iql_critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        batch_actions = self._batch_actions(batch)
        next_v = self.network.select("value")(batch["next_observations"][..., -1, :])
        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_v

        qs = self.network.select("critic")(
            batch["observations"],
            actions=batch_actions,
            params=grad_params,
        )
        valid = batch["valid"][..., -1] if "valid" in batch else jnp.ones_like(target_q)
        critic_loss = (
            jnp.square(qs - target_q[None, ...]) * valid[None, ...]
        ).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_v_mean": next_v.mean(),
        }

    def critic_loss(self, batch, grad_params, rng):
        if self._use_iql_update():
            return self._iql_critic_loss(batch, grad_params)
        return self._td_critic_loss(batch, grad_params, rng)

    def _apply_actor(self, observations, noises, params=None):
        actions = self.network.select("actor")(observations, noises, params=params)

        return actions
    def _apply_bc_actor(self, observations, noises, params=None):
        actions = self.network.select("bc_actor")(observations, noises, params=params)

        return actions
    def _apply_target_bc_actor(self, observations, noises):
        actions = self.network.select("target_bc_actor")(observations, noises)

        return actions
    def _apply_old_actor(self, observations, noises):
        actions = self.network.select("old_actor")(observations, noises)

        return actions

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

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
    def _sample_positive_actions(self, batch, batch_actions, rng, params=None):
        batch_size, actor_action_dim = batch_actions.shape
        positive_actions = batch_actions[:, None, :]
        pos_samples = self.config.get("pos_samples", 8)

        if pos_samples <= 0:
            return positive_actions

        candidate_samples = max(
            self.config.get("positive_topk_candidates", 32),
            pos_samples,
        )
        bc_drift_noises = jax.random.normal(
            rng,
            (batch_size * candidate_samples, actor_action_dim),
        )
        pos_observations = jnp.repeat(batch["observations"], candidate_samples, axis=0)
        if self.config.get("use_target_bc_actor", True):
            positive_actions = self._apply_target_bc_actor(
                pos_observations,
                bc_drift_noises,
            ).clip(-1, 1)
        else:
            positive_actions = self._apply_bc_actor(
                pos_observations,
                bc_drift_noises,
                params=params,
            ).clip(-1, 1)

        positive_actions = positive_actions.reshape(
            batch_size, candidate_samples, actor_action_dim
        )
        critic_observations = jnp.repeat(batch["observations"], candidate_samples, axis=0)
        qs = self.network.select("critic")(
            critic_observations,
            actions=positive_actions.reshape(batch_size * candidate_samples, actor_action_dim),
        )
        q_values = self._aggregate_action_q(qs)
        q_values = q_values.reshape(batch_size, candidate_samples)
        _, topk_idx = jax.lax.top_k(q_values, pos_samples)
        return jnp.take_along_axis(positive_actions, topk_idx[..., None], axis=1)

    def _score_gain(self):
        if self.config.get("use_dual_score_gain", False):
            return jnp.clip(
                self.score_gain,
                a_min=self.config["score_gain_min"],
                a_max=self.config["score_gain_max"],
            )

        if self.config["online_learning"]:
            return self.config["online_score_gain"]

        return self.config["score_gain"]

    def _normalize_q_values(self, q_values):
        if not self.config.get("normalize_q", False):
            return q_values

        q_mean = jnp.mean(q_values, axis=-1, keepdims=True)
        q_std = jnp.std(q_values, axis=-1, keepdims=True)
        if self.config.get("q_norm_stop_grad_stats", True):
            q_mean = jax.lax.stop_gradient(q_mean)
            q_std = jax.lax.stop_gradient(q_std)

        return (q_values - q_mean) / jnp.maximum(q_std, self.config["q_norm_eps"])

    def dual_objective(
        self,
        score_gain,
        positive_actions,
        old_actions,
        q_value_old,
    ):
        epsilon = self.config["epsilon"]
        sigma = self.config["dual_budget"]
        num_old_actions = old_actions.shape[1]

        diff = positive_actions[:, :, None, :] - old_actions[:, None, :, :]
        cost = jnp.sum(diff ** 2, axis=-1)

        logits = (
            score_gain * q_value_old[:, None, :]
            - cost
        ) / epsilon

        log_inner = (
            jax.scipy.special.logsumexp(logits, axis=-1)
            - jnp.log(num_old_actions)
        )
        inv_score_gain = 1.0 / score_gain
        return sigma * inv_score_gain + epsilon * inv_score_gain * jnp.mean(log_inner)

    def dual_update(
        self,
        positive_actions,
        old_actions,
        q_value_old,
    ):
        score_gain = jnp.clip(
            self.score_gain,
            self.config["score_gain_min"],
            self.config["score_gain_max"],
        )
        log_score_gain = jnp.log(score_gain)

        def objective(log_score_gain):
            return self.dual_objective(
                jnp.exp(log_score_gain),
                jax.lax.stop_gradient(positive_actions),
                jax.lax.stop_gradient(old_actions),
                jax.lax.stop_gradient(q_value_old),
            )

        dual_loss, dual_log_grad = jax.value_and_grad(objective)(log_score_gain)
        dual_grad = dual_log_grad / score_gain

        new_log_score_gain = log_score_gain - self.config["eta_score_gain"] * dual_log_grad
        new_score_gain = jnp.exp(new_log_score_gain)
        new_score_gain = jnp.clip(
            new_score_gain,
            self.config["score_gain_min"],
            self.config["score_gain_max"],
        )

        dual_info = {
            "dual_loss": dual_loss,
            "score_gain_log_grad": dual_log_grad,
            "score_gain_grad": dual_grad,
            "score_gain_dual": new_score_gain,
            "score_gain_delta": new_score_gain - score_gain,
            # d/d log(score_gain) = score_gain * d/d score_gain.
            "dual_ot_estimate": self.config["dual_budget"] + score_gain * dual_log_grad,
        }

        return new_score_gain, dual_info
    def _sinkhorn_drift(self, query_actions, positive_actions, old_actions, q_value_old, score):
        """Compute drift field with the toy mean-drift kernel normalization."""
        epsilon = self.config["epsilon"]
        bandwidth = self.config["bandwidth"]
        score_gain = self._score_gain()

        old_diff = old_actions[:, None, :, :] - query_actions[:, :, None, :]
        target_diff = positive_actions[:, None, :, :] - query_actions[:, :, None, :]
        self_diff = query_actions[:, None, :, :] - query_actions[:, :, None, :]

        dist_pos = jnp.sum(target_diff ** 2, axis=-1)
        dist_old = jnp.sum(old_diff ** 2, axis=-1)
        dist_self = jnp.sum(self_diff ** 2, axis=-1)

        # Need q_value_old = Q(s, old_actions)
        pos_old_diff = positive_actions[:, :, None, :] - old_actions[:, None, :, :]
        dist_pos_old = jnp.sum(pos_old_diff ** 2, axis=-1)
        log_Z = jax.scipy.special.logsumexp(
            (score_gain * q_value_old[:, None, :] - dist_pos_old) / epsilon,
            axis=-1
        ) - jnp.log(old_actions.shape[1])

        weights_pos = jax.nn.softmax(
            -dist_pos / epsilon - log_Z[:, None, :],
            axis=-1
        )

        weights_neg = jax.nn.softmax(-dist_self / bandwidth, axis=-1)
        weights_old = jax.nn.softmax(-dist_old / bandwidth, axis=-1)

        local_attraction = (2.0 / epsilon) * jnp.sum(
            weights_pos[..., None] * target_diff,
            axis=-2
        )

        local_old_attraction = (2.0 / bandwidth) * jnp.sum(
            weights_old[..., None] * old_diff,
            axis=-2
        )

        attraction_term = weights_neg @ local_attraction

        repulsion_term = (2.0 / bandwidth) * jnp.sum(
            weights_neg[..., None] * self_diff,
            axis=-2
        )

        tr_term = weights_neg @ local_old_attraction

        score_term = weights_neg @ ((score_gain / epsilon) * score)

        original_drift = attraction_term - repulsion_term
        drift = original_drift + score_term + tr_term
        drift_norm = jnp.sqrt(jnp.clip((drift ** 2).mean(), a_min=1e-8))
        return drift/drift_norm, {
            "bandwidth": bandwidth,
            "score_gain": jnp.asarray(score_gain),
            'loss': drift_norm,
            'original_scale' : jnp.sqrt((original_drift ** 2).mean()) / drift_norm,
            'score_drift_scale':  jnp.sqrt((score_term ** 2).mean()) / drift_norm,
            'attraction_scale':  jnp.sqrt((attraction_term ** 2).mean()) / drift_norm,
            'repulsion_scale':  jnp.sqrt((repulsion_term ** 2).mean()) / drift_norm,
            'tr_scale':  jnp.sqrt((tr_term ** 2).mean()) / drift_norm,
        }

    def _sinkhorn_loss(self, query_actions, positive_actions, old_actions, q_value_old, score, batch):
        sinkhorn_drift, drift_info = self._sinkhorn_drift(
            query_actions=query_actions,
            positive_actions=positive_actions,
            old_actions = old_actions, 
            q_value_old=q_value_old,
            score=score,
        )
        drift_targets = jax.lax.stop_gradient(query_actions + sinkhorn_drift)
        squared_error = jnp.square(query_actions - drift_targets)
        loss = squared_error.mean() # self._masked_action_mse(squared_error, batch)

        return loss, {
            "bandwidth": drift_info["bandwidth"],
            "epsilon": self.config["epsilon"],
            "drift_norm": drift_info["loss"],
            "original_drift_norm" : drift_info["original_scale"],
            "attraction_norm" : drift_info["attraction_scale"],
            "repulsion_norm" : drift_info["repulsion_scale"],
            "score_norm" : drift_info["score_drift_scale"],
            "tr_norm" : drift_info["tr_scale"],
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

        rng, generated_rng, bc_rng, pos_rng = jax.random.split(rng, 4)
        gen_per_label = self.config.get("gen_per_label", 8)
        obs_repeated = jnp.repeat(batch["observations"], gen_per_label, axis=0)
        drift_noises = jax.random.normal(bc_rng, (batch_size * gen_per_label, actor_action_dim))
        # Get actions from drift model
        drift_actions_all = self._apply_bc_actor(obs_repeated, drift_noises, params=grad_params)
        drift_actions_all = jnp.clip(drift_actions_all, -1, 1)
        # Reshape to [B, gen_per_label, action_dim]
        gen_samples = drift_actions_all.reshape(batch_size, gen_per_label, actor_action_dim)
        bc_pos_samples = jnp.expand_dims(batch_actions, axis=1)
        from utils.drift_loss import drift_loss
        bc_drift_loss, bc_drift_info = drift_loss(
            gen=gen_samples,
            fixed_pos=bc_pos_samples,
            R_list=tuple(self.config.get("drift_temps", [0.1])),
            plus_only=bool(self.config.get("drift_plus_only", False)),
        )
        bc_drift_loss = bc_drift_loss.mean()
        #sinkhorn loss
        positive_actions = self._sample_positive_actions(
            batch,
            batch_actions,
            pos_rng,
            params=grad_params,
        )

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
        
        generated_actions = jnp.clip(generated_actions, -1, 1)
        def q_fn(x):
            q_values = self.network.select("critic")(observations, x)
            q_values = self._aggregate_action_q(q_values)
            q_values = self._normalize_q_values(q_values)
            return q_values.mean()

        score = jax.grad(q_fn)(generated_actions)

        old_actions = self._apply_old_actor(
            observations,
            noises,
        )
        old_actions = jnp.clip(old_actions, -1, 1)
        q_value_old = self._aggregate_action_q(
            self.network.select("critic")(observations, old_actions)
        )
        q_value_old = self._normalize_q_values(q_value_old)
        sinkhorn_loss, sinkhorn_info = self._sinkhorn_loss(
            query_actions=generated_actions,            # [D, Q, A] or [D, Q, H * A]
            positive_actions=positive_actions,          # [D, P, A] or [D, P, H * A]
            old_actions=old_actions,
            q_value_old=q_value_old,
            score=score,
            batch=batch,
        )
        actor_loss = bc_drift_loss + sinkhorn_loss #+ q_loss # 

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_drift_loss": bc_drift_loss,
            "sinkhorn_loss" : sinkhorn_loss,
            "generated_to_data_mse": self._masked_action_mse(
                jnp.square(generated_actions - batch_actions[:, None, :]),
                batch,
            ),
            "score_mean": jnp.mean(score),
            "q_old_mean": jnp.mean(q_value_old),
            "q_old_std": jnp.mean(jnp.std(q_value_old, axis=-1)),
            "valid" : batch["valid"].mean(),
            "_dual_positive_actions": jax.lax.stop_gradient(positive_actions),
            "_dual_old_actions": jax.lax.stop_gradient(old_actions),
            "_dual_q_value_old": jax.lax.stop_gradient(q_value_old),
            **{k: v for k, v in sinkhorn_info.items()},
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        if self._use_iql_update():
            value_loss, value_info = self.value_loss(batch, grad_params)
            for k, v in value_info.items():
                info[f"value/{k}"] = v
        else:
            value_loss = jnp.zeros(())

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def actor_ema_update(self, network):
        """Polyak update for the actor EMA."""
        tau = self.config.get("actor_ema_tau")
        new_ema_params = jax.tree_util.tree_map(
            lambda p, ep: p * tau + ep * (1 - tau),
            network.params["modules_actor"],
            self.network.params["modules_old_actor"],
        )
        network.params["modules_old_actor"] = new_ema_params
    @staticmethod
    def _update(agent, batch):
        """Update the agent, target critic, and EMA old actor."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        if agent.config.get("use_target_bc_actor", True):
            agent.target_update(new_network, "bc_actor")
        agent.actor_ema_update(new_network)

        updated_agent = agent.replace(network=new_network, rng=new_rng)

        if agent.config.get("use_dual_score_gain", False):
            positive_actions = info.pop("actor/_dual_positive_actions")
            old_actions = info.pop("actor/_dual_old_actions")
            q_value_old = info.pop("actor/_dual_q_value_old")
            new_score_gain, dual_info = updated_agent.dual_update(
                positive_actions,
                old_actions,
                q_value_old,
            )
        else:
            info.pop("actor/_dual_positive_actions", None)
            info.pop("actor/_dual_old_actions", None)
            info.pop("actor/_dual_q_value_old", None)
            new_score_gain = agent.score_gain
            dual_info = {
                "score_gain_dual": updated_agent._score_gain(),
            }

        info = {
            **info,
            **dual_info,
        }
        return updated_agent.replace(score_gain=new_score_gain), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)


    @partial(jax.jit, static_argnames=("use_q_bfn",))
    def sample_actions(
        self,
        observations,
        rng=None,
        use_q_bfn=False,
    ):
        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        if self.config["actor_type"] == "best-of-n":
            actor_samples = self.config["q_bfn"] if use_q_bfn else self.config["actor_num_samples"]

            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config["ob_dims"])],
                    actor_samples,
                    action_dim,
                ),
            )
            observations = jnp.repeat(observations[..., None, :], actor_samples, axis=-2)
            actions = self._apply_actor(observations, noises)
            actions = jnp.clip(actions, -1, 1)

            q = self._aggregate_q(self.network.select("critic")(observations, actions))
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, actor_samples, action_dim))[
                jnp.arange(bsize), indices, :].reshape(bshape + (action_dim,))
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
            encoders["actor_bc_flow"] = encoder_module()

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
            bc_actor=(copy.deepcopy(actor_def), (ex_observations, full_actions)),
            target_bc_actor=(copy.deepcopy(actor_def), (ex_observations, full_actions)),
            old_actor=(copy.deepcopy(actor_def), (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if config.get("update_flag", "td") == "iql":
            value_def = Value(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                num_ensembles=1,
                encoder=encoders.get("critic"),
            )
            network_info["value"] = (value_def, (ex_observations,))

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
        params["modules_target_bc_actor"] = params["modules_bc_actor"]
        params["modules_old_actor"] = params["modules_actor"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        init_score_gain = config.get("score_gain_init")
        if init_score_gain is None:
            init_score_gain = config["score_gain"]
        if config.get("use_dual_score_gain", False):
            init_score_gain = float(jnp.clip(
                init_score_gain,
                config["score_gain_min"],
                config["score_gain_max"],
            ))

        agent = cls(
            rng, 
            network=network, 
            config=flax.core.FrozenDict(**config),
            score_gain=jnp.asarray(init_score_gain),
        )

        return agent


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="svgd",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=1e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            q_agg="pessimistic",
            action_q_agg="mean",
            num_qs=2,
            rho=0.5,
            epsilon=0.1,
            bandwidth=0.1,
            score_gain = 0.,
            tau=0.005,
            online_score_gain = 1.,
            num_generated_actions=8,
            pos_samples=8,
            positive_topk_candidates=8,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            online_learning=False,
            actor_type="best-of-n",
            q_bfn=1,
            actor_num_samples=1,# 16, #for bast of n
            weight_decay=0.0,
            actor_ema_tau = 1.,
            use_fourier_features=False,
            fourier_feature_dim=64,

            use_dual_score_gain=True,
            normalize_q=False,
            q_norm_eps=1e-6,
            q_norm_stop_grad_stats=True,
            update_flag="td",
            expectile=0.7,
            use_target_bc_actor=False,

            # score_gain / dual score_gain
            score_gain_init=100.,
            score_gain_min=1e-3,
            score_gain_max=10000.0,
            eta_score_gain=10.,

            # closed-form dual score_gain update
            dual_budget=1.,
        )
    )
    return config
