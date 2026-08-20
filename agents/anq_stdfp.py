"""Standalone value-free ANQ with a stochastic latent drift policy.

Networks: latent-noise actor, drift decoder, refinement actor, critic ensemble,
temperature, and target copies of the critic and the refinement actor.

The latent actor selects a behavior mode by proposing the noise fed to the drift
decoder.  The decoder is behavior-cloned on ``z ~ N(0, I)``, so the latent head
is an unsquashed diagonal Gaussian over the same support and its deviation is
controlled by an analytic KL to that prior (see ``noise_squash_tanh``).
"""

import copy
import math
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections

from utils.drift_loss import drift_loss
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    Actor,
    ActorVectorField,
    LogParam,
    MLP,
    Normal,
    TanhNormal,
    Value,
)
from utils.optimizers import make_optimizer


def td_expectile_loss(td_error, expectile):
    weight = jnp.where(td_error > 0.0, expectile, 1.0 - expectile)
    return weight * jnp.square(td_error)


def improvement_penalty_weight(
    improvement,
    alpha,
    improvement_clip,
    weight_min,
    weight_max,
):
    """Return a stopped exponential weight for a target-Q improvement."""
    improvement = jnp.clip(
        improvement, -improvement_clip, improvement_clip
    )
    weight = jnp.exp(-alpha * improvement)
    weight = jnp.clip(weight, weight_min, weight_max)
    return jax.lax.stop_gradient(weight)


def select_best(actions, scores):
    indices = jnp.argmax(scores, axis=-1)
    batch_shape = indices.shape
    flat_indices = indices.reshape(-1)
    flat_actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    selected = flat_actions[jnp.arange(flat_indices.size), flat_indices]
    return selected.reshape(batch_shape + (actions.shape[-1],))


class ANQSTDFPAgent(flax.struct.PyTreeNode):
    """Independent latent actor, drift BC, refiner, and expectile critic."""

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
        raise ValueError(f"Unsupported Q aggregation: {mode}")

    def _safe_clip(self, actions):
        actions = jnp.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        return jnp.clip(actions, -1.0, 1.0)

    def _add_output_noise(self, actions, rng):
        scale = self.config["noise_scale"]
        if scale == 0.0 or rng is None:
            return actions
        return actions + scale * jax.random.normal(
            rng, actions.shape, dtype=actions.dtype
        )

    def _refine(
        self, observations, base_actions, params=None, actor_name="refine_actor"
    ):
        # ``base_scale`` interpolates this agent to ReBRAC without changing any
        # other component.  At 0 the refiner sees a constant base, adds nothing
        # to it, and therefore emits the action directly from the observation --
        # exactly ReBRAC's actor, but with this agent's critic, anchor and
        # optimizer held fixed.  At 1 it is the full residual-on-drift policy.
        base_actions = self._safe_clip(base_actions) * self.config["base_scale"]
        inputs = jnp.concatenate([observations, base_actions], axis=-1)
        raw_delta = self.network.select(actor_name)(
            inputs, params=params
        ).mode()
        if self.config["refine_residual_space"] == "pretanh":
            # ``clip(base + delta)`` has exactly zero gradient wherever the sum
            # leaves [-1, 1], and the drift decoder saturates often here: 26% of
            # antmaze-giant dataset action components exceed 0.9 and the decoder
            # output is unbounded before ``_safe_clip``.  Applying the residual
            # in pre-tanh space keeps the action in (-1, 1) with a live gradient
            # in every dim, and delta = 0 still reproduces the base exactly.
            bound = 1.0 - 1e-4
            pre_base = jnp.arctanh(jnp.clip(base_actions, -bound, bound))
            refined = self._safe_clip(jnp.tanh(pre_base + raw_delta))
            return refined, refined - base_actions
        delta = raw_delta
        refined = self._safe_clip(base_actions + delta)
        return refined, refined - base_actions

    def _decode(self, observations, noises, output_rng=None):
        base = self.network.select("actor_drift")(observations, noises)
        base = self._add_output_noise(base, output_rng)
        return self._refine(observations, base)

    def _latent_base_actions(self, observations, rng):
        """Decode the current latent actor's action.

        Under ``latent_deterministic`` the mode is used, matching what
        ``_sample_refined_actions`` executes.  The refiner is then trained on the
        same base it is deployed on, and the ``||delta||^2`` penalty gets a fixed
        per-state anchor instead of one resampled every gradient step.
        """
        dist = self.network.select("noise_actor")(observations)
        if self.config["latent_deterministic"]:
            raw_noises = dist.mode()
        else:
            raw_noises = dist.sample(seed=rng)
        noises = raw_noises * self.config["latent_noise_scale"]
        base = self.network.select("actor_drift")(observations, noises)
        return (
            jax.lax.stop_gradient(self._safe_clip(base)),
            jax.lax.stop_gradient(noises),
        )

    def _drift_base_actions(self, observations, rng):
        """Decode unit-Gaussian noise without using the latent actor."""
        noises = jax.random.normal(
            rng,
            observations.shape[:-1] + (self.config["full_action_dim"],),
        )
        base = self.network.select("actor_drift")(observations, noises)
        return (
            jax.lax.stop_gradient(self._safe_clip(base)),
            jax.lax.stop_gradient(noises),
        )

    def _refine_base_actions(self, observations, rng):
        if self.config["refine_base_source"] == "latent":
            return self._latent_base_actions(observations, rng)
        return self._drift_base_actions(observations, rng)

    def critic_loss(self, batch, grad_params, rng):
        next_observations = batch["next_observations"][..., -1, :]
        # next_actions, target_delta = self._sample_refined_actions(
        #     next_observations, rng, critic_name="target_critic"
        # )
        rng1, rng2 = jax.random.split(rng)
        # The bootstrap must evaluate the policy that is actually executed, i.e.
        # the refined action.  ``target_base_mix`` optionally blends in the value
        # of the *unrefined* base action; it is zero by default.
        mu = self.config["target_base_mix"]
        next_action2, _ = self._sample_refined_actions(
            next_observations, rng2, actor_name="target_refine_actor"
        )
        next_qs2 = self.network.select("target_critic")(
            next_observations, actions=next_action2
        )
        next_q2 = self._aggregate_q(next_qs2)
        if mu == 0.0:
            next_q = next_q2
        else:
            next_action1, _ = self._latent_base_actions(
                next_observations, rng1
            )
            next_qs1 = self.network.select("target_critic")(
                next_observations, actions=next_action1
            )
            next_q = mu * self._aggregate_q(next_qs1) + (1.0 - mu) * next_q2
        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        qs = self.network.select("critic")(
            batch["observations"],
            actions=self._batch_actions(batch),
            params=grad_params,
        )
        losses = td_expectile_loss(
            target_q[None, ...] - qs,
            self.config["critic_expectile"],
        )
        valid = batch.get("valid", jnp.ones_like(batch["rewards"]))[..., -1]
        valid = jnp.broadcast_to(valid[None, ...], losses.shape)
        loss = (losses * valid).sum() / jnp.maximum(valid.sum(), 1.0)
        return loss, {
            "critic_loss": loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q": target_q.mean(),
        }

    def drift_bc_loss(self, batch, grad_params, rng):
        actions = self._batch_actions(batch)
        batch_size, action_dim = actions.shape
        observations = jnp.repeat(
            batch["observations"], self.config["gen_per_label"], axis=0
        )
        # actions = jnp.repeat(
        #     actions, self.config["gen_per_label"], axis=0
        # ).reshape(batch_size, self.config["gen_per_label"], action_dim)
        noises = jax.random.normal(
            rng,
            (batch_size * self.config["gen_per_label"], action_dim),
        )
        generated = self.network.select("actor_drift")(
            observations, noises, params=grad_params
        ).reshape(batch_size, self.config["gen_per_label"], action_dim)
        losses, drift_info = drift_loss(
            gen=generated,
            fixed_pos=actions[..., None, :],
            R_list=(self.config["drift_temps"],),
            force_norm=self.config["drift_force_norm"] if "drift_force_norm" in self.config else "unit",
        )
        loss = losses.mean()
        info = {
            "actor_drift_loss": loss,
            "drift_scale": drift_info.get("scale", 0.0),
            "generated_to_data_mse": jnp.mean(
                jnp.square(generated - actions[..., None, :])
            ),
        }
        for key, value in drift_info.items():
            if key.startswith("loss_"):
                info[f"drift_{key}"] = value
        return loss, info

    def refine_actor_loss(self, batch, grad_params, rng):
        observations = batch["observations"]
        base, noises = self._refine_base_actions(observations, rng)
        refined, delta = self._refine(
            observations, base, params=grad_params
        )
        qs = self.network.select("critic")(observations, actions=refined)
        q = self._aggregate_q(qs, mode=self.config["refine_q_agg"])

        # What the ||.||^2 trust region is measured against.  "base" keeps the
        # refined action near the generative sample it started from; "data"
        # regularizes toward the dataset action instead (ReBRAC-style), which
        # decouples the trust region from the stochastic base.
        if self.config["refine_anchor"] == "data":
            offset = refined - self._batch_actions(batch)
        else:
            offset = delta

        norm_q = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        loss = - norm_q * q.mean() + self.config["lam"] * ((offset**2).sum(axis=-1) * batch["valid"][..., -1]).mean()

        return loss, {
            "refine_actor_loss": loss,
            "refine_q": q.mean(),
            "delta_rms": jnp.sqrt(jnp.mean(jnp.square(delta))),
            "offset_rms": jnp.sqrt(jnp.mean(jnp.square(offset))),

            # "base_action_rms": jnp.sqrt(jnp.mean(jnp.square(base))),
            # "base_noise_std": noises.std(),
            # "uses_latent_base": jnp.asarray(
            #     self.config["refine_base_source"] == "latent",
            #     dtype=jnp.float32,
            # ),
        }

    def _latent_kl(self, dist, noises, log_probs, scale):
        """KL of the scaled latent noise distribution against ``N(0, I)``.

        The unsquashed head is a diagonal Gaussian, so the KL is available in
        closed form and is used directly.  A tanh-squashed head has no closed
        form, and its support is bounded, so it falls back to the one-sample
        estimator (and carries an irreducible KL floor of ``-d*log P(|z|<1)``).
        """
        if self.config["noise_squash_tanh"]:
            log_prior = -0.5 * (
                jnp.square(noises) + math.log(2.0 * math.pi)
            ).sum(axis=-1)
            return log_probs - log_prior
        mean = dist.loc * scale
        std = dist.stddev() * scale
        return 0.5 * (
            jnp.square(std) + jnp.square(mean) - 1.0 - 2.0 * jnp.log(std)
        ).sum(axis=-1)

    def latent_actor_loss(self, batch, grad_params, rng):
        observations = batch["observations"]
        dist = self.network.select("noise_actor")(
            observations, params=grad_params
        )
        raw_noises = dist.sample(seed=rng)
        scale = self.config["latent_noise_scale"]
        noises = raw_noises * scale
        log_probs = dist.log_prob(raw_noises) - self.config[
            "full_action_dim"
        ] * jnp.log(scale)
        actions, _ = self._decode(observations, noises)
        qs = self.network.select("critic")(observations, actions=actions)
        q = self._aggregate_q(qs, mode=self.config["base_q_agg"])

        alpha = self.network.select("noise_alpha")()
        train_alpha = self.network.select("noise_alpha")(params=grad_params)
        entropy = -log_probs
        if self.config["noise_regularizer"] == "entropy":
            policy_loss = (alpha * log_probs - q).mean()
            alpha_loss = (
                train_alpha
                * (
                    jax.lax.stop_gradient(entropy)
                    - self.config["noise_target_entropy"]
                )
            ).mean()
            kl = jnp.zeros_like(log_probs)
        else:
            kl = self._latent_kl(dist, noises, log_probs, scale)
            policy_loss = (alpha * kl - q).mean()
            alpha_loss = (
                train_alpha
                * (
                    self.config["noise_target_kl"]
                    - jax.lax.stop_gradient(kl)
                )
            ).mean()

        return policy_loss + alpha_loss, {
            "latent_policy_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "alpha": train_alpha,
            "latent_q": q.mean(),
            "latent_entropy": entropy.mean(),
            "latent_kl": kl.mean(),
            "noise_abs_mean": jnp.abs(noises).mean(),
            "noise_std": noises.std(),
        }

    def actor_loss(self, batch, grad_params, rng):
        drift_rng, refine_rng, latent_rng = jax.random.split(rng, 3)
        bc_loss, info = self.drift_bc_loss(batch, grad_params, drift_rng)
        # Drift-BC early stop (see stdfp.py): decoder is trained only by this
        # loss, so zeroing it after bc_stop_step freezes the decoder before the
        # BC fit overfits.
        bc_stop = self.config["bc_stop_step"] if "bc_stop_step" in self.config else 0
        if bc_stop:
            bc_on = (self.network.step < bc_stop).astype(bc_loss.dtype)
            bc_loss = bc_loss * bc_on
        refine_loss, refine_info = self.refine_actor_loss(
            batch, grad_params, refine_rng
        )
        latent_loss, latent_info = self.latent_actor_loss(
            batch, grad_params, latent_rng
        )
        loss = bc_loss + refine_loss + latent_loss
        info.update(refine_info)
        info.update(latent_info)
        info["total_loss"] = loss
        return loss, info

    def actor_loss_frozen_bc(self, batch, grad_params, rng):
        refine_rng, latent_rng = jax.random.split(rng)
        refine_loss, refine_info = self.refine_actor_loss(
            batch, grad_params, refine_rng
        )
        latent_loss, latent_info = self.latent_actor_loss(
            batch, grad_params, latent_rng
        )
        loss = refine_loss + latent_loss
        info = {}
        info.update(refine_info)
        info.update(latent_info)
        info["total_loss"] = loss
        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        actor_rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(
            batch, grad_params, critic_rng
        )
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info = {"total_loss": critic_loss + actor_loss}
        info.update({f"critic/{k}": v for k, v in critic_info.items()})
        info.update({f"actor/{k}": v for k, v in actor_info.items()})
        return critic_loss + actor_loss, info

    @jax.jit
    def total_loss_frozen_bc(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        actor_rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(
            batch, grad_params, critic_rng
        )
        actor_loss, actor_info = self.actor_loss_frozen_bc(
            batch, grad_params, actor_rng
        )
        info = {"total_loss": critic_loss + actor_loss}
        info.update({f"critic/{k}": v for k, v in critic_info.items()})
        info.update({f"actor/{k}": v for k, v in actor_info.items()})
        return critic_loss + actor_loss, info

    def _target_update(self, network, module_name):
        """EMA the target copy of ``module_name`` toward the updated params."""
        source = network.params[f"modules_{module_name}"]
        target = self.network.params[f"modules_target_{module_name}"]
        network.params[f"modules_target_{module_name}"] = jax.tree_util.tree_map(
            lambda p, tp: self.config["tau"] * p
            + (1.0 - self.config["tau"]) * tp,
            source,
            target,
        )

    @staticmethod
    def _update(agent, batch):
        new_rng, loss_rng = jax.random.split(agent.rng)
        new_network, info = agent.network.apply_loss_fn(
            lambda params: agent.total_loss(batch, params, loss_rng)
        )
        agent._target_update(new_network, "critic")
        agent._target_update(new_network, "refine_actor")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def pretrain_bc_loss(self, batch, grad_params, rng):
        """Compute only the drift behavior-cloning loss."""
        loss, info = self.drift_bc_loss(batch, grad_params, rng)
        info["bc_pretrain_loss"] = loss
        return loss, info

    @staticmethod
    def _pretrain_bc_update(agent, batch):
        new_rng, loss_rng = jax.random.split(agent.rng)
        new_network, info = agent.network.apply_loss_fn(
            lambda params: agent.pretrain_bc_loss(batch, params, loss_rng)
        )
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def pretrain_bc_update(self, batch):
        return self._pretrain_bc_update(self, batch)

    def frozen_bc_module_keys(self):
        return tuple(
            key
            for key in ("modules_actor_drift", "modules_target_actor_drift")
            if key in self.network.params
        )

    @staticmethod
    def _update_frozen_bc(agent, batch):
        new_rng, loss_rng = jax.random.split(agent.rng)
        new_network, info = agent.network.apply_loss_fn_with_frozen_modules(
            loss_fn=lambda params: agent.total_loss_frozen_bc(
                batch, params, loss_rng
            ),
            frozen_module_keys=agent.frozen_bc_module_keys(),
        )
        agent._target_update(new_network, "critic")
        agent._target_update(new_network, "refine_actor")
        info["bc_frozen"] = jnp.asarray(1.0)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_frozen_bc(self, batch):
        return self._update_frozen_bc(self, batch)

    @jax.jit
    def batch_update_frozen_bc(self, batch):
        agent, infos = jax.lax.scan(self._update_frozen_bc, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def _sample_refined_actions(
        self, observations, rng, critic_name="critic", actor_name="refine_actor"
    ):
        rng = jax.random.PRNGKey(0) if rng is None else rng
        latent_rng, output_rng = jax.random.split(rng)
        observations = jnp.repeat(
            observations[..., None, :], self.config["best_of_n"], axis=-2
        )
        dist = self.network.select("noise_actor")(observations)
        # Resampling z every step injects behavior-mode noise into the executed
        # action.  Measured on antmaze-giant that jitter is ~2.8x the refiner's
        # Q-directed correction and is Q-neutral, so the mode is used by default
        # and both the executed policy and the TD target stay deterministic.
        # ``best_of_n`` was previously a no-op: the observation is tiled n times
        # above, but ``dist.mode()`` is a deterministic function of it, so all n
        # candidates were byte-identical and ``select_best`` ranked n copies of
        # one action.  Sampling is therefore forced whenever n > 1, which is the
        # only way the tiling produces distinct candidates to rank.
        if self.config["latent_deterministic"] and self.config["best_of_n"] == 1:
            raw_noises = dist.mode()
        else:
            raw_noises = dist.sample(seed=latent_rng)
        noises = raw_noises * self.config["latent_noise_scale"]
        base = self.network.select("actor_drift")(observations, noises)
        base = self._add_output_noise(base, output_rng)
        refined, delta = self._refine(observations, base, actor_name=actor_name)
        scores = self._aggregate_q(
            self.network.select(critic_name)(observations, actions=refined),
            mode=self.config["bfn_q_agg"],
        )
        return select_best(refined, scores), select_best(delta, scores)

    @partial(jax.jit, static_argnames=("critic_name",))
    def sample_actions(self, observations, rng=None, critic_name="critic"):
        actions, _ = self._sample_refined_actions(
            observations, rng, critic_name=critic_name
        )
        return self._safe_clip(actions)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        cls._validate_config(config)
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        action_dim = ex_actions.shape[-1]
        full_actions = (
            jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
            if config["action_chunking"]
            else ex_actions
        )
        full_action_dim = full_actions.shape[-1]
        refine_inputs = jnp.concatenate([ex_observations, full_actions], axis=-1)
        if config["noise_target_entropy"] is None:
            config["noise_target_entropy"] = (
                config["target_multiplier"] * full_action_dim
            )
        if config["noise_target_kl"] is None:
            config["noise_target_kl"] = (
                config["target_multiplier"] * full_action_dim
            )

        encoders = {}
        if config["encoder"] is not None:
            encoder = encoder_modules[config["encoder"]]
            encoders = {
                "critic": encoder(),
                "noise_actor": encoder(),
                "actor_drift": encoder(),
            }
        critic = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
        )
        noise_actor_base = partial(
            MLP,
            hidden_dims=config["actor_hidden_dims"],
            activate_final=True,
            layer_norm=config["actor_layer_norm"],
        )
        # ``actor_drift`` is behavior-cloned on z ~ N(0, I).  A tanh-squashed
        # latent head is confined to (-1, 1)^d, which for d=8 covers only ~4.7%
        # of that prior mass and costs ~0.38*d nats of irreducible KL, so the
        # unsquashed Gaussian head is the default.
        noise_actor_cls = TanhNormal if config["noise_squash_tanh"] else Normal
        noise_actor = noise_actor_cls(
            noise_actor_base,
            full_action_dim,
            state_dependent_std=config["noise_state_dependent_std"],
            encoder=encoders.get("noise_actor"),
        )
        actor_drift = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_drift"),
        )
        refine_actor = Actor(
            hidden_dims=config["refine_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["refine_layer_norm"],
            # In "pretanh" space the squash is applied after the residual add,
            # so the head itself must be unbounded.
            tanh_squash=config["refine_residual_space"] == "action",
            final_fc_init_scale=config["refine_fc_scale"],
        )
        definitions = {
            "critic": (critic, (ex_observations, full_actions)),
            "target_critic": (
                copy.deepcopy(critic),
                (ex_observations, full_actions),
            ),
            "noise_actor": (noise_actor, (ex_observations,)),
            "noise_alpha": (
                LogParam(init_value=config["noise_init_temp"]),
                (),
            ),
            "actor_drift": (actor_drift, (ex_observations, full_actions)),
            "refine_actor": (refine_actor, (refine_inputs,)),
            "target_refine_actor": (
                copy.deepcopy(refine_actor),
                (refine_inputs,),
            ),
        }
        network_def = ModuleDict({k: v[0] for k, v in definitions.items()})
        params = network_def.init(
            init_rng, **{k: v[1] for k, v in definitions.items()}
        )["params"]
        params["modules_target_critic"] = params["modules_critic"]
        params["modules_target_refine_actor"] = params["modules_refine_actor"]
        network = TrainState.create(
            network_def,
            params,
            tx=make_optimizer(
                config["optimizer"], config["lr"],
                eps=config["adam_eps"] if "adam_eps" in config else 1e-8,
            ),
        )
        config["ob_dims"] = ex_observations.shape
        config["action_dim"] = action_dim
        config["full_action_dim"] = full_action_dim
        return cls(rng, network, flax.core.FrozenDict(**config))

    @staticmethod
    def _validate_config(config):
        q_modes = ("min", "mean", "pessimistic")
        for key in (
            "q_agg",
            "refine_q_agg",
            "improvement_q_agg",
            "base_q_agg",
            "bfn_q_agg",
        ):
            if config[key] not in q_modes:
                raise ValueError(f"{key} must be one of {q_modes}")
        if not 0.0 < config["critic_expectile"] < 1.0:
            raise ValueError("critic_expectile must be in (0, 1)")
        if config["noise_regularizer"] not in ("entropy", "kl"):
            raise ValueError("noise_regularizer must be 'entropy' or 'kl'")
        if config["refine_base_source"] not in ("latent", "drift"):
            raise ValueError(
                "refine_base_source must be 'latent' or 'drift'"
            )
        if config["refine_anchor"] not in ("base", "data"):
            raise ValueError("refine_anchor must be 'base' or 'data'")
        if not 0.0 <= config["base_scale"] <= 1.0:
            raise ValueError("base_scale must be in [0, 1]")
        if config["refine_residual_space"] not in ("action", "pretanh"):
            raise ValueError(
                "refine_residual_space must be 'action' or 'pretanh'"
            )
        if not 0.0 <= config["target_base_mix"] <= 1.0:
            raise ValueError("target_base_mix must be in [0, 1]")
        if config["num_qs"] < 1 or config["best_of_n"] < 1:
            raise ValueError("num_qs and best_of_n must be positive")
        if config["gen_per_label"] < 2:
            raise ValueError("gen_per_label must be at least 2")
        if config["latent_noise_scale"] <= 0.0:
            raise ValueError("latent_noise_scale must be positive")
        if config["noise_scale"] < 0.0:
            raise ValueError("noise_scale must be non-negative")
        if config["alpha"] < 0.0:
            raise ValueError("alpha must be non-negative")
        if config["improvement_clip"] <= 0.0:
            raise ValueError("improvement_clip must be positive")
def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="anq_stdfp",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            full_action_dim=ml_collections.config_dict.placeholder(int),
            optimizer="adam",
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            refine_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            layer_norm=True,
            refine_layer_norm=False,
            refine_fc_scale=0.01,
            discount=0.99,
            tau=0.005,
            num_qs=2,
            rho=0.5,
            q_agg="min",
            refine_q_agg="mean",
            improvement_q_agg="mean",
            base_q_agg="mean",
            bfn_q_agg="mean",
            critic_expectile=0.5,
            refine_base_source="latent",
            target_base_mix=0.0,    # weight on Q(s', unrefined base) in the TD target
            noise_squash_tanh=False,  # True restores the (-1,1)^d bounded latent head
            latent_deterministic=True,  # False resamples z per step at action time
            # Anchoring the trust region on the dataset action rather than on the
            # drift sample the policy happened to draw is what makes long-horizon
            # locomotion work: antmaze-giant-task2 goes 0.00 -> 0.68 at 500k.
            refine_anchor="data",       # "base" restores the drift-sample anchor
            # How the refiner's output is combined with the drift base.
            # "action": refined = clip(base + delta)  -- loses the gradient in
            #   every dim where the sum leaves the action box.
            # "pretanh": refined = tanh(atanh(base) + delta)  -- same fixed
            #   point at delta = 0, but differentiable everywhere.
            refine_residual_space="action",
            # 1.0 = full residual-on-drift policy, 0.0 = ReBRAC's actor.
            base_scale=1.0,
            # With refine_anchor="data" this is ReBRAC's actor BC coefficient and
            # 0.01 is the value validated on antmaze-giant.  The old "base"
            # anchor used much larger values (1-20); they do not carry over.
            lam=0.01,
            alpha=0.0,
            improvement_clip=10.0,
            refine_weight_min=0.01,
            refine_weight_max=10.0,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            best_of_n=1,
            gen_per_label=8,
            drift_temps=0.1,
            bc_stop_step=0,
            adam_eps=1e-8,
            drift_force_norm="unit",  # "raw" = un-normalised force, BC anneals naturally
            noise_regularizer="kl",
            noise_state_dependent_std=False,
            noise_target_entropy=ml_collections.config_dict.placeholder(float),
            noise_target_kl=ml_collections.config_dict.placeholder(float),
            # Nats of latent deviation per action dim.  With the unsquashed head
            # the whole budget is usable; the old tanh head burned ~0.38/dim on
            # its support floor, leaving 0.5-0.38 ~= 0.12 effective.
            target_multiplier=0.125,
            noise_init_temp=1.0,
            noise_scale=0.0,
            latent_noise_scale=1.0,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
