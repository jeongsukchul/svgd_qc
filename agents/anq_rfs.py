"""RFS-style ANQ: a *unified* latent-steering + residual actor over a drift policy.

This keeps every structural piece of ``agents/anq_stdfp.py`` -- the drift
decoder behaviour-cloned on ``z ~ N(0, I)``, latent steering of that decoder,
and a residual correction on top of the decoded action -- but changes how the
two policy heads are parameterised and trained.

In ``anq_stdfp`` the latent head and the refinement head are *separate networks
optimised by separate losses*:

* ``latent_actor_loss`` moves ``z`` to maximise ``Q(s, refine(decode(s, z)))``
  while the refiner is held fixed (``_refine`` is called with ``params=None``),
  and pays an analytic KL to ``N(0, I)`` through a learned dual.
* ``refine_actor_loss`` moves ``delta`` to maximise ``Q(s, refine(base))`` while
  the base is ``stop_gradient``-ed (``_refine_base_actions``).

So neither head ever sees the other's gradient: they optimise the same action
from two disjoint objectives and cannot co-adapt.  That is the most likely
reason the composite policy underperforms a plain unimodal ReBRAC actor.

Following RFS (arXiv:2602.01789, "Reinforcement Learning with Residual Flow
Steering"), this agent instead uses **one actor network** that emits both the
latent noise ``z`` and the residual ``delta`` from a shared trunk, trained by a
**single joint objective**

    max_theta  Q(s, a(theta))  -  lam * BC(a(theta), a_data)      [ - alpha * KL(z) ]

with ``a = compose(decode(s, z), delta)``.  The drift decoder's *weights* are
frozen with respect to this objective (it is still behaviour-cloned by its own
loss), exactly as RFS freezes its pretrained flow policy, while the gradient
still flows into ``z`` through the decoder's input.

Nothing here degenerates to ReBRAC: the executed action is always produced by
the drift decoder and then steered, and there is no ``base_scale``.
"""

import copy
import math
from functools import partial
from typing import Any, Sequence, Type

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from utils.drift_loss import drift_loss
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    ActorVectorField,
    LogParam,
    MLP,
    Value,
    default_init,
)
from utils.optimizers import make_optimizer, make_module_optimizer


def td_expectile_loss(td_error, expectile):
    weight = jnp.where(td_error > 0.0, expectile, 1.0 - expectile)
    return weight * jnp.square(td_error)


def select_best(actions, scores):
    indices = jnp.argmax(scores, axis=-1)
    batch_shape = indices.shape
    flat_indices = indices.reshape(-1)
    flat_actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    selected = flat_actions[jnp.arange(flat_indices.size), flat_indices]
    return selected.reshape(batch_shape + (actions.shape[-1],))


class RFSActor(nn.Module):
    """One trunk, two heads: latent steering ``z`` and residual ``delta``.

    ``mode`` selects which head runs, because ``z`` must be decoded into a base
    action before the residual head can condition on it:

    * ``"latent"``   -> ``(mean, std)`` of the diagonal Gaussian over ``z``
    * ``"residual"`` -> raw residual output, conditioned on ``(obs, base)``
    * ``"both"``     -> both, used at init so every parameter is created

    The shared trunk is what makes the two heads co-adapt: a gradient on the
    executed action reaches ``z`` and ``delta`` through the same features.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    state_dependent_std: bool = False
    residual_fc_scale: float = 0.01
    residual_hidden_dims: Sequence[int] = ()
    condition_residual_on_base: bool = True
    # SAC-style: the residual head emits (mean, log_std) and is sampled with the
    # reparameterisation trick, so the *whole* policy (latent + residual) is
    # stochastic and can carry an entropy bonus.
    residual_stochastic: bool = False
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    def setup(self):
        self.trunk = MLP(
            tuple(self.hidden_dims), activate_final=True, layer_norm=self.layer_norm
        )
        self.z_mean = nn.Dense(self.action_dim, kernel_init=default_init())
        if self.state_dependent_std:
            self.z_log_std_head = nn.Dense(self.action_dim, kernel_init=default_init())
        else:
            self.z_log_std_param = self.param(
                "z_log_std", nn.initializers.zeros, (self.action_dim,), jnp.float32
            )
        residual_out = self.action_dim * (2 if self.residual_stochastic else 1)
        self.residual_mlp = MLP(
            (*tuple(self.residual_hidden_dims), residual_out),
            activate_final=False,
            layer_norm=self.layer_norm,
            kernel_init=default_init(self.residual_fc_scale),
        )

    def _latent(self, features):
        mean = self.z_mean(features)
        if self.state_dependent_std:
            log_std = self.z_log_std_head(features)
        else:
            log_std = jnp.broadcast_to(self.z_log_std_param, mean.shape)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, jnp.exp(log_std)

    def _residual(self, features, base):
        if self.condition_residual_on_base:
            inputs = jnp.concatenate([features, base], axis=-1)
        else:
            inputs = features
        out = self.residual_mlp(inputs)
        if not self.residual_stochastic:
            return out
        mean, log_std = jnp.split(out, 2, axis=-1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, jnp.exp(log_std)

    def __call__(self, observations, base=None, mode="both"):
        features = self.trunk(observations)
        if mode == "latent":
            return self._latent(features)
        if mode == "residual":
            return self._residual(features, base)
        return self._latent(features), self._residual(features, base)


class ANQRFSAgent(flax.struct.PyTreeNode):
    """Unified latent+residual steering of a behaviour-cloned drift policy."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------ utils

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

    def _compose(self, base, raw):
        """Combine the decoded base action with the actor's residual output."""
        mode = self.config["refine_output_mode"]
        base = self._safe_clip(base)
        if mode == "latent_only":
            # No residual at all: the executed action IS the decoded action, so
            # the policy can only be steered through z and can never leave the
            # manifold the drift decoder learned.  This is the DSRL structure.
            # Diagnosed need: on cube-double the residual drives ||a - a_data||^2
            # to 2.85 (vs 0.36 on antmaze) while lam=0.01 barely penalises it.
            return base, jnp.zeros_like(base)
        if mode == "absolute":
            # The head emits the executed action directly; ``base`` only
            # conditions it.  Additivity is not imposed.
            refined = self._safe_clip(jnp.tanh(raw))
        elif mode == "pretanh":
            bound = 1.0 - 1e-4
            pre_base = jnp.arctanh(jnp.clip(base, -bound, bound))
            refined = self._safe_clip(jnp.tanh(pre_base + raw))
        else:  # "action"
            refined = self._safe_clip(base + raw)
        return refined, refined - base

    # -------------------------------------------------------------- policy

    def _latent_dist(self, observations, params=None, actor_name="rfs_actor"):
        mean, std = self.network.select(actor_name)(
            observations, mode="latent", params=params
        )
        return mean * self.config["latent_noise_scale"], std * self.config[
            "latent_noise_scale"
        ]

    def _decode(self, observations, noises):
        """Decode ``z`` with the drift policy.

        ``params`` is deliberately not forwarded: the decoder's weights are
        frozen with respect to every actor objective (RFS keeps its pretrained
        flow policy fixed), but the gradient still reaches ``z`` through the
        decoder's *input*.

        With ``use_target_latent`` the EMA copy decodes instead (DSRL's
        ``use_target_latent=True``; ``stdfp.py`` has the same option).  The
        latent policy then optimises against a slowly-moving latent space
        rather than chasing a decoder that shifts every gradient step.
        """
        name = (
            "target_actor_drift"
            if self.config["use_target_latent"]
            else "actor_drift"
        )
        return self.network.select(name)(observations, noises)

    @staticmethod
    def _diag_gauss_log_prob(x, mean, std):
        return (
            -0.5 * jnp.square((x - mean) / std)
            - jnp.log(std)
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(axis=-1)

    def _act(self, observations, rng, params=None, actor_name="rfs_actor",
             deterministic=None):
        """Full policy: obs -> (z, delta) -> base -> executed action.

        Returns the executed action, the realised residual, and everything the
        losses need: the latent sample and its (mean, std), plus the summed
        log-probability of whatever was sampled (used by the SAC-style entropy
        regulariser).
        """
        mean, std = self._latent_dist(observations, params=params, actor_name=actor_name)
        deterministic = (
            self.config["latent_deterministic"] if deterministic is None else deterministic
        )
        z_rng, d_rng = (None, None) if rng is None else jax.random.split(rng)
        if deterministic or rng is None:
            pre = mean
        else:
            pre = mean + std * jax.random.normal(z_rng, mean.shape, dtype=mean.dtype)
        log_prob = self._diag_gauss_log_prob(pre, mean, std)
        if self.config["latent_squash_tanh"]:
            # DSRL-style bounded latent.  The drift decoder is BC'd on z ~ N(0,I),
            # so a tanh head confines z to (-1,1)^d and cannot reach most of that
            # prior's mass -- which is why the unsquashed head is the default
            # here.  It is exposed because DSRL reports strong cube results with
            # a TanhNormal latent at target_multiplier=0.5.
            squashed = jnp.tanh(pre)
            log_prob = log_prob - jnp.log1p(-jnp.square(squashed) + 1e-6).sum(axis=-1)
            noises = squashed
        else:
            noises = pre

        base = self._decode(observations, noises)
        raw = self.network.select(actor_name)(
            observations, base=jax.lax.stop_gradient(base)
            if self.config["residual_sees_stopped_base"] else base,
            mode="residual", params=params,
        )
        # Curriculum: keep the policy purely in-support (delta = 0) until
        # refine_start_step, then let the refine head engage.  Mirror of the
        # decoder freeze at the other end of training.
        start = self.config.get("refine_start_step", 0)
        if start and not self.config["residual_stochastic"]:
            gate = (self.network.step >= start).astype(base.dtype)
            raw = raw * gate
        if self.config["residual_stochastic"]:
            r_mean, r_std = raw
            if deterministic or rng is None:
                raw = r_mean
            else:
                raw = r_mean + r_std * jax.random.normal(
                    d_rng, r_mean.shape, dtype=r_mean.dtype
                )
            log_prob = log_prob + self._diag_gauss_log_prob(raw, r_mean, r_std)
        refined, delta = self._compose(base, raw)
        return refined, delta, noises, mean, std, log_prob

    # --------------------------------------------------------------- losses

    def critic_loss(self, batch, grad_params, rng):
        next_observations = batch["next_observations"][..., -1, :]
        if self.config.get("critic_target_actions", "refined") == "base":
            # DSRL-style bootstrap: the TD target evaluates the in-support base
            # policy (latent + decoder), so a bad refine head cannot poison the
            # critic targets.  Execution still uses the full refined policy.
            refined, delta, noises, mean, std, log_prob = self._act(
                next_observations, rng, actor_name="target_rfs_actor",
            )
            next_actions = self._safe_clip(refined - delta)
        else:
            next_actions, _ = self._sample_actions_bestof(
                next_observations, rng, actor_name="target_rfs_actor",
                critic_name="target_critic",
            )
        next_qs = self.network.select("target_critic")(
            next_observations, actions=next_actions
        )
        next_q = self._aggregate_q(next_qs)
        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        qs = self.network.select("critic")(
            batch["observations"], actions=self._batch_actions(batch), params=grad_params
        )
        losses = td_expectile_loss(
            target_q[None, ...] - qs, self.config["critic_expectile"]
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
        noises = jax.random.normal(
            rng, (batch_size * self.config["gen_per_label"], action_dim)
        )
        generated = self.network.select("actor_drift")(
            observations, noises, params=grad_params
        ).reshape(batch_size, self.config["gen_per_label"], action_dim)
        # drift_multi_temp: multi-scale kernel around the base temperature
        # (drift_loss was designed for an R_list; we have only ever passed one).
        _R = ((self.config["drift_temps"] * 0.25, self.config["drift_temps"],
               self.config["drift_temps"] * 4.0)
              if self.config.get("drift_multi_temp", False)
              else (self.config["drift_temps"],))
        losses, drift_info = drift_loss(
            gen=generated,
            fixed_pos=actions[..., None, :],
            R_list=_R,
            force_norm=self.config["drift_force_norm"] if "drift_force_norm" in self.config else "unit",
            force_scale_const=self.config["drift_force_scale"] if "drift_force_scale" in self.config else 0.0,
        )
        loss = losses.mean()
        # Q-guided drift (Q-Flow-style guidance transplanted onto the SVGD
        # decoder): regress the generated samples one beta-scaled unit step
        # along the critic's action-gradient.  The target is stop-gradded, so
        # this adds a bounded steering force rather than raw Q-ascent; beta is
        # comparable to the unit-RMS drift-BC step (beta ~ 1/lambda_QFlow).
        qg = self.config.get("drift_q_guidance", 0.0)
        if qg:
            # Warmup gate: a cold critic's gradients destroy the decoder
            # (measured: all always-on guidance arms flat 0.00).  Engage
            # guidance only once the critic has trained for qg_start steps.
            qg_start = self.config.get("drift_qg_start", 0)
            if qg_start:
                qg = qg * (self.network.step >= qg_start).astype(jnp.float32)
            flat_gen = generated.reshape(-1, generated.shape[-1])

            def _qsum(a):
                qs = self.network.select("critic")(observations, actions=a)
                return self._aggregate_q(qs).sum()

            g = jax.lax.stop_gradient(jax.grad(_qsum)(flat_gen))
            g = g / jnp.clip(
                jnp.sqrt(jnp.square(g).mean(axis=-1, keepdims=True)), 1e-6
            )
            guide_goal = jax.lax.stop_gradient(flat_gen + qg * g)
            loss = loss + jnp.square(flat_gen - guide_goal).mean()
        info = {
            "actor_drift_loss": loss,
            "drift_scale": drift_info.get("scale", 0.0),
            "generated_to_data_mse": jnp.mean(
                jnp.square(generated - actions[..., None, :])
            ),
        }
        return loss, info

    def _sigreg_loss(self, key, x):
        """Match the *batch marginal* of ``z`` to N(0, I) by ECF regression.

        Ported from ``agents/stdfp.py::_sigreg_strong_loss``, where it plays the
        same role: in the DDPG (deterministic) branch there is no per-sample
        latent distribution to take a KL against, so the constraint is applied
        to the empirical distribution of ``z`` across the batch instead.  Random
        spherical projections are compared against the unit-Gaussian
        characteristic function exp(-t^2/2) on a grid, integrated by trapezoid.
        """
        eps = 1e-6
        _, c = x.shape
        a = jax.random.normal(key, (c, self.config["sigreg_sketch_dim"]))
        a = a / (jnp.linalg.norm(a, axis=0, keepdims=True) + eps)
        t = jnp.linspace(
            self.config["sigreg_t_min"],
            self.config["sigreg_t_max"],
            self.config["sigreg_num_t"],
        )
        target_cf = jnp.exp(-0.5 * jnp.square(t))
        proj = x @ a
        args = proj[:, :, None] * t[None, None, :]
        empirical_cf = jnp.mean(jnp.exp(1j * args), axis=0)
        err = (jnp.abs(empirical_cf - target_cf[None, :]) ** 2) * target_cf[None, :]
        dt = t[1:] - t[:-1]
        trap = 0.5 * (err[:, 1:] + err[:, :-1]) * dt[None, :]
        return jnp.mean(jnp.sum(trap, axis=1) * x.shape[0])

    def _latent_kl(self, mean, std):
        """Closed-form KL( N(mean, std) || N(0, I) ), summed over dims."""
        return 0.5 * (
            jnp.square(std) + jnp.square(mean) - 1.0 - 2.0 * jnp.log(std)
        ).sum(axis=-1)

    def unified_actor_loss(self, batch, grad_params, rng):
        """The single joint objective over ``(z, delta)``."""
        observations = batch["observations"]
        # Train with a reparameterised sample so the latent *scale* receives the
        # Q gradient too; execution can still use the mode (latent_deterministic).
        # With a deterministic latent in this loss, ``std`` would be driven only
        # by the KL term and the Q objective could never widen or narrow it.
        refined, delta, noises, mean, std, log_prob = self._act(
            observations, rng, params=grad_params,
            deterministic=not self.config["train_latent_stochastic"],
        )
        qs = self.network.select("critic")(observations, actions=refined)
        q = self._aggregate_q(qs, mode=self.config["refine_q_agg"])
        norm_q = jax.lax.stop_gradient(1.0 / jnp.abs(q).mean())

        data_actions = self._batch_actions(batch)
        valid = batch["valid"][..., -1]
        if self.config["bc_anchor"] == "data":
            # ReBRAC-style: the executed action is pulled toward the dataset action.
            offset = refined - data_actions
        elif self.config["bc_anchor"] == "residual":
            # RFS Eq. 11: penalise only the residual, leaving the latent free.
            offset = delta
        else:  # "none"
            offset = jnp.zeros_like(refined)
        bc = ((offset ** 2).sum(axis=-1) * valid).mean()

        if self.config["latent_squash_tanh"]:
            # No closed form through the tanh; use the one-sample estimator
            # KL ~= log q(z) - log p(z), matching agents/anq_stdfp.py.
            log_prior = -0.5 * (
                jnp.square(noises) + math.log(2.0 * math.pi)
            ).sum(axis=-1)
            kl = log_prob - log_prior
        else:
            kl = self._latent_kl(mean, std)
        reg = self.config["latent_reg"]
        if reg == "kl":
            alpha = self.network.select("noise_alpha")()
            train_alpha = self.network.select("noise_alpha")(params=grad_params)
            reg_term = alpha * kl.mean()
            # Dual ascent on a KL budget: push alpha up while KL exceeds target.
            alpha_loss = (
                train_alpha
                * (self.config["noise_target_kl"] - jax.lax.stop_gradient(kl.mean()))
            )
        elif reg == "entropy":
            # SAC: maximise Q + alpha * H(pi), with alpha auto-tuned so the
            # policy entropy tracks ``noise_target_entropy``.  The Q term is
            # scale-normalised (norm_q) so ``lam`` stays comparable across
            # regularisers; alpha adapts to whatever scale that leaves.
            alpha = self.network.select("noise_alpha")()
            train_alpha = self.network.select("noise_alpha")(params=grad_params)
            reg_term = alpha * log_prob.mean()
            alpha_loss = -(
                train_alpha
                * (
                    jax.lax.stop_gradient(log_prob.mean())
                    + self.config["noise_target_entropy"]
                )
            )
        elif reg == "sigreg":
            # Population-level constraint, no dual: works with a deterministic
            # latent head, unlike the per-sample KL / entropy terms.
            sig_rng = jax.random.fold_in(rng, 7)
            reg_term = self.config["sigreg_coeff"] * self._sigreg_loss(sig_rng, noises)
            alpha_loss = 0.0
            train_alpha = jnp.asarray(0.0)
        else:
            reg_term = 0.0
            alpha_loss = 0.0
            train_alpha = jnp.asarray(0.0)

        loss = -norm_q * q.mean() + self.config["lam"] * bc + reg_term + alpha_loss
        return loss, {
            "actor_loss": loss,
            "actor_q": q.mean(),
            "bc_offset": bc,
            "delta_rms": jnp.sqrt(jnp.mean(jnp.square(delta))),
            "action_to_data_rms": jnp.sqrt(
                jnp.mean(jnp.square(refined - data_actions))
            ),
            "base_to_data_rms": jnp.sqrt(
                jnp.mean(jnp.square(refined - delta - data_actions))
            ),
            "latent_kl": kl.mean(),
            "log_prob": log_prob.mean(),
            "entropy": -log_prob.mean(),
            "latent_mean_abs": jnp.abs(mean).mean(),
            "latent_std": std.mean(),
            "alpha": train_alpha,
        }

    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        critic_rng, bc_rng, actor_rng = jax.random.split(rng, 3)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        bc_loss, bc_info = self.drift_bc_loss(batch, grad_params, bc_rng)
        # Drift-BC hard stop (see anq_stdfp.py): the decoder is trained only by
        # this loss, so zeroing it after bc_stop_step freezes the decoder.
        bc_stop = self.config["bc_stop_step"] if "bc_stop_step" in self.config else 0
        if bc_stop:
            bc_on = (self.network.step < bc_stop).astype(bc_loss.dtype)
            bc_loss = bc_loss * bc_on
        actor_loss, actor_info = self.unified_actor_loss(batch, grad_params, actor_rng)
        loss = critic_loss + self.config["bc_coef"] * bc_loss + actor_loss
        info = {"total_loss": loss}
        info.update({f"critic/{k}": v for k, v in critic_info.items()})
        info.update({f"actor/{k}": v for k, v in bc_info.items()})
        info.update({f"actor/{k}": v for k, v in actor_info.items()})
        return loss, info

    # --------------------------------------------------------------- updates

    def _target_update(self, network, module_name):
        source = network.params[f"modules_{module_name}"]
        target = self.network.params[f"modules_target_{module_name}"]
        network.params[f"modules_target_{module_name}"] = jax.tree_util.tree_map(
            lambda p, tp: self.config["tau"] * p + (1.0 - self.config["tau"]) * tp,
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
        agent._target_update(new_network, "rfs_actor")
        if agent.config["use_target_latent"]:
            agent._target_update(new_network, "actor_drift")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # ------------------------------------------------------------- inference

    def _sample_actions_bestof(
        self, observations, rng, actor_name="rfs_actor", critic_name="critic"
    ):
        n = self.config["best_of_n"]
        rng = jax.random.PRNGKey(0) if rng is None else rng
        if n > 1:
            observations = jnp.repeat(observations[..., None, :], n, axis=-2)
        refined, delta, _, _, _, _ = self._act(
            observations, rng, actor_name=actor_name,
            # best-of-n over identical deterministic latents would be a no-op
            deterministic=self.config["latent_deterministic"] and n == 1,
        )
        if n == 1:
            return refined, delta
        scores = self._aggregate_q(
            self.network.select(critic_name)(observations, actions=refined),
            mode=self.config["bfn_q_agg"],
        )
        return select_best(refined, scores), select_best(delta, scores)

    @partial(jax.jit, static_argnames=("critic_name",))
    def sample_actions(self, observations, rng=None, critic_name="critic"):
        # eval_use_target: execute the Polyak-averaged (target) actor instead
        # of the live one.  Retention aid: measured t5 arms repeatedly FIND the
        # solution then lose it to policy oscillation; the EMA smooths that.
        actor = ("target_rfs_actor"
                 if self.config.get("eval_use_target", False) else "rfs_actor")
        actions, _ = self._sample_actions_bestof(
            observations, rng, actor_name=actor, critic_name=critic_name
        )
        return self._safe_clip(actions)

    # ---------------------------------------------------------------- create

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
        if config["noise_target_kl"] is None:
            config["noise_target_kl"] = config["target_multiplier"] * full_action_dim
        if config["noise_target_entropy"] is None:
            n_sampled = full_action_dim * (2 if config["residual_stochastic"] else 1)
            config["noise_target_entropy"] = -config["entropy_scale"] * n_sampled

        encoders = {}
        if config["encoder"] is not None:
            encoder = encoder_modules[config["encoder"]]
            encoders = {"critic": encoder(), "actor_drift": encoder()}

        critic = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
        )
        actor_drift = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_drift"),
        )
        rfs_actor = RFSActor(
            hidden_dims=config["refine_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["refine_layer_norm"],
            state_dependent_std=config["noise_state_dependent_std"],
            residual_fc_scale=config["refine_fc_scale"],
            residual_hidden_dims=config["residual_head_hidden_dims"],
            condition_residual_on_base=config["condition_residual_on_base"],
            residual_stochastic=config["residual_stochastic"],
        )

        actor_init = dict(
            observations=ex_observations, base=full_actions, mode="both"
        )
        definitions = {
            "critic": (critic, (ex_observations, full_actions)),
            "target_critic": (copy.deepcopy(critic), (ex_observations, full_actions)),
            "actor_drift": (actor_drift, (ex_observations, full_actions)),
            "target_actor_drift": (
                copy.deepcopy(actor_drift),
                (ex_observations, full_actions),
            ),
            "rfs_actor": (rfs_actor, actor_init),
            "target_rfs_actor": (copy.deepcopy(rfs_actor), actor_init),
            "noise_alpha": (LogParam(init_value=config["noise_init_temp"]), ()),
        }
        network_def = ModuleDict({k: v[0] for k, v in definitions.items()})
        params = network_def.init(
            init_rng, **{k: v[1] for k, v in definitions.items()}
        )["params"]
        params["modules_target_critic"] = params["modules_critic"]
        params["modules_target_rfs_actor"] = params["modules_rfs_actor"]
        params["modules_target_actor_drift"] = params["modules_actor_drift"]
        network = TrainState.create(
            network_def,
            params,
            tx=make_module_optimizer(
                config["optimizer"], config["lr"],
                eps=config["adam_eps"] if "adam_eps" in config else 1e-8,
                module_eps=(
                    {"actor_drift": config["drift_adam_eps"]}
                    if "drift_adam_eps" in config and config["drift_adam_eps"]
                    else None
                ),
            ),
        )
        config["ob_dims"] = ex_observations.shape
        config["action_dim"] = action_dim
        config["full_action_dim"] = full_action_dim
        return cls(rng, network, flax.core.FrozenDict(**config))

    @staticmethod
    def _validate_config(config):
        q_modes = ("min", "mean", "pessimistic")
        for key in ("q_agg", "refine_q_agg", "bfn_q_agg"):
            if config[key] not in q_modes:
                raise ValueError(f"{key} must be one of {q_modes}")
        if not 0.0 < config["critic_expectile"] < 1.0:
            raise ValueError("critic_expectile must be in (0, 1)")
        if config["refine_output_mode"] not in (
            "pretanh", "action", "absolute", "latent_only"
        ):
            raise ValueError(
                "refine_output_mode must be 'pretanh', 'action', 'absolute' "
                "or 'latent_only'"
            )
        if config["bc_anchor"] not in ("data", "residual", "none"):
            raise ValueError("bc_anchor must be 'data', 'residual' or 'none'")
        if config["latent_reg"] not in ("kl", "entropy", "sigreg", "none"):
            raise ValueError(
                "latent_reg must be 'kl', 'entropy', 'sigreg' or 'none'"
            )
        if config["latent_reg"] == "entropy" and not config["train_latent_stochastic"]:
            raise ValueError(
                "latent_reg='entropy' needs train_latent_stochastic=True: a "
                "deterministic policy has no entropy to regularise"
            )
        if config["num_qs"] < 1 or config["best_of_n"] < 1:
            raise ValueError("num_qs and best_of_n must be positive")
        if config["gen_per_label"] < 2:
            raise ValueError("gen_per_label must be at least 2")
        if config["latent_noise_scale"] <= 0.0:
            raise ValueError("latent_noise_scale must be positive")


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="anq_rfs",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            full_action_dim=ml_collections.config_dict.placeholder(int),
            optimizer="adam",
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            refine_hidden_dims=(512, 512, 512, 512),
            # Extra layers on the residual head after the shared trunk.  Empty
            # means a single linear map off the trunk features (+ base).
            residual_head_hidden_dims=(),
            actor_layer_norm=False,
            layer_norm=True,
            refine_layer_norm=False,
            refine_fc_scale=0.01,
            discount=0.99,
            tau=0.005,
            num_qs=2,
            rho=0.5,
            q_agg="min",
            critic_target_actions="refined",  # "base" = DSRL-style in-support TD targets
            refine_q_agg="mean",
            bfn_q_agg="mean",
            critic_expectile=0.5,
            # How the residual head's output becomes the executed action.
            #   "pretanh":  a = tanh(atanh(base) + raw)   (differentiable everywhere)
            #   "action":   a = clip(base + raw)
            #   "absolute": a = tanh(raw), with base only as a conditioning input
            refine_output_mode="pretanh",
            # Whether the residual head sees a stop_gradient'd base.  False lets
            # the Q gradient reach z through *both* the decoder input and the
            # residual head's conditioning input.
            residual_sees_stopped_base=True,
            # Does the residual head get the decoded base as an input at all?
            condition_residual_on_base=True,
            # What the quadratic behaviour penalty is measured against.
            #   "data":     ||a - a_data||^2   (ReBRAC-style, validated at 0.01)
            #   "residual": ||delta||^2        (RFS Eq. 11)
            bc_anchor="data",
            lam=0.01,
            # Weight on the drift behaviour-cloning loss.
            bc_coef=1.0,
            bc_stop_step=0,
            refine_start_step=0,  # >0: refine head engages only after this step
            drift_q_guidance=0.0,  # >0: Q-Flow-style guidance force on the drift decoder
            drift_qg_start=0,  # >0: engage guidance only after this step
            eval_use_target=False,  # execute the EMA (target) actor at eval
            drift_multi_temp=False,  # True: 3-scale kernel (0.25x, 1x, 4x drift_temps)
            # "kl":      alpha * KL(z || N(0,I)), dual on noise_target_kl
            # "entropy": SAC -- alpha * log pi, dual on noise_target_entropy
            # "none":    unregularised
            # "sigreg": ECF match of the batch marginal of z to N(0,I) --
            #   the DDPG-compatible regulariser from agents/stdfp.py, usable
            #   with train_latent_stochastic=False where KL/entropy degenerate.
            latent_reg="kl",
            sigreg_coeff=1.0,
            sigreg_sketch_dim=64,
            sigreg_num_t=17,
            sigreg_t_min=-5.0,
            sigreg_t_max=5.0,
            # SAC-style stochastic residual head (mean, log_std) on top of the
            # stochastic latent, so the entropy bonus covers the whole policy.
            residual_stochastic=False,
            noise_target_entropy=ml_collections.config_dict.placeholder(float),
            # target_entropy = -entropy_scale * (#sampled dims)
            entropy_scale=1.0,
            # Squash the latent through tanh (DSRL-style bounded latent space).
            # False = unsquashed Gaussian, the anq_stdfp/anq_rfs default.
            latent_squash_tanh=False,
            # Decode z through the EMA'd drift decoder (DSRL use_target_latent):
            # a slowly-moving latent space instead of one that shifts every step.
            use_target_latent=False,
            latent_deterministic=True,   # execution uses the mode
            train_latent_stochastic=True,  # actor loss uses a reparameterised sample
            latent_noise_scale=1.0,
            noise_state_dependent_std=False,
            noise_target_kl=ml_collections.config_dict.placeholder(float),
            target_multiplier=0.125,
            noise_init_temp=1.0,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            best_of_n=1,
            gen_per_label=8,
            drift_temps=0.1,
            adam_eps=1e-8,
            drift_adam_eps=ml_collections.config_dict.placeholder(float),
            drift_force_norm="unit",
            drift_force_scale=0.0,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
