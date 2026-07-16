
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections
import optax
from functools import partial

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, Actor
from utils.drift_loss import drift_loss


def iql_loss(adv, expectile):
    """Compute the IQL loss."""
    # adv = jnp.minimum(adv, 5.0)  # clip to prevent from gradeint exploding
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (adv**2)


def sql_loss(x, expectile):
    """Compute the SQL loss."""
    # x = jnp.minimum(x, 5.0)  # clip to prevent from gradeint exploding
    sp_term = x / (2 * expectile) + 1.0
    sp_weight = jnp.where(sp_term > 0, 1., 0.)
    return expectile * sp_weight * (sp_term**2) - x


def target_update(model, target_model, tau):
    """Update the target network."""
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def pairwise_squared_distances(X, Y):
    """Pairwise squared distances for [B, n, d] and [B, m, d]."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    X2 = jnp.sum(X * X, axis=-1, keepdims=True)                                 # [B, n, 1]
    Y2 = jnp.sum(Y * Y, axis=-1, keepdims=True).transpose(0, 2, 1)              # [B, 1, m]
    XY = jnp.matmul(X, Y.transpose(0, 2, 1))                                    # [B, n, m]
    dnorm2 = X2 + Y2 - 2.0 * XY                                                 # [B, n, m]
    return jnp.maximum(dnorm2, 0.0)


def median_heuristic_sigma(dnorm2, particle_count):
    # Median heuristic per batch.
    h = jnp.median(dnorm2, axis=(1, 2)) / (2.0 * jnp.log(particle_count + 1.0))  # [B]
    sigma_val = jnp.sqrt(jnp.maximum(h, 1e-12))                                  # [B]
    return sigma_val[:, None, None]


def rbf_kernel(X, Y, sigma=None):
    """
    X: [B, n, d], Y: [B, m, d]
    returns K_XY: [B, n, m] with RBF kernel entries
    """
    dnorm2 = pairwise_squared_distances(X, Y)

    if sigma is None:
        sigma_val = median_heuristic_sigma(dnorm2, X.shape[1])
    else:
        sigma_val = jnp.asarray(sigma)
        if sigma_val.ndim == 0:
            sigma_val = jnp.broadcast_to(sigma_val, (X.shape[0], 1, 1))

    gamma = 1.0 / (1e-6 + 2.0 * (sigma_val ** 2))
    K_XY = jnp.exp(-gamma * dnorm2)                                             # [B, n, m]
    return K_XY, dnorm2, gamma*2


class SVGD_VGF:
    def __init__(
        self,
        q,
        q_agg,
        optimizer: optax.GradientTransformation,
        svgd_type='naive',
        sinkhorn_tau=1.0,
    ):
        """
        q: state-action value function
        optimizer: an optax optimizer, e.g. optax.adam(1e-2)
        """
        self.q = q
        self.q_agg = q_agg
        self.optim = optimizer
        self.opt_state = None
        self.svgd_type = svgd_type
        self.sinkhorn_tau = sinkhorn_tau

    def init(self, particles):
        """Initialize optimizer state for the particle array X."""
        self.opt_state = self.optim.init(particles)
        return particles, self.opt_state

    def phi(self, obs, particles, anchors=None):
        # obs: [B, D], particles: [B, N, D]
        if self.svgd_type not in ('naive', 'sinkhorn'):
            raise ValueError(f"Unknown svgd_type: {self.svgd_type}")
        anchors = particles if anchors is None else anchors
        anchors = jax.lax.stop_gradient(anchors)

        # Score terms
        def sum_target(action):
            obs_flatten = obs.reshape(-1, obs.shape[-1])                        # [B*N, D]
            action_flatten = action.reshape(-1, action.shape[-1])               # [B*N, D]
            qs = self.q(obs_flatten, action_flatten)
            q = jnp.min(qs, axis=0) if self.q_agg == 'min' else jnp.mean(qs, axis=0)
            q = q.reshape(action.shape[:-1])
            # q normalization to stabilize gradient (doesn't work)
            # q /= jax.lax.stop_gradient(jnp.mean(jnp.abs(q)))
            if self.svgd_type == 'sinkhorn':
                cost = 0.5 * jnp.sum(jnp.square(action - anchors), axis=-1)
                use_median_tau = self.sinkhorn_tau is None or (
                    isinstance(self.sinkhorn_tau, str)
                    and self.sinkhorn_tau.lower() == 'none'
                )
                if use_median_tau:
                    action_stop = jax.lax.stop_gradient(action)
                    dnorm2 = pairwise_squared_distances(action_stop, action_stop)
                    tau = median_heuristic_sigma(dnorm2, action.shape[1]).squeeze(-1)
                else:
                    tau = jnp.asarray(self.sinkhorn_tau)
                q = q - cost / jnp.maximum(tau, 1e-12)
            return jnp.sum(q)
        score = jax.grad(sum_target)(particles)                                 # [B, N, D]

        # Kernel terms
        particles_stop = jax.lax.stop_gradient(particles)
        K_xx, K_dist, K_gamma = rbf_kernel(particles, particles_stop)                            # [B, N, N]
        K_xx = jax.lax.stop_gradient(K_xx)

        # grad_K := -∂/∂X sum_{i,j} K(X_i, X_j_stop)
        def sum_K(x):
            return jnp.sum(rbf_kernel(x, particles_stop)[0])
        grad_K = -jax.grad(sum_K)(particles)                                    # [B, N, D]

        # φ(X) = (K_xx * score + grad_K) / N
        phi_val = (K_xx @ score + grad_K) / particles.shape[1]
        return phi_val      

    def step(self, obs, particles, opt_state, anchors=None):
        """In Optax, we pass "grads" = -phi(X), which yields X ← X + lr * phi(X)"""
        phi_val = self.phi(obs, particles, anchors)
        grads = -phi_val

        updates, new_opt_state = self.optim.update(grads, opt_state, params=particles)
        new_particles = optax.apply_updates(particles, updates)
        return new_particles, new_opt_state


class VGFAgent(flax.struct.PyTreeNode):
    """Value Gradient Flow (VGF) agent."""

    rng: Any
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor_bc: TrainState
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config['action_chunking']:
            return jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        return batch['actions'][..., 0, :]

    def _next_observations(self, batch):
        return batch['next_observations'][..., -1, :]

    def _discounted_rewards(self, batch):
        return batch['rewards'][..., -1]

    def _discounted_masks(self, batch):
        return batch['masks'][..., -1]

    def _valid(self, batch, target):
        if 'valid' in batch:
            return batch['valid'][..., -1]
        return jnp.ones_like(target)

    def _actor_action_dim(self):
        horizon = self.config['horizon_length'] if self.config['action_chunking'] else 1
        return self.config['action_dim'] * horizon

    def _masked_action_mse(self, squared_error, batch):
        if self.config['action_chunking']:
            squared_error = jnp.reshape(
                squared_error,
                (
                    squared_error.shape[0],
                    self.config['horizon_length'],
                    self.config['action_dim'],
                ),
            )
            if 'valid' in batch:
                squared_error = squared_error * batch['valid'][..., None]
        return jnp.mean(squared_error)

    def _bc_update_active(self):
        freeze_after = self.config['freeze_actor_bc_after']
        return jnp.logical_or(freeze_after < 0, self.actor_bc.step <= freeze_after)

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, _ = jax.random.split(self.rng)
        def critic_loss_fn(critic_params):
            batch_actions = self._batch_actions(batch)
            batch_size, action_dim = batch_actions.shape
            _, noise_rng = jax.random.split(self.rng)

            # compute bc actions
            next_obs = self._next_observations(batch)
            next_obs_rep = jnp.repeat(jnp.expand_dims(next_obs, 1), self.config['vgf_particles'], axis=1)
            next_obs_rep = next_obs_rep.reshape(-1, next_obs_rep.shape[-1])
            bc_next_actions = self.sample_bc_actions(next_obs_rep, noise_rng)
            bc_next_actions = bc_next_actions.reshape(batch_size, self.config['vgf_particles'], -1)
            next_obs_rep = next_obs_rep.reshape(batch_size, self.config['vgf_particles'], -1)

            # value gradient flow
            svgd = SVGD_VGF(
                self.critic,
                self.config['vgf_q_agg'],
                optax.adam(learning_rate=self.config['vgf_lr']),
                svgd_type=self.config['vgf_svgd_type'],
                sinkhorn_tau=self.config['vgf_sinkhorn_tau'],
            )
            particles, opt_state = svgd.init(bc_next_actions)
            anchors = jax.lax.stop_gradient(bc_next_actions)
            # phi_q_list, phi_k_list = [], []

            for _ in range(self.config['train_vgf_steps']):
                particles, new_opt_state = svgd.step(next_obs_rep, particles, opt_state, anchors)
                particles = jnp.clip(particles, -1, 1)
                # phi_q_list.append(phi_q)
                # phi_k_list.append(phi_k)
                opt_state = new_opt_state

            next_obs_flatten = next_obs_rep.reshape(-1, next_obs_rep.shape[-1])
            next_actions_flatten = particles.reshape(-1, particles.shape[-1])
            next_qs = self.target_critic(next_obs_flatten, actions=next_actions_flatten)      # [2, B*N]
            if self.config['train_q_agg'] == 'min':
                next_q = next_qs.min(axis=0)
            else:
                next_q = next_qs.mean(axis=0)
            
            # particle selection
            next_q = next_q.reshape(-1, self.config['vgf_particles'])  
            if self.config['train_particle_select'] == 'max':
                next_q = jnp.max(next_q, axis=1)    # shape (B,)
            else:
                next_q = jnp.mean(next_q, axis=1)   # shape (B,)
            
            target_q = self._discounted_rewards(batch) + (
                self.config['discount'] ** self.config['horizon_length']
            ) * self._discounted_masks(batch) * next_q
            q1, q2 = self.critic(batch['observations'], actions=batch_actions, params=critic_params)

            if self.config['critic_loss']  == 'td':  # TD update
                critic_loss = (
                    ((target_q - q1) ** 2 + (target_q - q2) ** 2)
                    * self._valid(batch, target_q)
                ).mean()
            elif self.config['critic_loss'] == 'sql-q':
                critic_loss = (
                    (
                        sql_loss(target_q - q1, self.config['expectile'])
                        + sql_loss(target_q - q2, self.config['expectile'])
                    )
                    * self._valid(batch, target_q)
                ).mean()
            elif self.config['critic_loss'] == 'iql-q':
                critic_loss = (
                    (
                        iql_loss(target_q - q1, self.config['expectile'])
                        + iql_loss(target_q - q2, self.config['expectile'])
                    )
                    * self._valid(batch, target_q)
                ).mean()

            q_info = {
                'critic_loss': critic_loss,
                'q_mean': q1.mean(),
            }
            # if self.config['critic_loss']  == 'td':
            # q_info.update({f"phi_q_{i}": v.mean() for i, v in enumerate(phi_q_list)})
            # q_info.update({f"phi_k_{i}": v.mean() for i, v in enumerate(phi_k_list)})
            q_info['bc_flow_var'] = jnp.var(bc_next_actions, axis=1, ddof=1).mean()
            q_info['vgf_var'] = jnp.var(particles, axis=1, ddof=1).mean()
            # 'actor_actions_abs_mean': jnp.abs(actor_actions).mean(),
            # 'dataset_actions_abs_mean': jnp.abs(batch['actions']).mean(),
            # 'mse': jnp.mean((actor_actions - batch['actions']) ** 2),

            return critic_loss, q_info

        def actor_bc_loss_fn(actor_bc_params):
            """Behavior cloning loss for the selected BC actor."""
            batch_actions = self._batch_actions(batch)
            batch_size, action_dim = batch_actions.shape

            if self.config['bc_policy_type'] == 'flow': 
                _, x_rng, t_rng = jax.random.split(self.rng, 3)

                # BC flow loss.
                x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
                x_1 = batch_actions
                t = jax.random.uniform(t_rng, (batch_size, 1))
                x_t = (1 - t) * x_0 + t * x_1
                vel = x_1 - x_0

                pred = self.actor_bc(batch['observations'], x_t, t, params=actor_bc_params)
                bc_loss = self._masked_action_mse((pred - vel) ** 2, batch)
                bc_info = {}
            elif self.config['bc_policy_type'] == 'drift':
                _, drift_rng = jax.random.split(self.rng)
                gen_per_label = self.config['gen_per_label']
                obs_repeated = jnp.repeat(batch['observations'], gen_per_label, axis=0)
                drift_noises = jax.random.normal(
                    drift_rng,
                    (batch_size * gen_per_label, action_dim),
                )
                drift_actions = self.actor_bc(
                    obs_repeated,
                    drift_noises,
                    params=actor_bc_params,
                )
                gen_samples = drift_actions.reshape(batch_size, gen_per_label, action_dim)
                pos_samples = jnp.expand_dims(batch_actions, axis=1)
                bc_loss, drift_info = drift_loss(
                    gen=gen_samples,
                    fixed_pos=pos_samples,
                    R_list=tuple(self.config['drift_temps']),
                )
                bc_loss = bc_loss.mean()
                bc_info = {f'bc_drift_{key}': val for key, val in drift_info.items()}
            elif self.config['bc_policy_type'] == 'gau': 
                dist = self.actor_bc(batch['observations'], params=actor_bc_params)
                log_prob = dist.log_prob(batch_actions)
                bc_loss = -jnp.mean(log_prob)
                bc_info = {}
            else:
                raise ValueError(f"Unknown bc_policy_type: {self.config['bc_policy_type']}")

            bc_loss_raw = bc_loss
            bc_update_active = self._bc_update_active()
            bc_loss = jnp.where(
                bc_update_active,
                bc_loss_raw,
                jnp.zeros_like(bc_loss_raw),
            )

            return bc_loss, {
                'bc_loss': bc_loss,
                'bc_loss_raw': bc_loss_raw,
                'bc_update_active': bc_update_active.astype(jnp.float32),
                **bc_info,
            }

        new_value, value_info = self.value, {}  
        new_critic, critic_info = self.critic.apply_loss_fn(loss_fn=critic_loss_fn)
        new_target_critic = target_update(self.critic, self.target_critic, self.config['tau'])
        new_actor_bc, actor_bc_info = self.actor_bc.apply_loss_fn(loss_fn=actor_bc_loss_fn)
        bc_update_active = self._bc_update_active()
        new_actor_bc = new_actor_bc.replace(
            params=jax.tree_util.tree_map(
                lambda new, old: jnp.where(bc_update_active, new, old),
                new_actor_bc.params,
                self.actor_bc.params,
            )
        )

        return self.replace(rng=new_rng, critic=new_critic, target_critic=new_target_critic, value=new_value, actor_bc=new_actor_bc), {
            **value_info, **critic_info, **actor_bc_info
        }

    @partial(jax.jit, static_argnames=("eval_vgf_steps",))
    def sample_actions(
        self,
        observations,
        rng=None,
        seed=None,
        eval_vgf_steps=0,
    ):
        """(Evaluation) Sample actions via value gradient flow."""
        seed = rng if seed is None else seed
        seed = jax.random.PRNGKey(0) if seed is None else seed
        obs = jnp.expand_dims(observations, 0)
        obs_rep = jnp.repeat(jnp.expand_dims(obs, 1), self.config['vgf_particles'], axis=1)
        obs_rep = obs_rep.reshape(-1, obs_rep.shape[-1])

        # generate bc flow actions: [1, particle_num, action_dim]
        _, noise_seed = jax.random.split(seed)
        bc_actions = self.sample_bc_actions(obs_rep, noise_seed)
        bc_actions = bc_actions.reshape(-1, self.config['vgf_particles'], bc_actions.shape[-1])
        obs_rep = obs_rep.reshape(-1, self.config['vgf_particles'], obs_rep.shape[-1])

        # value gradient flow
        svgd = SVGD_VGF(
            self.critic,
            self.config['vgf_q_agg'],
            optax.adam(learning_rate=self.config['vgf_lr']),
            svgd_type=self.config['vgf_svgd_type'],
            sinkhorn_tau=self.config['vgf_sinkhorn_tau'],
        )
        particles, opt_state = svgd.init(bc_actions)
        anchors = jax.lax.stop_gradient(bc_actions)

        for _ in range(eval_vgf_steps):
            particles, new_opt_state = svgd.step(obs_rep, particles, opt_state, anchors)
            particles = jnp.clip(particles, -1, 1)
            opt_state = new_opt_state

        obs_flatten = obs_rep.reshape(-1, obs_rep.shape[-1])
        particles_flatten = particles.reshape(-1, particles.shape[-1])
        qs = self.critic(obs_flatten, particles_flatten)
        if self.config['train_q_agg'] == 'min':
            q = jnp.min(qs, axis=0)
        else:
            q = jnp.mean(qs, axis=0)

        # particle selection (1, N, act_dim)
        q = q.reshape(-1, self.config['vgf_particles'])  
        best_idx = jnp.argmax(q, axis=1)   

        actions = particles[jnp.arange(particles.shape[0]), best_idx]   # (1, D)
        return actions.squeeze() 

    @jax.jit
    def sample_bc_actions(
        self,
        observations,
        seed,
    ):
        """Compute actions from the selected BC actor."""
        if self.config['bc_policy_type'] == 'gau': 
            dist = self.actor_bc(observations)
            actions = dist.sample(seed=seed)
        elif self.config['bc_policy_type'] == 'flow':
            noises = jax.random.normal(
                seed,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self._actor_action_dim(),
                ), 
            )
            actions = noises
            # Euler method.
            for i in range(self.config['bc_flow_steps']):
                t = jnp.full((*observations.shape[:-1], 1), i / self.config['bc_flow_steps'])
                vels = self.actor_bc(observations, actions, t)
                actions = actions + vels / self.config['bc_flow_steps']
        elif self.config['bc_policy_type'] == 'drift':
            noises = jax.random.normal(
                seed,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self._actor_action_dim(),
                ),
            )
            actions = self.actor_bc(observations, noises)
        else:
            raise ValueError(f"Unknown policy_type: {self.config['bc_policy_type']}")
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
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config['action_chunking']:
            full_actions = jnp.concatenate([ex_actions] * config['horizon_length'], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]
        ex_times = full_actions[..., :1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_bc'] = encoder_module()

        # Define bc actors
        if config['bc_policy_type'] == 'flow':
            actor_bc_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=full_action_dim,
                layer_norm=config['actor_layer_norm'],
                encoder=encoders.get('actor_bc'),
            )
            actor_bc_params = actor_bc_def.init(actor_key, ex_observations, full_actions, ex_times)['params']
        elif config['bc_policy_type'] == 'drift':
            actor_bc_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=full_action_dim,
                layer_norm=True,
                encoder=encoders.get('actor_bc'),
                swap=True,
            )
            actor_bc_params = actor_bc_def.init(actor_key, ex_observations, full_actions)['params']
        elif config['bc_policy_type'] == 'gau':
            actor_bc_def = Actor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=full_action_dim,
                layer_norm=config['actor_layer_norm'],
                state_dependent_std=False,
                const_std=None,
                tanh_squash=config['bc_use_tanh'],
                encoder=encoders.get('actor_bc'),
            )
            actor_bc_params = actor_bc_def.init(actor_key, ex_observations)['params']
        else:
            raise ValueError(f"Unknown policy_type: {config['bc_policy_type']}")
        actor_bc = TrainState.create(actor_bc_def, actor_bc_params, tx=optax.adam(learning_rate=config['actor_lr']))

        # Define value functions
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        critic_params = critic_def.init(critic_key, ex_observations, full_actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=config['value_lr']))
        target_critic = TrainState.create(critic_def, critic_params)

        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        value_params = value_def.init(value_key, ex_observations)['params']
        value = TrainState.create(value_def, value_params, tx=optax.adam(learning_rate=config['value_lr']))

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, critic=critic, target_critic=target_critic, value=value, actor_bc=actor_bc, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='vgf',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            value_lr=3e-4,  # Value learning rate.
            actor_lr=3e-4,  # BC actor learning rate.
            batch_size=256,  # Batch size.
            activations='relu',  # 'relu' or 'gelu'.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            critic_loss='td',  # Critic loss type ('td' or 'ivr-q').
            expectile=0.9,  # SQL alpha or IQL expectile if choosing ivr in critic loss.
            train_q_agg='mean',  # Aggregation method for target Q values.
            vgf_q_agg='mean',  # Aggregation method for Q values during evaluation.
            vgf_svgd_type='naive',  # ('naive' or 'sinkhorn') SVGD target score.
            vgf_sinkhorn_tau=None,  # None uses the same median heuristic as the RBF kernel.
            bc_policy_type='drift',  # ('gau', 'flow', or 'drift') BC actor family.
            bc_flow_steps=10,  # Number of bc flow steps.
            bc_use_tanh=False,  # Whether to use tanh squash for the bc actor.
            drift_temps=(0.1,),  # Kernel temperatures for drift BC loss.
            gen_per_label=8,  # Number of generated drift samples per dataset action.
            freeze_actor_bc_after=-1,  # Freeze BC actor after this many actor updates; -1 disables.
            vgf_particles=10,  # Number of vgf particles.
            train_particle_select='mean',  # ('max' or 'mean').
            eval_particle_select='max',  # ('max', 'softmax' or 'mean').
            train_vgf_steps=1,  # Number of vgf steps during training.
            vgf_lr=0.05,  # Learning rate of vgf.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int),  # Will be set from --horizon_length.
            action_chunking=True,  # If False, use H-step returns with only the first action.
        )
    )
    return config
