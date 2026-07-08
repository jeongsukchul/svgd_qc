import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from functools import partial
from einops import repeat

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, MLP

class MDFPAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    @partial(jax.jit, static_argnames=("R_list"))
    def drift_loss(
        gen,
        fixed_pos,
        fixed_neg=None,
        weight_gen=None,
        weight_pos=None,
        weight_neg=None,
        R_list=(0.05,),
    ):
        '''
        Args:
            gen: [B, C_g, S]
            fixed_pos: [B, C_p, S]
            fixed_neg: [B, C_n, S] (optional, can be None)
            weight_gen: [B, C_g] (optional; if None: weight is 1)
            weight_pos: [B, C_p] (optional; if None: weight is 1)
            weight_neg: [B, C_n] (optional; if None: weight is 1)
            R_list: a list of R values to use for the kernel function
        Returns:
            loss: [batch_size]
            (optional) info: a dict with entries:
                scale: the scale of the loss
                loss_R: the loss for each R value
        '''
        
        # 1. Defaults & Casting
        B, C_g, S = gen.shape
        C_p = fixed_pos.shape[1]
        
        if fixed_neg is None:
            fixed_neg = jnp.zeros_like(gen[:, :0, :])
        C_n = fixed_neg.shape[1]

        if weight_gen is None:
            weight_gen = jnp.ones_like(gen[:, :, 0])
        if weight_pos is None:
            weight_pos = jnp.ones_like(fixed_pos[:, :, 0])
        if weight_neg is None:
            weight_neg = jnp.ones_like(fixed_neg[:, :, 0])
        gen = gen.astype(jnp.float32)
        fixed_pos = fixed_pos.astype(jnp.float32)
        fixed_neg = fixed_neg.astype(jnp.float32)
        weight_gen = weight_gen.astype(jnp.float32)
        weight_pos = weight_pos.astype(jnp.float32)
        weight_neg = weight_neg.astype(jnp.float32)
        old_gen = jax.lax.stop_gradient(gen)
        targets = jnp.concatenate([old_gen, fixed_neg, fixed_pos], axis=1)
        targets_w = jnp.concatenate([weight_gen, weight_neg, weight_pos], axis=1)

        def cdist(x, y, eps=1e-8):
            # [B, N, D] x [B, M, D] -> [B, N, M]
            xydot = jnp.einsum("bnd,bmd->bnm", x, y)
            xnorms = jnp.einsum("bnd,bnd->bn", x, x)
            ynorms = jnp.einsum("bmd,bmd->bm", y, y)
            sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
            return jnp.clip(sq_dist, a_min=eps)

        # 2. Core Logic (Wrapped for stop_gradient)
        def calculate_scaled_goal_and_factor(old_gen_in, targets_in, targets_w_in):
            # --- Scaling ---
            info = {}
            dist = cdist(old_gen_in, targets_in)
            weighted_dist = dist * targets_w_in[:, None, :] # [B, C_g, C_g + C_n + C_p]
            scale = weighted_dist.mean() / targets_w_in.mean() # [B]
            info["scale"] = scale

            scale_inputs = jnp.clip(scale / jnp.sqrt(S), a_min=1e-3) # Normalize coords to have order 1
            old_gen_scaled = old_gen_in / scale_inputs
            targets_scaled = targets_in / scale_inputs
            
            # Normalize distance for kernel
            dist_normed = dist / jnp.clip(scale, a_min=1e-3)
            
            # --- Masking ---
            mask_val = 100.0
            diag_mask = jnp.eye(C_g, dtype=jnp.float32)
            block_mask = jnp.pad(diag_mask, ((0, 0), (0, C_n + C_p)))
            block_mask = jnp.expand_dims(block_mask, 0)
            dist_normed = dist_normed + block_mask * mask_val

            # score = jax.grad(score_fn)(old_gen_in)
            # scores_scaled = scores * scale_inputs

            # --- Force Loop ---
            force_across_R = jnp.zeros_like(old_gen_scaled)
            
            for R in R_list:
                logits = -dist_normed / (2 * R * R)

                affinity = jax.nn.softmax(logits, axis=-1)
                aff_transpose = jax.nn.softmax(logits, axis=-2)
                affinity = jnp.sqrt(jnp.clip(affinity * aff_transpose, a_min=1e-6))

                affinity = affinity * targets_w_in[:, None, :]

                split_idx = C_g + C_n
                aff_neg = affinity[:, :, :split_idx]
                aff_pos = affinity[:, :, split_idx:]
                
                sum_pos = jnp.sum(aff_pos, axis=-1, keepdims=True)
                r_coeff_neg = -aff_neg * sum_pos
                sum_neg = jnp.sum(aff_neg, axis=-1, keepdims=True)
                r_coeff_pos = aff_pos * sum_neg
                
                R_coeff = jnp.concatenate([r_coeff_neg, r_coeff_pos], axis=2)

                total_force_R = jnp.einsum("biy,byx->bix", R_coeff, targets_scaled)

                total_coeffs = R_coeff.sum(axis=-1) # guaranteed to be 0, in no_repulsion case
                total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
                f_norm_val = (total_force_R ** 2).mean() # [B]

                # aff_gen = aff_neg[:, :, :C_g]
                # # aff_gen = aff_gen / jnp.clip(aff_gen.sum(axis=-1, keepdims=True), a_min=1e-6)
                
                # score_force = jnp.einsum("bij,bjd->bid", aff_gen, scores_scaled)
                # s_norm_val = (score_force ** 2).mean()
                # score_force = score_force / jnp.sqrt(jnp.clip(s_norm_val, 1e-8))

                info[f"loss_{R}"] = f_norm_val
                # info[f"score_norm"] = s_norm_val

                force_scale = jnp.sqrt(jnp.clip(f_norm_val, a_min=1e-8)) # normalize force of each temperature
                force_across_R = force_across_R + total_force_R / force_scale
                # force_across_R = force_across_R + total_force_R / force_scale + score_coef * score_force
            
            goal_scaled = old_gen_scaled + force_across_R
            
            return goal_scaled, scale_inputs, info

        # 3. Compute Goal (No Gradients)
        goal_scaled, scale_inputs, info = jax.lax.stop_gradient(
            calculate_scaled_goal_and_factor(old_gen, targets, targets_w)
        )
        gen_scaled = gen / scale_inputs
        diff = gen_scaled - goal_scaled
        loss = jnp.mean(diff ** 2, axis=(-1, -2))
        info = jax.tree.map(lambda x: x.mean(), info)

        return loss, info

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
        
        # TD loss
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
        
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the Drifting actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, noise_rng, sample_rng = jax.random.split(rng, 3)

        # BC drift loss.
        noises = jax.random.normal(noise_rng, (batch_size * self.config['num_generations'], action_dim))
        obs_repeated = jnp.repeat(batch['observations'], self.config['num_generations'], axis=0)
        generated_actions = self.network.select('actor_drift')(obs_repeated, noises, params=grad_params).reshape(batch_size, self.config['num_generations'], action_dim)

        positive_actions = jnp.expand_dims(batch_actions, axis=1)  # [B, 1, horizon_length * action_dim]


        bc_drift_loss, drift_info = self.drift_loss(gen=generated_actions, fixed_pos=positive_actions, R_list=self.config['R_list'])
        bc_drift_loss = bc_drift_loss.mean()

        # Total loss.
        actor_loss = bc_drift_loss

        return actor_loss, {
            'actor_loss': actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        # agent.target_update(new_network, 'obs_rep')
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )
            actions = self.network.select(f'actor_drift')(observations, noises)
            # actions = self.network.select(f'actor_distill')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config["actor_num_samples"], action_dim
                ),
            )
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.network.select(f'actor_drift')(observations, noises)
            # actions = self.network.select(f'actor_distill')(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            if self.config["q_agg"] == "mean":
                q = self.network.select("critic")(observations, actions).mean(axis=0)
            else:
                q = self.network.select("critic")(observations, actions).min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
                bshape + (action_dim,))

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
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )
        actor_drift_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        
        network_info = dict(
            actor_drift=(actor_drift_def, (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        # if encoders.get('actor_bc_flow') is not None:
        #     # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
        #     network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        # params[f'modules_target_obs_rep'] = params[f'modules_obs_rep']
        params[f'modules_target_critic'] = params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='mdfp',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            num_qs=2, # critic ensemble size
            num_generations=8,  # Number of sample generations.
            R_list=(0.2,),  # Kernel temperatures for drift loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            actor_type="best-of-n",
            actor_num_samples=16,  # for actor_type="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            what=ml_collections.config_dict.placeholder(str),
        )
    )
    return config