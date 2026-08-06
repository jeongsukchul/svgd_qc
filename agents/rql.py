import copy
from typing import Any

from functools import partial
import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class RQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def _action_mask(self, batch, action_dim):
        if self.config["action_chunking"] and "valid" in batch:
            return jnp.repeat(batch["valid"], self.config["action_dim"], axis=-1)
        return jnp.ones((batch["actions"].shape[0], action_dim), dtype=batch["actions"].dtype)

    def _valid(self, batch, target):
        if "valid" in batch:
            return batch["valid"][..., -1]
        return jnp.ones_like(target)

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        batch_size = batch["actions"].shape[0]
        action_dim = self.config["full_action_dim"]

        rng, n_rng, u_rng, r_rng = jax.random.split(rng, 4)

        next_obs = batch["next_observations"][..., -1, :]
        next_state = jnp.concatenate(
            [
                next_obs,
                jax.random.normal(n_rng, (batch_size, action_dim)),
                jnp.zeros((batch_size, 1,)),
            ],
            axis=-1,
        )  # s', x_0', 0
        next_qs = self.network.select("target_value")(next_state)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        num_uniform = batch_size // 2
        d = jnp.concatenate(
            [
                jax.random.uniform(u_rng, (num_uniform,)),
                jax.random.randint(
                    r_rng,
                    (batch_size - num_uniform,),
                    0,
                    self.config["flow_steps"] + 1,
                )
                / self.config["flow_steps"],
            ],
            0,
        )
        d_b = d / self.config["flow_steps"]

        actions = self._batch_actions(batch)

        # Reversing flow
        x_f = jnp.copy(actions)
        f = jnp.ones((batch_size, 1))
        for i in range(self.config["flow_steps"]):
            fm_actor = jnp.concatenate([batch["observations"], x_f, f], -1)
            out = self.network.select("actor")(fm_actor).mode()
            x_f = x_f - out * d_b[..., None]
            f = f - d_b[..., None]

        state = jnp.concatenate(
            [batch["observations"], jax.lax.stop_gradient(x_f), f],
            axis=-1,
        )  # s, x_f, f

        q = self.network.select("value")(state, params=grad_params)

        tqt_q = (
            batch["rewards"][..., -1]
            + (self.config["discount"] ** self.config["horizon_length"])
            * next_q
            * batch["masks"][..., -1]
        )

        valids = self._valid(batch, tqt_q)
        critic_loss = (self.expectile_loss(tqt_q - q, tqt_q - q, self.config["expectile"]) * valids).mean()

        # BC flow loss
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = actions
        t = jax.random.uniform(t_rng, (batch_size, 1))

        x_t = (1 - t) * x_0 + t * x_1
        tgt = x_1 - x_0
        fm_actor = jnp.concatenate([batch["observations"], x_t, t], axis=-1)
        pred = self.network.select("actor")(
            fm_actor, params=grad_params
        ).mode()
        q_pe = self.network.select("value")(
            jnp.concatenate(
                [
                    batch["observations"],
                    x_t
                    + pred * jnp.minimum(1 / self.config["flow_steps"], 1 - t),
                    jnp.clip(t + 1 / self.config["flow_steps"], max=1),
                ],
                axis=-1,
            )
        )
        q_pe = q_pe.mean(axis=0)

        ac_mask = self._action_mask(batch, action_dim)
        bc_loss = (jnp.square(pred - tgt) * ac_mask).mean()
        actor_loss = -(q_pe * valids).mean()

        total_loss = (
            actor_loss + bc_loss * self.config["alpha"] + critic_loss
        )

        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "q": q.mean(),
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),

            "q_pe_mean": q_pe.mean(),
            "q_pe_max": q_pe.max(),
            "q_pe_min": q_pe.min(),
        }

    def target_update(self, network, module_name, d):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        self.target_update(new_network, "value", d=self.config["tau"])
        self.target_update(new_network, "actor", d=1-self.config["ema"])

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_flow_actions(
        self,
        observations,
        noise,
        seed=None,
        temperature=0.0,
    ):
        actions = noise
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            fm_actor = jnp.concatenate([observations, actions, t], axis=-1)
            out = self.network.select("actor" if temperature > 0 else "target_actor")(fm_actor).mode()
            actions = actions + (out / self.config["flow_steps"])
        actions = jnp.clip(actions, -1, 1)
        return actions

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(
        self,
        observations,
        rng=None,
        seed=None,
        temperature=0.0,
    ):
        seed = rng if seed is None else seed
        seed = jax.random.PRNGKey(0) if seed is None else seed
        action_rng, n_rng = jax.random.split(seed)

        noise = jax.random.normal(
            n_rng,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                self.config["full_action_dim"],
            ),
        )
        actions = self.compute_flow_actions(
            observations,
            seed=action_rng,
            noise=noise,
            temperature=temperature,
        )
        if self.config["action_chunking"]:
            actions = jnp.reshape(
                actions,
                actions.shape[:-1]
                + (self.config["horizon_length"], self.config["action_dim"]),
            )
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

        if config.get("horizon_length") is None:
            config["horizon_length"] = config.get("h", 1)
        if config.get("action_chunking") is None:
            config["action_chunking"] = True

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], -1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]
        ex_times = full_actions[..., :1]
        ex_in = jnp.concatenate([ex_observations, full_actions, ex_times], -1)

        value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )

        actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=False,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=1,
        )

        network_info = dict(
            value=(value_def, (ex_in,)),
            target_value=(copy.deepcopy(value_def), (ex_in)),
            actor=(actor_def, (ex_in,)),
            target_actor=(
                copy.deepcopy(actor_def),
                (ex_in,)
            ),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]
        params["modules_target_actor"] = params["modules_actor"]
        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        config["full_action_dim"] = full_action_dim
        config["h"] = config["horizon_length"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="rql",
            h=5,
            ob_dims=mlc.config_dict.placeholder(list),
            action_dim=mlc.config_dict.placeholder(int),
            full_action_dim=mlc.config_dict.placeholder(int),
            alpha=1.0,
            expectile=0.5,
            ensemble_ct=10,
            rho=0.0, 
            lr=3e-4,
            discount=0.99,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            tau=0.005,
            ema=0.999,
            flow_steps=10,
            q_agg="mean",
            horizon_length=mlc.config_dict.placeholder(int),
            action_chunking=True,
        )
    )
    return config
