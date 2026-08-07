import unittest

import jax
import jax.numpy as jnp

from agents.anq_dfp import (
    ANQDFPAgent,
    get_config as get_anq_dfp_config,
    refine_actions,
    td_expectile_loss,
)
from agents.anq_stdfp import (
    ANQSTDFPAgent,
    get_config as get_anq_stdfp_config,
)


def make_batch(batch_size=4, horizon_length=1):
    ob_rng, action_rng, next_ob_rng = jax.random.split(
        jax.random.PRNGKey(12), 3
    )
    return {
        "observations": jax.random.normal(ob_rng, (batch_size, 5)),
        "actions": jnp.tanh(
            jax.random.normal(action_rng, (batch_size, horizon_length, 3))
        ),
        "next_observations": jax.random.normal(
            next_ob_rng, (batch_size, horizon_length, 5)
        ),
        "rewards": -jnp.ones(
            (batch_size, horizon_length), dtype=jnp.float32
        ),
        "masks": jnp.ones(
            (batch_size, horizon_length), dtype=jnp.float32
        ),
        "valid": jnp.ones(
            (batch_size, horizon_length), dtype=jnp.float32
        ),
    }


def make_dfp(**overrides):
    config = get_anq_dfp_config()
    config.horizon_length = 1
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.num_qs = 2
    config.actor_num_samples = 3
    config.gen_per_label = 4
    for key, value in overrides.items():
        config[key] = value
    return ANQDFPAgent.create(
        0,
        jnp.zeros((5,), dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.float32),
        config,
    )


def make_stdfp(**overrides):
    config = get_anq_stdfp_config()
    config.horizon_length = 1
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.num_qs = 2
    config.gen_per_label = 4
    config.best_of_n = 2
    config.noise_state_dependent_std = False
    for key, value in overrides.items():
        config[key] = value
    return ANQSTDFPAgent.create(
        0,
        jnp.zeros((5,), dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.float32),
        config,
    )


class ANQDriftVariantsTest(unittest.TestCase):
    def test_td_expectile_is_asymmetric(self):
        losses = td_expectile_loss(
            jnp.array([-2.0, 2.0]), expectile=0.8
        )
        self.assertTrue(bool(jnp.allclose(losses, jnp.array([0.8, 3.2]))))

    def test_dfp_has_refiner_but_no_value_auxiliary_or_final_actor(self):
        agent = make_dfp()
        modules = set(agent.network.params)
        self.assertEqual(
            modules,
            {
                "modules_actor_drift",
                "modules_critic",
                "modules_refine_actor",
                "modules_target_critic",
                "modules_target_refine_actor",
            },
        )

    def test_stdfp_has_latent_and_refine_actors_but_no_value_or_final_actor(self):
        agent = make_stdfp()
        modules = set(agent.network.params)
        self.assertIn("modules_noise_actor", modules)
        self.assertIn("modules_actor_drift", modules)
        self.assertIn("modules_refine_actor", modules)
        self.assertIn("modules_target_refine_actor", modules)
        self.assertNotIn("modules_value", modules)
        self.assertNotIn("modules_aux_actor", modules)
        self.assertNotIn("modules_actor", modules)

    def test_refinement_is_bounded_by_action_space_and_radius(self):
        agent = make_dfp(refine_radius=0.15)
        observations = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        base_actions = jnp.zeros((6, 3), dtype=jnp.float32)
        refined, delta = refine_actions(agent, observations, base_actions)

        self.assertTrue(bool(jnp.all(jnp.isfinite(refined))))
        self.assertTrue(bool(jnp.all(refined >= -1.0)))
        self.assertTrue(bool(jnp.all(refined <= 1.0)))
        self.assertTrue(
            bool(jnp.all(jnp.linalg.norm(delta, axis=-1) <= 0.15001))
        )

    def test_sampling_shapes_and_bounds(self):
        for agent in (make_dfp(), make_stdfp()):
            with self.subTest(agent=agent.config["agent_name"]):
                actions = agent.sample_actions(
                    jnp.zeros((5, 5), dtype=jnp.float32),
                    jax.random.PRNGKey(2),
                )
                self.assertEqual(actions.shape, (5, 3))
                self.assertTrue(bool(jnp.all(jnp.isfinite(actions))))
                self.assertTrue(bool(jnp.all(actions >= -1.0)))
                self.assertTrue(bool(jnp.all(actions <= 1.0)))

    def test_complete_updates_are_finite(self):
        for agent in (make_dfp(), make_stdfp()):
            with self.subTest(agent=agent.config["agent_name"]):
                updated, info = agent.update(make_batch())
                self.assertEqual(
                    int(updated.network.step), int(agent.network.step) + 1
                )
                for key in (
                    "critic/critic_loss",
                    "critic/target_delta_rms",
                    "actor/actor_drift_loss",
                    "actor/refine_actor_loss",
                    "grad/norm",
                ):
                    self.assertTrue(bool(jnp.isfinite(info[key])), key)

    def test_refine_actor_and_target_are_updated(self):
        agent = make_dfp()
        source_before = agent.network.params["modules_refine_actor"]
        target_before = agent.network.params["modules_target_refine_actor"]
        updated, _ = agent.update(make_batch())
        source_after = updated.network.params["modules_refine_actor"]
        target_after = updated.network.params["modules_target_refine_actor"]

        source_changed = any(
            not bool(jnp.allclose(before, after))
            for before, after in zip(
                jax.tree_util.tree_leaves(source_before),
                jax.tree_util.tree_leaves(source_after),
            )
        )
        target_changed = any(
            not bool(jnp.allclose(before, after))
            for before, after in zip(
                jax.tree_util.tree_leaves(target_before),
                jax.tree_util.tree_leaves(target_after),
            )
        )
        self.assertTrue(source_changed)
        self.assertTrue(target_changed)

    def test_refinement_is_one_actor_forward_pass(self):
        config = get_anq_dfp_config()
        self.assertNotIn("refine_steps", config)
        self.assertNotIn("refine_step_size", config)

    def test_q_aggregation_modes_update(self):
        for q_agg in ("min", "mean"):
            with self.subTest(q_agg=q_agg):
                agent = make_dfp(q_agg=q_agg)
                _, info = agent.update(make_batch())
                self.assertTrue(bool(jnp.isfinite(info["critic/critic_loss"])))

    def test_invalid_refinement_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "critic_expectile"):
            make_dfp(critic_expectile=1.0)
        with self.assertRaisesRegex(ValueError, "refine_radius"):
            make_dfp(refine_radius=-0.1)
        with self.assertRaisesRegex(ValueError, "refine_lambda"):
            make_dfp(refine_lambda=-0.1)
        with self.assertRaisesRegex(ValueError, "q_agg"):
            make_dfp(q_agg="pessimistic")


if __name__ == "__main__":
    unittest.main()
