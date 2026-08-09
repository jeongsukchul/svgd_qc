import unittest

import jax
import jax.numpy as jnp

from agents.anq2 import ANQ2Agent, expectile_loss, get_config


def make_batch(batch_size=4, horizon_length=1):
    keys = jax.random.split(jax.random.PRNGKey(21), 4)
    return {
        "observations": jax.random.normal(keys[0], (batch_size, 5)),
        "actions": jnp.tanh(
            jax.random.normal(keys[1], (batch_size, horizon_length, 3))
        ),
        "next_observations": jax.random.normal(
            keys[2], (batch_size, horizon_length, 5)
        ),
        "next_actions": jnp.tanh(
            jax.random.normal(keys[3], (batch_size, horizon_length, 3))
        ),
        "rewards": -jnp.ones((batch_size, horizon_length), dtype=jnp.float32),
        "masks": jnp.ones((batch_size, horizon_length), dtype=jnp.float32),
        "valid": jnp.ones((batch_size, horizon_length), dtype=jnp.float32),
    }


def make_agent(action_chunking=False, horizon_length=1, **overrides):
    config = get_config()
    config.actor_hidden_dims = (16, 16)
    config.critic_hidden_dims = (16, 16)
    config.num_qs = 2
    config.horizon_length = horizon_length
    config.action_chunking = action_chunking
    for key, value in overrides.items():
        config[key] = value
    return ANQ2Agent.create(
        0,
        jnp.zeros((5,), dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.float32),
        config,
    )


class ANQ2AgentTest(unittest.TestCase):
    def test_has_no_value_and_only_critic_target(self):
        agent = make_agent()
        self.assertEqual(
            set(agent.network.params),
            {
                "modules_actor",
                "modules_aux_actor",
                "modules_critic",
                "modules_target_critic",
            },
        )
        self.assertFalse(hasattr(agent, "value_loss"))

    def test_expectile_loss_is_asymmetric(self):
        losses = expectile_loss(jnp.array([-2.0, 2.0]), 0.8)
        self.assertTrue(bool(jnp.allclose(losses, jnp.array([0.8, 3.2]))))

    def test_complete_update_is_finite(self):
        for q_agg in ("min", "mean"):
            with self.subTest(q_agg=q_agg):
                agent = make_agent(q_agg=q_agg)
                updated, info = agent.update(make_batch())
                self.assertEqual(
                    int(updated.network.step), int(agent.network.step) + 1
                )
                self.assertNotIn("value/loss", info)
                for key in (
                    "total_loss",
                    "critic/loss",
                    "critic/target_delta_rms",
                    "aux_actor/loss",
                    "aux_actor/improvement",
                    "actor/loss",
                    "actor/improvement",
                    "grad/norm",
                ):
                    self.assertTrue(bool(jnp.isfinite(info[key])), key)

    def test_data_and_refine_aggregations_are_independent(self):
        qs = jnp.array([[1.0, 4.0], [3.0, 2.0]])
        agent = make_agent(data_q_agg="min", refine_q_agg="mean")
        self.assertTrue(
            bool(
                jnp.allclose(
                    agent._aggregate_qs(qs, agent.config["data_q_agg"]),
                    jnp.array([1.0, 2.0]),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    agent._aggregate_qs(qs, agent.config["refine_q_agg"]),
                    jnp.array([2.0, 3.0]),
                )
            )
        )

    def test_action_shapes_and_bounds(self):
        agent = make_agent()
        actions = agent.sample_actions(jnp.zeros((6, 5), dtype=jnp.float32))
        self.assertEqual(actions.shape, (6, 3))
        self.assertTrue(bool(jnp.all(actions >= -1.0)))
        self.assertTrue(bool(jnp.all(actions <= 1.0)))

        chunked = make_agent(action_chunking=True, horizon_length=3)
        chunked_actions = chunked.sample_actions(
            jnp.zeros((2, 5), dtype=jnp.float32)
        )
        self.assertEqual(chunked_actions.shape, (2, 3, 3))
        updated, info = chunked.update(make_batch(horizon_length=3))
        self.assertEqual(int(updated.network.step), 2)
        self.assertTrue(bool(jnp.isfinite(info["total_loss"])))

    def test_invalid_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "critic_expectile"):
            make_agent(critic_expectile=1.0)
        with self.assertRaisesRegex(ValueError, "data_q_agg"):
            make_agent(data_q_agg="median")
        with self.assertRaisesRegex(ValueError, "refine_q_agg"):
            make_agent(refine_q_agg="median")
        with self.assertRaisesRegex(ValueError, "lam"):
            make_agent(lam=-0.1)


if __name__ == "__main__":
    unittest.main()
