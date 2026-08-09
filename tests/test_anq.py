import unittest

import jax
import jax.numpy as jnp

from agents.anq import ANQAgent, expectile_loss, get_config


def make_agent(action_chunking=False, horizon_length=1, **overrides):
    config = get_config()
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.num_qs = 2
    config.horizon_length = horizon_length
    config.action_chunking = action_chunking
    config.use_actor_lr_schedule = False
    for key, value in overrides.items():
        config[key] = value
    return ANQAgent.create(
        seed=0,
        ex_observations=jnp.zeros((5,), dtype=jnp.float32),
        ex_actions=jnp.zeros((3,), dtype=jnp.float32),
        config=config,
    )


def make_batch(batch_size=4, horizon_length=1):
    ob_rng, action_rng, next_ob_rng = jax.random.split(
        jax.random.PRNGKey(7), 3
    )
    return {
        "observations": jax.random.normal(ob_rng, (batch_size, 5)),
        "actions": jnp.tanh(
            jax.random.normal(action_rng, (batch_size, horizon_length, 3))
        ),
        "valid": jnp.ones((batch_size, horizon_length), dtype=jnp.float32),
        "next_observations": jax.random.normal(
            next_ob_rng, (batch_size, horizon_length, 5)
        ),
        "rewards": -jnp.ones((batch_size, horizon_length), dtype=jnp.float32),
        "masks": jnp.ones((batch_size, horizon_length), dtype=jnp.float32),
    }


def tree_allclose(left, right):
    comparisons = jax.tree_util.tree_map(jnp.allclose, left, right)
    return all(bool(value) for value in jax.tree_util.tree_leaves(comparisons))


class ANQAgentTest(unittest.TestCase):
    def test_critic_is_the_only_target_network(self):
        agent = make_agent()
        targets = {
            key for key in agent.network.params if "modules_target_" in key
        }
        self.assertEqual(targets, {"modules_target_critic"})

    def test_expectile_loss_is_asymmetric(self):
        losses = expectile_loss(jnp.array([-2.0, 2.0]), expectile=0.8)
        self.assertTrue(bool(jnp.allclose(losses, jnp.array([0.8, 3.2]))))

    def test_single_task_action_shape_and_bounds(self):
        agent = make_agent()
        action = agent.sample_actions(
            jnp.zeros((5,), dtype=jnp.float32), jax.random.PRNGKey(1)
        )
        batched_actions = agent.sample_actions(
            jnp.zeros((6, 5), dtype=jnp.float32), jax.random.PRNGKey(2)
        )

        self.assertEqual(action.shape, (3,))
        self.assertEqual(batched_actions.shape, (6, 3))
        self.assertTrue(bool(jnp.all(action >= -1.0)))
        self.assertTrue(bool(jnp.all(action <= 1.0)))

    def test_chunked_action_shape_is_supported(self):
        agent = make_agent(action_chunking=True, horizon_length=3)
        actions = agent.sample_actions(
            jnp.zeros((2, 5), dtype=jnp.float32), jax.random.PRNGKey(3)
        )
        self.assertEqual(actions.shape, (2, 3, 3))

        updated, info = agent.update(make_batch(horizon_length=3))
        self.assertEqual(int(updated.network.step), int(agent.network.step) + 1)
        self.assertTrue(bool(jnp.isfinite(info["total_loss"])))

    def test_complete_update_is_finite(self):
        for q_agg in ("min", "mean"):
            with self.subTest(q_agg=q_agg):
                agent = make_agent(q_agg=q_agg)
                updated, info = agent.update(make_batch())

                self.assertEqual(
                    int(updated.network.step), int(agent.network.step) + 1
                )
                for key in (
                    "total_loss",
                    "value/loss",
                    "critic/loss",
                    "aux_actor/loss",
                    "actor/loss",
                    "grad/norm",
                ):
                    self.assertTrue(bool(jnp.isfinite(info[key])), key)

    def test_q_aggregation_matches_requested_reduction(self):
        qs = jnp.array([[1.0, 4.0], [3.0, 2.0]])
        min_agent = make_agent(q_agg="min")
        mean_agent = make_agent(q_agg="mean")

        self.assertTrue(
            bool(jnp.allclose(min_agent._aggregate_qs(qs), jnp.array([1.0, 2.0])))
        )
        self.assertTrue(
            bool(jnp.allclose(mean_agent._aggregate_qs(qs), jnp.array([2.0, 3.0])))
        )

        mixed_agent = make_agent(data_q_agg="min", refine_q_agg="mean")
        self.assertTrue(
            bool(
                jnp.allclose(
                    mixed_agent._aggregate_qs(
                        qs, mode=mixed_agent.config["data_q_agg"]
                    ),
                    jnp.array([1.0, 2.0]),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    mixed_agent._aggregate_qs(
                        qs, mode=mixed_agent.config["refine_q_agg"]
                    ),
                    jnp.array([2.0, 3.0]),
                )
            )
        )

    def test_actor_and_target_updates_are_delayed(self):
        agent = make_agent(policy_freq=2)
        original_actor = agent.network.params["modules_actor"]
        original_target = agent.network.params["modules_target_critic"]

        first, first_info = agent.update(make_batch())
        self.assertEqual(float(first_info["policy_update"]), 0.0)
        self.assertTrue(
            tree_allclose(original_actor, first.network.params["modules_actor"])
        )
        self.assertTrue(
            tree_allclose(
                original_target,
                first.network.params["modules_target_critic"],
            )
        )

        second, second_info = first.update(make_batch())
        self.assertEqual(float(second_info["policy_update"]), 1.0)
        self.assertFalse(
            tree_allclose(original_actor, second.network.params["modules_actor"])
        )
        self.assertFalse(
            tree_allclose(
                original_target,
                second.network.params["modules_target_critic"],
            )
        )

    def test_invalid_hyperparameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "expectile"):
            make_agent(expectile=1.0)
        with self.assertRaisesRegex(ValueError, "lam"):
            make_agent(lam=-0.1)
        with self.assertRaisesRegex(ValueError, "policy_freq"):
            make_agent(policy_freq=0)
        with self.assertRaisesRegex(ValueError, "q_agg"):
            make_agent(q_agg="median")
        with self.assertRaisesRegex(ValueError, "data_q_agg"):
            make_agent(data_q_agg="median")
        with self.assertRaisesRegex(ValueError, "refine_q_agg"):
            make_agent(refine_q_agg="median")
        with self.assertRaisesRegex(ValueError, "clipping range"):
            make_agent(aux_weight_min=2.0, aux_weight_max=1.0)


if __name__ == "__main__":
    unittest.main()
