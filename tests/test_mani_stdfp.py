import unittest

import jax
import jax.numpy as jnp

from agents.mani_stdfp import (
    ManiSTDFPAgent,
    generator_covariance,
    get_config,
    manifold_quadratic,
)


def make_agent(action_chunking=False, horizon_length=1, **overrides):
    config = get_config()
    config.horizon_length = horizon_length
    config.action_chunking = action_chunking
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.refine_hidden_dims = (16, 16)
    config.num_qs = 2
    config.gen_per_label = 4
    config.best_of_n = 2
    config.noise_state_dependent_std = False
    for key, value in overrides.items():
        config[key] = value
    return ManiSTDFPAgent.create(
        0,
        jnp.zeros((5,), dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.float32),
        config,
    )


def make_batch(batch_size=4, horizon_length=1):
    keys = jax.random.split(jax.random.PRNGKey(9), 3)
    return {
        "observations": jax.random.normal(keys[0], (batch_size, 5)),
        "actions": jnp.tanh(
            jax.random.normal(keys[1], (batch_size, horizon_length, 3))
        ),
        "next_observations": jax.random.normal(
            keys[2], (batch_size, horizon_length, 5)
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


class ManiSTDFPTest(unittest.TestCase):
    def test_covariance_and_quadratic_match_diagonal_geometry(self):
        jacobian = jnp.diag(jnp.array([2.0, 0.5], dtype=jnp.float32))
        delta = jnp.array([1.0, 1.0], dtype=jnp.float32)

        covariance = generator_covariance(jacobian)
        penalty = manifold_quadratic(delta, jacobian, ridge=0.25)

        self.assertTrue(
            bool(jnp.allclose(covariance, jnp.diag(jnp.array([4.0, 0.25]))))
        )
        self.assertTrue(
            bool(jnp.allclose(penalty, 1.0 / 4.25 + 1.0 / 0.5))
        )

    def test_generator_jacobian_has_action_by_latent_shape(self):
        agent = make_agent()
        observations = jnp.zeros((4, 5), dtype=jnp.float32)
        noises = jnp.zeros((4, 3), dtype=jnp.float32)

        jacobians = agent.generator_jacobians(observations, noises)

        self.assertEqual(jacobians.shape, (4, 3, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(jacobians))))

    def test_update_and_sampling_are_finite(self):
        agent = make_agent()
        updated, info = agent.update(make_batch())
        actions = updated.sample_actions(
            jnp.zeros((4, 5), dtype=jnp.float32),
            jax.random.PRNGKey(4),
        )

        self.assertEqual(int(updated.network.step), int(agent.network.step) + 1)
        self.assertEqual(actions.shape, (4, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(actions))))
        for key in (
            "total_loss",
            "actor/manifold_penalty",
            "actor/euclidean_penalty",
            "actor/generator_variance",
            "grad/norm",
        ):
            self.assertIn(key, info)
            self.assertTrue(bool(jnp.isfinite(info[key])), key)

    def test_chunked_actions_are_supported(self):
        agent = make_agent(action_chunking=True, horizon_length=2)
        updated, info = agent.update(make_batch(horizon_length=2))
        actions = updated.sample_actions(
            jnp.zeros((2, 5), dtype=jnp.float32),
            jax.random.PRNGKey(5),
        )

        # Chunked policies use the repository's flat action-chunk convention;
        # evaluation reshapes this to (horizon_length, action_dim).
        self.assertEqual(actions.shape, (2, 6))
        self.assertTrue(bool(jnp.isfinite(info["total_loss"])))

    def test_invalid_geometry_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "manifold_ridge"):
            make_agent(manifold_ridge=0.0)
        with self.assertRaisesRegex(ValueError, "lam"):
            make_agent(lam=-1.0)


if __name__ == "__main__":
    unittest.main()
