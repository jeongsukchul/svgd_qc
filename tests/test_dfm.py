import unittest

import jax
import jax.numpy as jnp

from agents.dfm import (
    DFMAgent,
    dfm_drift_loss,
    dfm_log_kde_loss,
    dfm_sinkhorn_loss,
    get_config,
    sample_time_pairs,
)


def make_agent(**overrides):
    config = get_config()
    config.encoder = None
    config.horizon_length = 2
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.actor_num_samples = 4
    config.gen_per_label = 4
    config.num_flow_steps = 2
    config.time_grid_ratio = 0.5
    config.sinkhorn_iters = 3
    for key, value in overrides.items():
        config[key] = value
    return DFMAgent.create(
        seed=0,
        ex_observations=jnp.zeros((3,), dtype=jnp.float32),
        ex_actions=jnp.zeros((2,), dtype=jnp.float32),
        config=config,
    )


def make_batch(batch_size=3, valid=None):
    if valid is None:
        valid = jnp.ones((batch_size, 2), dtype=jnp.float32)
    return {
        "observations": jnp.zeros((batch_size, 3), dtype=jnp.float32),
        "actions": jnp.zeros((batch_size, 2, 2), dtype=jnp.float32),
        "valid": valid,
        "next_observations": jnp.zeros((batch_size, 2, 3), dtype=jnp.float32),
        "rewards": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "masks": jnp.ones((batch_size, 2), dtype=jnp.float32),
    }


class DFMAgentTest(unittest.TestCase):
    def test_configuration_defaults_to_mixed_time_pairs_and_multiple_steps(self):
        config = get_config()
        self.assertGreater(config.num_flow_steps, 1)
        self.assertGreaterEqual(config.time_grid_ratio, 0.0)
        self.assertLess(config.time_grid_ratio, 1.0)

        make_agent(num_flow_steps=4, time_grid_ratio=0.25)
        with self.assertRaisesRegex(ValueError, "num_flow_steps"):
            make_agent(num_flow_steps=0)
        with self.assertRaisesRegex(ValueError, "time_grid_ratio"):
            make_agent(time_grid_ratio=-0.1)
        with self.assertRaisesRegex(ValueError, "time_grid_ratio"):
            make_agent(time_grid_ratio=1.1)

    def test_time_pair_sampler_mixes_exact_number_of_grid_pairs(self):
        start, end = sample_time_pairs(
            jax.random.PRNGKey(20),
            num_pairs=10,
            num_flow_steps=4,
            time_grid_ratio=0.5,
        )

        self.assertEqual(start.shape, (10, 1))
        self.assertEqual(end.shape, (10, 1))
        self.assertTrue(bool(jnp.all(start >= 0.0)))
        self.assertTrue(bool(jnp.all(end <= 1.0)))
        self.assertTrue(bool(jnp.all(start < end)))
        grid_pairs = jnp.isclose(end - start, 0.25) & jnp.isclose(
            start * 4.0, jnp.round(start * 4.0)
        )
        self.assertEqual(int(grid_pairs.sum()), 5)

        endpoint_start, endpoint_end = sample_time_pairs(
            jax.random.PRNGKey(21),
            num_pairs=4,
            num_flow_steps=1,
            time_grid_ratio=1.0,
        )
        self.assertTrue(bool(jnp.all(endpoint_start == 0.0)))
        self.assertTrue(bool(jnp.all(endpoint_end == 1.0)))

    def test_transport_is_an_identity_plus_residual_map(self):
        agent = make_agent()
        source = jax.random.normal(jax.random.PRNGKey(1), (5, 4, 4))
        observations = agent._expand_observations_for_particles(
            jnp.zeros((5, 3), dtype=jnp.float32), 4
        )

        mapped, residual = agent._one_step_transport(observations, source)

        self.assertTrue(bool(jnp.allclose(mapped, source + residual)))

        partial_map, partial_increment = agent._transport(
            observations, source, 0.25, 0.5
        )
        self.assertTrue(
            bool(jnp.allclose(partial_map, source + partial_increment))
        )

    def test_best_of_n_sampling_has_expected_shape_and_bounds(self):
        agent = make_agent()
        actions = agent.sample_actions(
            jnp.zeros((5, 3), dtype=jnp.float32),
            jax.random.PRNGKey(2),
        )

        self.assertEqual(actions.shape, (5, 4))
        self.assertTrue(bool(jnp.all(jnp.isfinite(actions))))
        self.assertTrue(bool(jnp.all(actions >= -1.0)))
        self.assertTrue(bool(jnp.all(actions <= 1.0)))

    def test_actor_loss_is_finite_and_masks_incomplete_chunks(self):
        for drift_backend in ("drift_loss", "log_kde", "sinkhorn"):
            with self.subTest(drift_backend=drift_backend):
                agent = make_agent(drift_backend=drift_backend)
                loss, info = agent.actor_loss(
                    make_batch(), agent.network.params, jax.random.PRNGKey(3)
                )
                self.assertTrue(bool(jnp.isfinite(loss)))
                self.assertTrue(bool(jnp.isfinite(info["residual_rms"])))

                invalid = jnp.zeros((3, 2), dtype=jnp.float32)
                masked_loss, masked_info = agent.actor_loss(
                    make_batch(valid=invalid),
                    agent.network.params,
                    jax.random.PRNGKey(4),
                )
                self.assertAlmostEqual(float(masked_loss), 0.0)
                self.assertAlmostEqual(
                    float(masked_info["valid_group_fraction"]), 0.0
                )

    def test_ddpg_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "best-of-n"):
            make_agent(actor_type="ddpg")

    def test_invalid_sinkhorn_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temp_pos"):
            make_agent(drift_backend="sinkhorn", temp_pos=0.0)
        with self.assertRaisesRegex(ValueError, "temp_neg"):
            make_agent(drift_backend="sinkhorn", temp_neg=-1.0)
        with self.assertRaisesRegex(ValueError, "positive odd"):
            make_agent(drift_backend="sinkhorn", sinkhorn_iters=2)

    def test_invalid_drift_backend_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "drift_backend"):
            make_agent(drift_backend="unknown")
        with self.assertRaisesRegex(ValueError, "drift_temps"):
            make_agent(drift_backend="drift_loss", drift_temps=())
        with self.assertRaisesRegex(ValueError, "log_kde_bandwidth"):
            make_agent(drift_backend="log_kde", log_kde_bandwidth=0.0)

    def test_all_loss_functions_return_finite_per_group_values(self):
        predicted = jax.random.normal(jax.random.PRNGKey(10), (3, 4, 2))
        target = jax.random.normal(jax.random.PRNGKey(11), (3, 4, 2))
        valid = jnp.ones((3,), dtype=jnp.float32)

        drift_values, drift_info = dfm_drift_loss(
            predicted, target, valid, (0.1,)
        )
        sinkhorn_values, sinkhorn_info = dfm_sinkhorn_loss(
            predicted, target, 1.0, 1.0, 3
        )
        log_kde_values, log_kde_info = dfm_log_kde_loss(
            predicted, target, 0.4
        )

        for values, info in (
            (drift_values, drift_info),
            (sinkhorn_values, sinkhorn_info),
            (log_kde_values, log_kde_info),
        ):
            self.assertEqual(values.shape, (3,))
            self.assertTrue(bool(jnp.all(jnp.isfinite(values))))
            self.assertTrue(
                all(bool(jnp.all(jnp.isfinite(value))) for value in info.values())
            )

    def test_log_kde_loss_matches_gaussian_leave_one_out_formula(self):
        predicted = jnp.array([[[0.0], [2.0]]])
        target = jnp.array([[[-1.0], [3.0]]])

        loss, info = dfm_log_kde_loss(predicted, target, bandwidth=1.0)

        expected_log_p = jnp.logaddexp(-0.5, -4.5) - jnp.log(2.0)
        expected_log_q = -2.0
        self.assertTrue(
            bool(jnp.allclose(info["per_group_log_p"], expected_log_p))
        )
        self.assertTrue(
            bool(jnp.allclose(info["per_group_log_q"], expected_log_q))
        )
        self.assertTrue(
            bool(jnp.allclose(loss, expected_log_q - expected_log_p))
        )

        gradient = jax.grad(
            lambda samples: dfm_log_kde_loss(samples, target, 1.0)[0].sum()
        )(predicted)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_complete_update_is_finite(self):
        for drift_backend in ("drift_loss", "log_kde", "sinkhorn"):
            with self.subTest(drift_backend=drift_backend):
                agent = make_agent(
                    drift_backend=drift_backend
                )
                updated_agent, info = agent.update(make_batch())

                self.assertEqual(
                    int(updated_agent.network.step), int(agent.network.step) + 1
                )
                self.assertTrue(bool(jnp.isfinite(info["critic/critic_loss"])))
                self.assertTrue(bool(jnp.isfinite(info["actor/actor_loss"])))
                self.assertTrue(bool(jnp.isfinite(info["grad/norm"])))

    def test_non_chunked_unbatched_sampling(self):
        agent = make_agent(action_chunking=False)
        action = agent.sample_actions(
            jnp.zeros((3,), dtype=jnp.float32), jax.random.PRNGKey(5)
        )

        self.assertEqual(action.shape, (2,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(action))))


if __name__ == "__main__":
    unittest.main()
