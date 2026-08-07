import unittest

import jax
import jax.numpy as jnp

from agents.dfp import DFPAgent, dfp_log_kde_loss, get_config


def make_agent(**overrides):
    config = get_config()
    config.encoder = None
    config.horizon_length = 2
    config.actor_hidden_dims = (16, 16)
    config.value_hidden_dims = (16, 16)
    config.actor_num_samples = 4
    config.gen_per_label = 4
    config.alpha_pos = 1.0
    config.alpha_target = 1.0
    config.actor_ema_tau = 0.005
    for key, value in overrides.items():
        config[key] = value
    return DFPAgent.create(
        seed=0,
        ex_observations=jnp.zeros((3,), dtype=jnp.float32),
        ex_actions=jnp.zeros((2,), dtype=jnp.float32),
        config=config,
    )


def make_batch(batch_size=3):
    return {
        "observations": jnp.zeros((batch_size, 3), dtype=jnp.float32),
        "actions": jnp.zeros((batch_size, 2, 2), dtype=jnp.float32),
        "valid": jnp.ones((batch_size, 2), dtype=jnp.float32),
        "next_observations": jnp.zeros(
            (batch_size, 2, 3), dtype=jnp.float32
        ),
        "rewards": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "masks": jnp.ones((batch_size, 2), dtype=jnp.float32),
    }


class DFPLogKDETest(unittest.TestCase):
    def test_loss_matches_gaussian_leave_one_out_formula(self):
        generated = jnp.array([[[0.0], [2.0]]])
        positives = jnp.array([[[-1.0], [3.0]]])

        loss, info = dfp_log_kde_loss(
            generated, positives, bandwidth=1.0
        )

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
            lambda samples: dfp_log_kde_loss(samples, positives, 1.0)[0].sum()
        )(generated)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_actor_loss_supports_both_backends(self):
        for drift_backend in ("drift_loss", "log_kde"):
            with self.subTest(drift_backend=drift_backend):
                agent = make_agent(drift_backend=drift_backend)
                loss, info = agent.actor_loss(
                    make_batch(), agent.network.params, jax.random.PRNGKey(1)
                )

                self.assertTrue(bool(jnp.isfinite(loss)))
                if drift_backend == "log_kde":
                    self.assertTrue(bool(jnp.isfinite(info["log_kde_loss"])))
                    self.assertIn("log_kde_log_p", info)
                    self.assertIn("log_kde_log_q", info)
                else:
                    self.assertIn("drift_scale", info)

    def test_noise_scale_controls_sampled_actor_noise(self):
        agent = make_agent(noise_scale=0.25)
        observations = jnp.zeros((3, 3), dtype=jnp.float32)
        rng = jax.random.PRNGKey(12)

        noises = agent.sample_noises(observations, rng)
        expected = jax.random.normal(rng, (3, 4)) * 0.25

        self.assertTrue(bool(jnp.allclose(noises, expected)))

    def test_log_kde_complete_update_is_finite(self):
        agent = make_agent(drift_backend="log_kde")

        updated_agent, info = agent.update(make_batch())

        self.assertEqual(
            int(updated_agent.network.step), int(agent.network.step) + 1
        )
        self.assertTrue(bool(jnp.isfinite(info["actor/actor_loss"])))
        self.assertTrue(bool(jnp.isfinite(info["actor/log_kde_loss"])))
        self.assertTrue(bool(jnp.isfinite(info["grad/norm"])))

    def test_invalid_log_kde_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "drift_backend"):
            make_agent(drift_backend="unknown")
        with self.assertRaisesRegex(ValueError, "log_kde_bandwidth"):
            make_agent(drift_backend="log_kde", log_kde_bandwidth=0.0)
        with self.assertRaisesRegex(ValueError, "gen_per_label"):
            make_agent(drift_backend="log_kde", gen_per_label=1)
        with self.assertRaisesRegex(ValueError, "noise_scale"):
            make_agent(noise_scale=-0.1)


if __name__ == "__main__":
    unittest.main()
