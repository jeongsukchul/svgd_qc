import unittest

import jax
import jax.numpy as jnp
import optax

from utils.optimizers import make_optimizer


class OptimizerTest(unittest.TestCase):
    def test_kron_step_is_finite(self):
        params = {
            "weight": jnp.array([[1.0, -2.0], [0.5, 3.0]]),
            "bias": jnp.array([0.25, -0.75]),
        }
        optimizer = make_optimizer("kron", learning_rate=1e-3)
        state = optimizer.init(params)

        updates, state = optimizer.update(params, state, params)
        new_params = optax.apply_updates(params, updates)

        self.assertTrue(
            all(
                bool(jnp.all(jnp.isfinite(value)))
                for value in jax.tree.leaves(new_params)
            )
        )

    def test_unknown_optimizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported optimizer"):
            make_optimizer("unknown", learning_rate=1e-3)


if __name__ == "__main__":
    unittest.main()
