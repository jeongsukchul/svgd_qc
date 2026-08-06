import unittest

import jax
import jax.numpy as jnp

from utils.dfm_drift import grouped_sinkhorn_drift, pairwise_quadratic_cost


class GroupedSinkhornDriftTest(unittest.TestCase):
    def test_quadratic_cost_matches_paper_definition(self):
        x = jnp.array([[[0.0, 0.0], [1.0, 0.0]]])
        y = jnp.array([[[0.0, 2.0], [1.0, 1.0]]])

        cost = pairwise_quadratic_cost(x, y)

        expected = jnp.array([[[2.0, 1.0], [2.5, 0.5]]])
        self.assertTrue(bool(jnp.allclose(cost, expected)))

    def test_one_iteration_is_row_normalized_drift(self):
        x = jnp.array([[[0.0], [2.0], [4.0]]])
        pos = jnp.array([[[1.0], [3.0], [6.0]]])

        _, weights_pos, _ = grouped_sinkhorn_drift(
            x=x,
            pos=pos,
            neg=x,
            temp_pos=2.0,
            temp_neg=1.0,
            sinkhorn_iters=1,
        )

        expected = jax.nn.softmax(
            -pairwise_quadratic_cost(x, pos) / 2.0, axis=-1
        )
        self.assertTrue(bool(jnp.allclose(weights_pos, expected, atol=1e-6)))

    def test_equal_positive_and_negative_measures_have_zero_drift(self):
        samples = jnp.array(
            [
                [[-1.0, 0.0], [0.0, 0.5], [2.0, 1.0]],
                [[4.0, -2.0], [3.0, 0.0], [5.0, 2.0]],
            ]
        )

        drift, weights_pos, weights_neg = grouped_sinkhorn_drift(
            x=samples,
            pos=samples,
            neg=samples,
            temp_pos=1.0,
            temp_neg=1.0,
            sinkhorn_iters=3,
        )

        self.assertTrue(bool(jnp.allclose(drift, 0.0, atol=1e-6)))
        self.assertTrue(bool(jnp.allclose(weights_pos, weights_neg, atol=1e-6)))
        self.assertTrue(
            bool(jnp.allclose(weights_pos.sum(axis=-1), 1.0, atol=1e-6))
        )

    def test_additional_iterations_improve_column_balance(self):
        x = jnp.array([[[0.0], [0.1], [4.0], [8.0]]])
        pos = jnp.array([[[0.0], [2.0], [2.1], [9.0]]])

        _, weights_one, _ = grouped_sinkhorn_drift(
            x, pos, x, 0.5, 0.5, sinkhorn_iters=1
        )
        _, weights_three, _ = grouped_sinkhorn_drift(
            x, pos, x, 0.5, 0.5, sinkhorn_iters=3
        )
        error_one = jnp.mean(jnp.abs(weights_one.sum(axis=-2) - 1.0))
        error_three = jnp.mean(jnp.abs(weights_three.sum(axis=-2) - 1.0))

        self.assertLess(float(error_three), float(error_one))


if __name__ == "__main__":
    unittest.main()
