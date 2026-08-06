"""Grouped Sinkhorn drift used by Drift Flow Matching.

This is intentionally separate from ``utils.drift_loss``.  DFM uses two
independent entropic-transport plans (prediction-to-target and
prediction-to-prediction), whereas the legacy drift loss combines samples and
applies a geometric-mean affinity plus force rescaling.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def pairwise_quadratic_cost(x, y):
    """Return C(x, y) = 0.5 * ||x - y||^2 over the final feature axis."""
    x_sq = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    y_sq = jnp.sum(jnp.square(y), axis=-1, keepdims=True)
    squared_distance = x_sq + jnp.swapaxes(y_sq, -1, -2)
    squared_distance = squared_distance - 2.0 * jnp.einsum(
        "...id,...jd->...ij", x, y
    )
    return 0.5 * jnp.maximum(squared_distance, 0.0)


def sinkhorn_log_plan_from_logits(
    logits,
    row_marginal,
    col_marginal,
    sinkhorn_iters,
):
    """Apply truncated alternating Sinkhorn projections in log space.

    One iteration is one marginal projection, matching the convention in the
    Sinkhorn-Drifting paper: odd iterations normalize rows, even iterations
    normalize columns, and ``sinkhorn_iters=1`` recovers row-normalized drift.
    Leading dimensions on ``logits`` are treated as batch dimensions.
    """
    tiny = jnp.finfo(logits.dtype).tiny
    log_row_marginal = jnp.log(jnp.maximum(row_marginal, tiny))
    log_col_marginal = jnp.log(jnp.maximum(col_marginal, tiny))

    # Add singleton leading axes so the marginal vectors broadcast over every
    # plan batch (including the positive/negative and DFM group axes).
    row_shape = (1,) * (logits.ndim - 2) + (logits.shape[-2], 1)
    col_shape = (1,) * (logits.ndim - 2) + (1, logits.shape[-1])
    log_row_marginal = jnp.reshape(log_row_marginal, row_shape)
    log_col_marginal = jnp.reshape(log_col_marginal, col_shape)

    def project_rows(log_plan):
        return log_plan + log_row_marginal - logsumexp(
            log_plan, axis=-1, keepdims=True
        )

    def project_cols(log_plan):
        return log_plan + log_col_marginal - logsumexp(
            log_plan, axis=-2, keepdims=True
        )

    def project(iteration, log_plan):
        return jax.lax.cond(
            iteration % 2 == 0,
            project_rows,
            project_cols,
            log_plan,
        )

    return jax.lax.fori_loop(0, sinkhorn_iters, project, logits)


@partial(jax.jit, static_argnames=("sinkhorn_iters",))
def grouped_sinkhorn_drift(
    x,
    pos,
    neg,
    temp_pos,
    temp_neg,
    sinkhorn_iters,
):
    """Compute the DFM grouped cross-minus-self Sinkhorn drift.

    Args:
        x: Query/predicted states with shape ``[G, B, D]``.
        pos: Positive target states with shape ``[G, B, D]``.
        neg: Negative predicted states with shape ``[G, B, D]``.
        temp_pos: Positive Gibbs-kernel temperature.
        temp_neg: Negative Gibbs-kernel temperature.
        sinkhorn_iters: Number of alternating marginal projections.

    Returns:
        A tuple ``(drift, weights_pos, weights_neg)``.  The drift has shape
        ``[G, B, D]`` and each weight matrix has shape ``[G, B, B]``.
    """
    batch_size = x.shape[-2]
    dtype = x.dtype
    marginal = jnp.full((batch_size,), 1.0 / batch_size, dtype=dtype)

    logits_pos = -pairwise_quadratic_cost(x, pos) / jnp.asarray(
        temp_pos, dtype=dtype
    )
    logits_neg = -pairwise_quadratic_cost(x, neg) / jnp.asarray(
        temp_neg, dtype=dtype
    )

    # Both plans have the same shape, so solve them together.  This preserves
    # their independent normalizations while avoiding a second Sinkhorn loop.
    logits = jnp.stack((logits_pos, logits_neg), axis=0)
    log_plans = sinkhorn_log_plan_from_logits(
        logits,
        marginal,
        marginal,
        sinkhorn_iters,
    )

    # Algorithm 2 row-normalizes the truncated transport plans before taking
    # their barycentric projections.  Softmax(log(plan)) is the stable form.
    weights_pos, weights_neg = jax.nn.softmax(log_plans, axis=-1)
    drift_pos = jnp.einsum("...ij,...jd->...id", weights_pos, pos)
    drift_neg = jnp.einsum("...ij,...jd->...id", weights_neg, neg)
    return drift_pos - drift_neg, weights_pos, weights_neg
