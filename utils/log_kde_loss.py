"""Gaussian leave-one-out log-KDE objectives."""

import jax
import jax.numpy as jnp

def cdist(x, y, eps=1e-8):
    """Pairwise L2 distance: [B, N, D] x [B, M, D] -> [B, N, M]."""
    xydot = jnp.einsum("bnd,bmd->bnm", x, y)
    xnorms = jnp.einsum("bnd,bnd->bn", x, x)
    ynorms = jnp.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return jnp.clip(sq_dist, a_min=eps)

def grouped_log_kde_loss(generated, positives, bandwidth):
    """Estimate ``E_q[log q_KDE - log p_KDE]`` for each sample group.

    The generated samples remain differentiable KDE queries.  Both KDE center
    sets are detached, and the generated-density estimate excludes the query's
    own center, as in Algorithm 1 of arXiv:2604.06333.

    Args:
        generated: Generated samples with shape ``[..., num_generated, dim]``.
        positives: Target samples with shape ``[..., num_positive, dim]``.
        bandwidth: Isotropic Gaussian kernel bandwidth.

    Returns:
        A per-group loss and per-group mean log-density diagnostics.
    """
    negative_centers = jax.lax.stop_gradient(generated)
    positive_centers = jax.lax.stop_gradient(positives)
    logits_pos = -cdist(generated, positive_centers) / bandwidth
    logits_neg = -cdist(generated, negative_centers) / bandwidth

    num_positive = positives.shape[-2]
    num_negative = generated.shape[-2]
    logits_neg = jnp.where(
        jnp.eye(num_negative, dtype=jnp.bool_), -jnp.inf, logits_neg
    )

    dtype = generated.dtype
    log_p = jax.nn.logsumexp(logits_pos, axis=-1) - jnp.log(
        jnp.asarray(num_positive, dtype=dtype)
    )
    log_q = jax.nn.logsumexp(logits_neg, axis=-1) - jnp.log(
        jnp.asarray(num_negative - 1, dtype=dtype)
    )
    per_group_loss = jnp.mean(log_q - log_p, axis=-1)
    info = {
        "per_group_log_p": jnp.mean(log_p, axis=-1),
        "per_group_log_q": jnp.mean(log_q, axis=-1),
    }
    return per_group_loss, info
