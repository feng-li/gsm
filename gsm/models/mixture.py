"""Shared finite-mixture helpers."""

import jax.numpy as jnp
from jax.nn import logsumexp


def log_mixture_weights(gating_coef, Z):
    """Reference-class multinomial-logit mixture weights.

    MATLAB identifies the first component by setting its gating coefficients to
    zero. ``gating_coef`` therefore has shape ``(n_components - 1, n_cov_mix)``.
    """

    if gating_coef.size == 0:
        return jnp.zeros((Z.shape[0], 1))

    scores = jnp.concatenate(
        [jnp.zeros((Z.shape[0], 1)), Z @ gating_coef.T],
        axis=1,
    )
    return scores - logsumexp(scores, axis=1, keepdims=True)
