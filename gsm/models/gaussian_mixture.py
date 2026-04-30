"""Small JAX sketch for MATLAB's HeteroGauss smooth-mixture model.

The MATLAB setting has two component features:

- Mean, with identity link.
- Variance, with log link.

This module only sketches the kernel needed by a future VB implementation: compute
component features, multinomial-logit mixture weights, and the marginal mixture
log likelihood.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp

from gsm.links import inverse_link


@dataclass(frozen=True)
class GaussianMixtureParams:
    """Regression coefficients for a covariate-dependent Gaussian mixture."""

    mean_coef: jnp.ndarray
    log_variance_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: GaussianMixtureParams, X_mean, X_variance):
    """Return per-observation mean and variance for each component.

    Shapes:
    - ``mean_coef``: ``(n_components, n_cov_mean)``
    - ``log_variance_coef``: ``(n_components, n_cov_variance)``
    - output arrays: ``(n_obs, n_components)``
    """

    mean = inverse_link(X_mean @ params.mean_coef.T, "identity")
    variance = inverse_link(X_variance @ params.log_variance_coef.T, "log")
    return mean, variance


def log_mixture_weights(gating_coef, Z):
    """Reference-class multinomial-logit weights.

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


def component_log_prob(y, mean, variance):
    """Gaussian log density per observation and component."""

    y = jnp.asarray(y).reshape((-1, 1))
    return -0.5 * (jnp.log(2.0 * jnp.pi * variance) + ((y - mean) ** 2) / variance)


def responsibilities(params: GaussianMixtureParams, y, X_mean, X_variance, Z):
    """Return variational/posterior component responsibilities."""

    mean, variance = component_features(params, X_mean, X_variance)
    logits = log_mixture_weights(params.gating_coef, Z) + component_log_prob(y, mean, variance)
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob(params: GaussianMixtureParams, y, X_mean, X_variance, Z):
    """Marginal log likelihood with mixture allocations integrated out."""

    mean, variance = component_features(params, X_mean, X_variance)
    comp_lp = component_log_prob(y, mean, variance)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return jnp.sum(logsumexp(log_w + comp_lp, axis=1))


def predict_mean_variance(params: GaussianMixtureParams, X_mean, X_variance, Z):
    """Return mixture predictive mean and variance."""

    mean, variance = component_features(params, X_mean, X_variance)
    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * mean, axis=1)
    pred_second = jnp.sum(weights * (variance + mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance
