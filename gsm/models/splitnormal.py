"""Split-normal mixture kernel for MATLAB's asymmetric normal model."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp

from gsm.links import inverse_link
from gsm.models.gaussian import log_mixture_weights


@dataclass(frozen=True)
class SplitNormalMixtureParams:
    """Regression coefficients for a covariate-dependent split-normal mixture."""

    mean_coef: jnp.ndarray
    scale_coef: jnp.ndarray
    skewness_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: SplitNormalMixtureParams, X_mean, X_scale, X_skewness):
    """Return split-normal feature values for each observation and component."""

    mean = inverse_link(X_mean @ params.mean_coef.T, "identity")
    scale = inverse_link(X_scale @ params.scale_coef.T, "log")
    skewness = inverse_link(X_skewness @ params.skewness_coef.T, "log")
    return mean, _positive(scale), _positive(skewness)


def component_log_prob(y, mean, scale, skewness):
    """Asymmetric normal log density per observation and component."""

    y = jnp.asarray(y).reshape((-1, 1))
    left = y <= mean
    sign = jnp.where(left, 1.0, skewness**2)
    return (
        0.5 * jnp.log(2.0 / jnp.pi)
        - jnp.log1p(skewness)
        - jnp.log(scale)
        - ((y - mean) ** 2) / (2.0 * scale**2 * sign)
    )


def responsibilities(params: SplitNormalMixtureParams, y, X_mean, X_scale, X_skewness, Z):
    """Return posterior component responsibilities."""

    mean, scale, skewness = component_features(params, X_mean, X_scale, X_skewness)
    logits = (
        log_mixture_weights(params.gating_coef, Z)
        + component_log_prob(y, mean, scale, skewness)
    )
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(
    params: SplitNormalMixtureParams,
    y,
    X_mean,
    X_scale,
    X_skewness,
    Z,
):
    """Return pointwise marginal log likelihoods."""

    mean, scale, skewness = component_features(params, X_mean, X_scale, X_skewness)
    comp_lp = component_log_prob(y, mean, scale, skewness)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(params: SplitNormalMixtureParams, y, X_mean, X_scale, X_skewness, Z):
    """Marginal log likelihood with mixture allocations integrated out."""

    return jnp.sum(log_prob_observations(params, y, X_mean, X_scale, X_skewness, Z))


def predict_mean_variance(params: SplitNormalMixtureParams, X_mean, X_scale, X_skewness, Z):
    """Return mixture predictive mean and variance."""

    mean, scale, skewness = component_features(params, X_mean, X_scale, X_skewness)
    post_c = jnp.sqrt(2.0 / jnp.pi) * scale * (skewness - 1.0)
    component_mean = mean + post_c
    component_variance = scale**2 * (
        skewness + ((jnp.pi - 2.0) / jnp.pi) * (skewness - 1.0) ** 2
    )

    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * component_mean, axis=1)
    pred_second = jnp.sum(weights * (component_variance + component_mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance


def _positive(value):
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
