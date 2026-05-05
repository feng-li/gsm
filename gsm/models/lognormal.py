"""Lognormal mixture kernels for MATLAB's LogNorm and LogNormRep models."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp

from gsm.links import inverse_link
from gsm.models.gaussian import log_mixture_weights


@dataclass(frozen=True)
class LogNormalMixtureParams:
    """Regression coefficients for a covariate-dependent lognormal mixture."""

    mean_coef: jnp.ndarray
    scale_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: LogNormalMixtureParams, X_mean, X_scale):
    """Return positive mean/location and scale features for each component."""

    mean = inverse_link(X_mean @ params.mean_coef.T, "log")
    scale = inverse_link(X_scale @ params.scale_coef.T, "log")
    return _positive(mean), _positive(scale)


def component_log_prob(y, mean, scale):
    """LogNorm log density with ``log(y) ~ N(mean, scale**2)``."""

    y = jnp.asarray(y).reshape((-1, 1))
    safe_y = jnp.maximum(y, jnp.finfo(y.dtype).tiny)
    log_y = jnp.log(safe_y)
    log_density = (
        -0.5 * ((log_y - mean) / scale) ** 2
        - jnp.log(scale)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_y
    )
    return jnp.where(y > 0.0, log_density, -jnp.inf)


def component_log_prob_reparameterized(y, mean, scale):
    """LogNormRep log density, parameterized by response-scale mean and sd."""

    location, log_scale = standard_lognormal_params(mean, scale)
    return component_log_prob(y, location, log_scale)


def standard_lognormal_params(mean, scale):
    """Convert response-scale mean/sd to standard lognormal location/scale."""

    mean = _positive(mean)
    scale = _positive(scale)
    ratio2 = (scale / mean) ** 2
    log_scale = jnp.sqrt(jnp.log1p(ratio2))
    location = jnp.log(mean) - 0.5 * log_scale**2
    return location, _positive(log_scale)


def responsibilities(
    params: LogNormalMixtureParams,
    y,
    X_mean,
    X_scale,
    Z,
    reparameterized: bool = False,
):
    """Return posterior component responsibilities."""

    mean, scale = component_features(params, X_mean, X_scale)
    comp_lp = (
        component_log_prob_reparameterized(y, mean, scale)
        if reparameterized
        else component_log_prob(y, mean, scale)
    )
    logits = log_mixture_weights(params.gating_coef, Z) + comp_lp
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(
    params: LogNormalMixtureParams,
    y,
    X_mean,
    X_scale,
    Z,
    reparameterized: bool = False,
):
    """Return pointwise marginal log likelihoods."""

    mean, scale = component_features(params, X_mean, X_scale)
    comp_lp = (
        component_log_prob_reparameterized(y, mean, scale)
        if reparameterized
        else component_log_prob(y, mean, scale)
    )
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(
    params: LogNormalMixtureParams,
    y,
    X_mean,
    X_scale,
    Z,
    reparameterized: bool = False,
):
    """Marginal log likelihood with mixture allocations integrated out."""

    return jnp.sum(
        log_prob_observations(params, y, X_mean, X_scale, Z, reparameterized)
    )


def predict_mean_variance(
    params: LogNormalMixtureParams,
    X_mean,
    X_scale,
    Z,
    reparameterized: bool = False,
):
    """Return mixture predictive mean and variance."""

    mean, scale = component_features(params, X_mean, X_scale)
    if reparameterized:
        component_mean = mean
        component_variance = scale**2
    else:
        component_mean = jnp.exp(mean + 0.5 * scale**2)
        component_variance = jnp.exp(2.0 * mean + scale**2) * jnp.expm1(scale**2)

    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * component_mean, axis=1)
    pred_second = jnp.sum(weights * (component_variance + component_mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance


def _positive(value):
    value = jnp.asarray(value)
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
