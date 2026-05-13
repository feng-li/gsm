"""Gamma mixture kernels for Gamma and GammaRep models."""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax.nn import logsumexp
from jax.scipy.special import gammaln

from gsm.links import inverse_link
from gsm.models.mixture import log_mixture_weights

GammaParameterization = Literal["mean_variance", "shape_scale"]


@dataclass(frozen=True)
class GammaMixtureParams:
    """Regression coefficients for a covariate-dependent gamma mixture."""

    mean_coef: jnp.ndarray
    variance_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: GammaMixtureParams, X_mean, X_variance):
    """Return positive feature values for each observation and component."""

    mean = inverse_link(X_mean @ params.mean_coef.T, "log")
    variance = inverse_link(X_variance @ params.variance_coef.T, "log")
    return _positive(mean), _positive(variance)


def component_log_prob(
    y,
    mean,
    variance,
    parameterization: GammaParameterization = "mean_variance",
):
    """Gamma log density under the selected feature parameterization."""

    shape, scale = to_standard_gamma_params(mean, variance, parameterization)
    return _standard_gamma_log_prob(y, shape, scale)


def _standard_gamma_log_prob(y, shape, scale):
    """Log density for ``y ~ Gamma(shape, scale)``."""

    y = jnp.asarray(y).reshape((-1, 1))
    safe_y = jnp.maximum(y, jnp.finfo(y.dtype).tiny)
    log_density = (
        (shape - 1.0) * jnp.log(safe_y)
        - safe_y / scale
        - shape * jnp.log(scale)
        - gammaln(shape)
    )
    return jnp.where(y > 0.0, log_density, -jnp.inf)


def to_standard_gamma_params(
    mean,
    variance,
    parameterization: GammaParameterization,
):
    """Map Gamma or GammaRep features to shape and scale."""

    mean = _positive(mean)
    variance = _positive(variance)
    if parameterization == "mean_variance":
        shape = mean**2 / variance
        scale = variance / mean
    elif parameterization == "shape_scale":
        shape = mean
        scale = variance
    else:
        raise ValueError(f"unknown gamma parameterization: {parameterization}")
    return _positive(shape), _positive(scale)


def responsibilities(
    params: GammaMixtureParams,
    y,
    X_mean,
    X_variance,
    Z,
    parameterization: GammaParameterization = "mean_variance",
):
    """Return posterior component responsibilities."""

    mean, variance = component_features(params, X_mean, X_variance)
    comp_lp = component_log_prob(y, mean, variance, parameterization)
    logits = log_mixture_weights(params.gating_coef, Z) + comp_lp
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(
    params: GammaMixtureParams,
    y,
    X_mean,
    X_variance,
    Z,
    parameterization: GammaParameterization = "mean_variance",
):
    """Return pointwise marginal log likelihoods."""

    mean, variance = component_features(params, X_mean, X_variance)
    comp_lp = component_log_prob(y, mean, variance, parameterization)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(
    params: GammaMixtureParams,
    y,
    X_mean,
    X_variance,
    Z,
    parameterization: GammaParameterization = "mean_variance",
):
    """Marginal log likelihood with mixture allocations integrated out."""

    return jnp.sum(
        log_prob_observations(params, y, X_mean, X_variance, Z, parameterization)
    )


def predict_mean_variance(
    params: GammaMixtureParams,
    X_mean,
    X_variance,
    Z,
    parameterization: GammaParameterization = "mean_variance",
):
    """Return mixture predictive mean and variance."""

    mean, variance = component_features(params, X_mean, X_variance)
    if parameterization == "mean_variance":
        component_mean = mean
        component_variance = variance
    elif parameterization == "shape_scale":
        component_mean = mean * variance
        component_variance = mean * variance**2
    else:
        raise ValueError(f"unknown gamma parameterization: {parameterization}")

    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * component_mean, axis=1)
    pred_second = jnp.sum(weights * (component_variance + component_mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance


def _positive(value):
    value = jnp.asarray(value)
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
