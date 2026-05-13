"""Beta-regression mixture kernel for MATLAB's BetaReg model."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp
from jax.scipy.special import betaln

from gsm.links import inverse_link
from gsm.models.mixture import log_mixture_weights


@dataclass(frozen=True)
class BetaRegMixtureParams:
    """Regression coefficients for a covariate-dependent beta-regression mixture."""

    mean_coef: jnp.ndarray
    dispersion_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: BetaRegMixtureParams, X_mean, X_dispersion):
    """Return mean and precision/dispersion values for each observation."""

    mean = inverse_link(X_mean @ params.mean_coef.T, "logit")
    dispersion = inverse_link(X_dispersion @ params.dispersion_coef.T, "log")
    return _unit_interval(mean), _positive(dispersion)


def component_log_prob(y, mean, dispersion):
    """Beta log density per observation and component.

    MATLAB parameterizes BetaReg as alpha = phi * mu and
    beta = phi * (1 - mu), where ``phi`` is the dispersion feature.
    """

    y = jnp.asarray(y).reshape((-1, 1))
    safe_y = jnp.where(y == 0.0, 0.001, jnp.where(y == 1.0, 0.999, y))
    safe_y = jnp.clip(
        safe_y,
        jnp.finfo(y.dtype).tiny,
        1.0 - jnp.finfo(y.dtype).eps,
    )
    alpha = _positive(dispersion * mean)
    beta = _positive(dispersion * (1.0 - mean))
    log_density = (
        (alpha - 1.0) * jnp.log(safe_y)
        + (beta - 1.0) * jnp.log1p(-safe_y)
        - betaln(alpha, beta)
    )
    return jnp.where((y >= 0.0) & (y <= 1.0), log_density, -jnp.inf)


def responsibilities(params: BetaRegMixtureParams, y, X_mean, X_dispersion, Z):
    """Return posterior component responsibilities."""

    mean, dispersion = component_features(params, X_mean, X_dispersion)
    logits = (
        log_mixture_weights(params.gating_coef, Z)
        + component_log_prob(y, mean, dispersion)
    )
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(params: BetaRegMixtureParams, y, X_mean, X_dispersion, Z):
    """Return pointwise marginal log likelihoods."""

    mean, dispersion = component_features(params, X_mean, X_dispersion)
    comp_lp = component_log_prob(y, mean, dispersion)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(params: BetaRegMixtureParams, y, X_mean, X_dispersion, Z):
    """Marginal log likelihood with mixture allocations integrated out."""

    return jnp.sum(log_prob_observations(params, y, X_mean, X_dispersion, Z))


def predict_mean_variance(params: BetaRegMixtureParams, X_mean, X_dispersion, Z):
    """Return mixture predictive mean and variance."""

    mean, dispersion = component_features(params, X_mean, X_dispersion)
    component_variance = mean * (1.0 - mean) / (1.0 + dispersion)
    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * mean, axis=1)
    pred_second = jnp.sum(weights * (component_variance + mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance


def _positive(value):
    value = jnp.asarray(value)
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)


def _unit_interval(value):
    value = jnp.asarray(value)
    return jnp.clip(value, jnp.finfo(value.dtype).tiny, 1.0 - jnp.finfo(value.dtype).eps)
