"""Poisson mixture kernel for MATLAB's Pois model."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp
from jax.scipy.special import gammaln

from gsm.links import inverse_link
from gsm.models.gaussian import log_mixture_weights


@dataclass(frozen=True)
class PoissonMixtureParams:
    """Regression coefficients for a covariate-dependent Poisson mixture."""

    mean_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def features_from_coefficients(beta, X, link_type: str | float = "log"):
    return inverse_link(X @ beta, link_type)


def component_features(params: PoissonMixtureParams, X_mean):
    """Return Poisson mean values for each observation and component."""

    mean = inverse_link(X_mean @ params.mean_coef.T, "log")
    return _positive(mean)


def component_log_prob(y, mean):
    """Poisson log PMF per observation and component."""

    y = jnp.asarray(y).reshape((-1, 1))
    mean = _positive(mean)
    log_density = y * jnp.log(mean) - mean - gammaln(y + 1.0)
    support = (y >= 0.0) & jnp.isclose(y, jnp.round(y))
    return jnp.where(support, log_density, -jnp.inf)


def responsibilities(params: PoissonMixtureParams, y, X_mean, Z):
    """Return posterior component responsibilities."""

    mean = component_features(params, X_mean)
    logits = log_mixture_weights(params.gating_coef, Z) + component_log_prob(y, mean)
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(params: PoissonMixtureParams, y, X_mean, Z):
    """Return pointwise marginal log likelihoods."""

    mean = component_features(params, X_mean)
    comp_lp = component_log_prob(y, mean)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(*args):
    """Return either a simple Poisson log PMF sum or mixture log likelihood."""

    if isinstance(args[0], PoissonMixtureParams):
        params, y, X_mean, Z = args
        return jnp.sum(log_prob_observations(params, y, X_mean, Z))
    y = jnp.asarray(args[0]).reshape((-1,))
    mean = jnp.asarray(args[1]).reshape((-1, 1))
    return jnp.sum(component_log_prob(y, mean))


def predict_mean_variance(params: PoissonMixtureParams, X_mean, Z):
    """Return mixture predictive mean and variance."""

    mean = component_features(params, X_mean)
    component_variance = mean
    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * mean, axis=1)
    pred_second = jnp.sum(weights * (component_variance + mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance


def _positive(value):
    value = jnp.asarray(value)
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
