"""Negative-binomial mixture kernel using MATLAB's mean/dispersion form."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp
from jax.scipy.special import gammaln

from gsm.links import inverse_link
from gsm.models.gaussian import log_mixture_weights


@dataclass(frozen=True)
class NegBinMixtureParams:
    """Regression coefficients for a covariate-dependent NegBin mixture."""

    mean_coef: jnp.ndarray
    dispersion_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: NegBinMixtureParams, X_mean, X_dispersion):
    """Return mean and dispersion values for each observation and component."""

    mean = inverse_link(X_mean @ params.mean_coef.T, "log")
    dispersion = inverse_link(X_dispersion @ params.dispersion_coef.T, "log")
    return _positive(mean), _positive(dispersion)


def component_log_prob(y, mean, dispersion):
    """Log PMF for ``y ~ NegBin(phi, phi / (phi + mu))``."""

    y = jnp.asarray(y).reshape((-1, 1))
    mu = _positive(mean)
    phi = _positive(dispersion)
    log_density = (
        gammaln(y + phi)
        - gammaln(phi)
        - gammaln(y + 1.0)
        + phi * (jnp.log(phi) - jnp.log(phi + mu))
        + y * (jnp.log(mu) - jnp.log(phi + mu))
    )
    support = (y >= 0.0) & jnp.isclose(y, jnp.round(y))
    return jnp.where(support, log_density, -jnp.inf)


def responsibilities(params: NegBinMixtureParams, y, X_mean, X_dispersion, Z):
    """Return posterior component responsibilities."""

    mean, dispersion = component_features(params, X_mean, X_dispersion)
    logits = (
        log_mixture_weights(params.gating_coef, Z)
        + component_log_prob(y, mean, dispersion)
    )
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(params: NegBinMixtureParams, y, X_mean, X_dispersion, Z):
    """Return pointwise marginal log likelihoods."""

    mean, dispersion = component_features(params, X_mean, X_dispersion)
    comp_lp = component_log_prob(y, mean, dispersion)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(*args):
    """Return either a simple NegBin log PMF sum or mixture log likelihood."""

    if isinstance(args[0], NegBinMixtureParams):
        params, y, X_mean, X_dispersion, Z = args
        return jnp.sum(log_prob_observations(params, y, X_mean, X_dispersion, Z))
    y, mean, dispersion = args
    return jnp.sum(component_log_prob(y, mean, dispersion))


def predict_mean_variance(params: NegBinMixtureParams, X_mean, X_dispersion, Z):
    """Return mixture predictive mean and variance."""

    mean, dispersion = component_features(params, X_mean, X_dispersion)
    component_variance = mean + mean**2 / dispersion
    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * mean, axis=1)
    pred_second = jnp.sum(weights * (component_variance + mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance


def _positive(value):
    value = jnp.asarray(value)
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
