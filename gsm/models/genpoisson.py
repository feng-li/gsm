"""Generalized Poisson mixture kernels for GenPois and GenPoisAlt."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp
from jax.scipy.special import gammaln

from gsm.links import inverse_link
from gsm.models.gaussian import log_mixture_weights


@dataclass(frozen=True)
class GenPoissonMixtureParams:
    """Regression coefficients for generalized Poisson mixtures."""

    mean_coef: jnp.ndarray
    dispersion_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(
    params: GenPoissonMixtureParams,
    X_mean,
    X_dispersion,
    parameterization: str = "standard",
):
    """Return mean and dispersion values for each observation and component."""

    mean = inverse_link(X_mean @ params.mean_coef.T, "log")
    dispersion_link = "log1" if parameterization == "alternative" else "log"
    dispersion = inverse_link(X_dispersion @ params.dispersion_coef.T, dispersion_link)
    return _positive(mean), _positive(dispersion)


def component_log_prob(y, mean, dispersion, parameterization: str = "standard"):
    """Generalized Poisson log PMF per observation and component."""

    y = jnp.asarray(y).reshape((-1, 1))
    mean = _positive(mean)
    dispersion = _positive(dispersion)
    if parameterization == "alternative":
        log_density = _alternative_log_prob(y, mean, dispersion)
        extra_support = mean + (dispersion - 1.0) * y > 0.0
    else:
        log_density = _standard_log_prob(y, mean, dispersion)
        extra_support = 1.0 + dispersion * y > 0.0
    support = (y >= 0.0) & jnp.isclose(y, jnp.round(y)) & extra_support
    return jnp.where(support, log_density, -jnp.inf)


def responsibilities(
    params: GenPoissonMixtureParams,
    y,
    X_mean,
    X_dispersion,
    Z,
    parameterization: str = "standard",
):
    """Return posterior component responsibilities."""

    mean, dispersion = component_features(params, X_mean, X_dispersion, parameterization)
    logits = log_mixture_weights(params.gating_coef, Z) + component_log_prob(
        y,
        mean,
        dispersion,
        parameterization=parameterization,
    )
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(
    params: GenPoissonMixtureParams,
    y,
    X_mean,
    X_dispersion,
    Z,
    parameterization: str = "standard",
):
    """Return pointwise marginal log likelihoods."""

    mean, dispersion = component_features(params, X_mean, X_dispersion, parameterization)
    comp_lp = component_log_prob(
        y,
        mean,
        dispersion,
        parameterization=parameterization,
    )
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(*args, parameterization: str = "standard"):
    """Return either a simple GenPois log PMF sum or mixture log likelihood."""

    if isinstance(args[0], GenPoissonMixtureParams):
        params, y, X_mean, X_dispersion, Z = args
        return jnp.sum(
            log_prob_observations(
                params,
                y,
                X_mean,
                X_dispersion,
                Z,
                parameterization=parameterization,
            )
        )
    y, mean, dispersion = args
    return jnp.sum(component_log_prob(y, mean, dispersion, parameterization))


def predict_mean_variance(
    params: GenPoissonMixtureParams,
    X_mean,
    X_dispersion,
    Z,
    parameterization: str = "standard",
):
    """Return mixture predictive mean and variance."""

    mean, dispersion = component_features(params, X_mean, X_dispersion, parameterization)
    if parameterization == "alternative":
        component_variance = mean * dispersion**2
    else:
        component_variance = mean * (1.0 + mean * dispersion) ** 2
    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * mean, axis=1)
    pred_second = jnp.sum(weights * (component_variance + mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance


def _standard_log_prob(y, mean, dispersion):
    return (
        y * jnp.log(mean)
        - y * jnp.log1p(dispersion * mean)
        + (y - 1.0) * jnp.log1p(dispersion * y)
        - mean * (1.0 + dispersion * y) / (1.0 + dispersion * mean)
        - gammaln(y + 1.0)
    )


def _alternative_log_prob(y, mean, dispersion):
    return (
        jnp.log(mean)
        + (y - 1.0) * jnp.log(mean + (dispersion - 1.0) * y)
        - y * jnp.log(dispersion)
        - (mean + (dispersion - 1.0) * y) / dispersion
        - gammaln(y + 1.0)
    )


def _positive(value):
    value = jnp.asarray(value)
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
