"""Weibull interval survival mixture kernel.

This reimplements the likelihood shape from the historical
``GSMPython/Models/WeibullSurvival.py`` file using the current small JAX model
style.  Each row represents an interval ``(t0, t]`` with an event indicator.
Both the scale/rate and shape features are positive and log-linked.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp

from gsm.links import inverse_link
from gsm.models.exphazard import _as_column, _as_float_column, _log1mexp
from gsm.models.gaussian import log_mixture_weights


@dataclass(frozen=True)
class WeibullSurvivalMixtureParams:
    """Regression coefficients for a covariate-dependent Weibull survival mixture."""

    rate_coef: jnp.ndarray
    shape_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: WeibullSurvivalMixtureParams, X_rate, X_shape):
    """Return positive Weibull rate and shape values."""

    rate = inverse_link(X_rate @ params.rate_coef.T, "log")
    shape = inverse_link(X_shape @ params.shape_coef.T, "log")
    return _positive(rate), _positive(shape)


def component_log_prob(t, t0, event, rate, shape):
    """Interval log likelihood per observation and component."""

    t = _as_column(t)
    t0 = _as_column(t0)
    event = _as_float_column(event)
    increment = _cumulative_hazard_increment(t, t0, rate, shape)
    safe_increment = jnp.maximum(increment, 0.0)
    log_survived = -safe_increment
    log_event = _log1mexp(safe_increment)
    log_density = jnp.where(event > 0.5, log_event, log_survived)
    support = _interval_support(t, t0, event)
    return jnp.where(support, log_density, -jnp.inf)


def responsibilities(params, t, t0, event, X_rate, X_shape, Z):
    """Return posterior component responsibilities."""

    rate, shape = component_features(params, X_rate, X_shape)
    logits = log_mixture_weights(params.gating_coef, Z) + component_log_prob(
        t,
        t0,
        event,
        rate,
        shape,
    )
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(params, t, t0, event, X_rate, X_shape, Z):
    """Return pointwise marginal log likelihoods."""

    rate, shape = component_features(params, X_rate, X_shape)
    comp_lp = component_log_prob(t, t0, event, rate, shape)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(*args):
    """Return either a simple interval log likelihood sum or mixture log likelihood."""

    if isinstance(args[0], WeibullSurvivalMixtureParams):
        params, t, t0, event, X_rate, X_shape, Z = args
        return jnp.sum(log_prob_observations(params, t, t0, event, X_rate, X_shape, Z))
    t, t0, event, rate, shape = args
    return jnp.sum(component_log_prob(t, t0, event, rate, shape))


def predict_survival_probability(params, t, t0, X_rate, X_shape, Z):
    """Return mixture probability of surviving interval ``(t0, t]``."""

    rate, shape = component_features(params, X_rate, X_shape)
    increment = _cumulative_hazard_increment(_as_column(t), _as_column(t0), rate, shape)
    survival = jnp.exp(-jnp.maximum(increment, 0.0))
    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    return jnp.sum(weights * survival, axis=1)


def _cumulative_hazard_increment(t, t0, rate, shape):
    t = jnp.maximum(t, 0.0)
    t0 = jnp.maximum(t0, 0.0)
    rate = _positive(rate)
    shape = _positive(shape)
    return rate * (t**shape - t0**shape)


def _interval_support(t, t0, event):
    event_is_binary = (
        (event >= 0.0)
        & (event <= 1.0)
        & jnp.isclose(event, jnp.round(event))
    )
    return (t >= t0) & (t0 >= 0.0) & event_is_binary


def _positive(value):
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.floating):
        value = value.astype(jnp.float32)
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
