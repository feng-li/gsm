"""Exponential-hazard interval survival mixture kernel.

This reimplements the likelihood shape from the historical
``GSMPython/Models/ExpHazard.py`` file using the current small JAX model style.
Each row represents an interval ``(t0, t]`` with an event indicator.  The
component hazard is constant over the interval and log-linked to covariates.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp

from gsm.links import inverse_link
from gsm.models.gaussian import log_mixture_weights


@dataclass(frozen=True)
class ExpHazardMixtureParams:
    """Regression coefficients for a covariate-dependent hazard mixture."""

    rate_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: ExpHazardMixtureParams, X_rate):
    """Return positive hazard rates for each observation and component."""

    rate = inverse_link(X_rate @ params.rate_coef.T, "log")
    return _positive(rate)


def component_log_prob(t, t0, event, rate):
    """Interval log likelihood per observation and component.

    ``event=0`` means the subject survived the whole interval, and ``event=1``
    means the event occurred in the interval.
    """

    t = _as_column(t)
    t0 = _as_column(t0)
    event = _as_float_column(event)
    increment = _positive(rate) * (t - t0)
    safe_increment = jnp.maximum(increment, 0.0)
    log_survived = -safe_increment
    log_event = _log1mexp(safe_increment)
    log_density = jnp.where(event > 0.5, log_event, log_survived)
    support = _interval_support(t, t0, event)
    return jnp.where(support, log_density, -jnp.inf)


def responsibilities(params: ExpHazardMixtureParams, t, t0, event, X_rate, Z):
    """Return posterior component responsibilities."""

    rate = component_features(params, X_rate)
    logits = log_mixture_weights(params.gating_coef, Z) + component_log_prob(
        t,
        t0,
        event,
        rate,
    )
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(params: ExpHazardMixtureParams, t, t0, event, X_rate, Z):
    """Return pointwise marginal log likelihoods."""

    rate = component_features(params, X_rate)
    comp_lp = component_log_prob(t, t0, event, rate)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(*args):
    """Return either a simple interval log likelihood sum or mixture log likelihood."""

    if isinstance(args[0], ExpHazardMixtureParams):
        params, t, t0, event, X_rate, Z = args
        return jnp.sum(log_prob_observations(params, t, t0, event, X_rate, Z))
    t, t0, event, rate = args
    return jnp.sum(component_log_prob(t, t0, event, rate))


def predict_survival_probability(params: ExpHazardMixtureParams, t, t0, X_rate, Z):
    """Return mixture probability of surviving interval ``(t0, t]``."""

    rate = component_features(params, X_rate)
    increment = rate * (_as_column(t) - _as_column(t0))
    survival = jnp.exp(-jnp.maximum(increment, 0.0))
    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    return jnp.sum(weights * survival, axis=1)


def _interval_support(t, t0, event):
    event_is_binary = (
        (event >= 0.0)
        & (event <= 1.0)
        & jnp.isclose(event, jnp.round(event))
    )
    return (t >= t0) & (t0 >= 0.0) & event_is_binary


def _log1mexp(x):
    """Stable ``log(1 - exp(-x))`` for non-negative ``x``."""

    x = jnp.maximum(jnp.asarray(x), 0.0)
    cutoff = jnp.log(2.0)
    return jnp.where(
        x <= cutoff,
        jnp.log(-jnp.expm1(-x)),
        jnp.log1p(-jnp.exp(-x)),
    )


def _as_column(value):
    return jnp.asarray(value).reshape((-1, 1))


def _as_float_column(value):
    return _as_column(value).astype(jnp.result_type(value, jnp.float32))


def _positive(value):
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.floating):
        value = value.astype(jnp.float32)
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
