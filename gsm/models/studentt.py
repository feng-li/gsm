"""Symmetric Student-t mixture kernel for MATLAB's studT model."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp
from jax.scipy.special import betaln

from gsm.links import inverse_link
from gsm.models.gaussian import log_mixture_weights


@dataclass(frozen=True)
class StudentTMixtureParams:
    """Regression coefficients for a covariate-dependent Student-t mixture."""

    mean_coef: jnp.ndarray
    df_coef: jnp.ndarray
    scale_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: StudentTMixtureParams, X_mean, X_df, X_scale):
    """Return Student-t feature values for each observation and component."""

    mean = inverse_link(X_mean @ params.mean_coef.T, "identity")
    df = inverse_link(X_df @ params.df_coef.T, "log")
    scale = inverse_link(X_scale @ params.scale_coef.T, "log")
    return mean, _positive(df), _positive(scale)


def component_log_prob(y, mean, df, scale):
    """Student-t log density per observation and component."""

    y = jnp.asarray(y).reshape((-1, 1))
    standardized = ((y - mean) ** 2) / (scale**2)
    return (
        0.5 * (1.0 + df) * jnp.log(df / (df + standardized))
        - jnp.log(scale)
        - 0.5 * jnp.log(df)
        - betaln(0.5 * df, 0.5)
    )


def responsibilities(params: StudentTMixtureParams, y, X_mean, X_df, X_scale, Z):
    """Return posterior component responsibilities."""

    mean, df, scale = component_features(params, X_mean, X_df, X_scale)
    logits = (
        log_mixture_weights(params.gating_coef, Z)
        + component_log_prob(y, mean, df, scale)
    )
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(
    params: StudentTMixtureParams,
    y,
    X_mean,
    X_df,
    X_scale,
    Z,
):
    """Return pointwise marginal log likelihoods."""

    mean, df, scale = component_features(params, X_mean, X_df, X_scale)
    comp_lp = component_log_prob(y, mean, df, scale)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(params: StudentTMixtureParams, y, X_mean, X_df, X_scale, Z):
    """Marginal log likelihood with mixture allocations integrated out."""

    return jnp.sum(log_prob_observations(params, y, X_mean, X_df, X_scale, Z))


def predict_mean_variance(params: StudentTMixtureParams, X_mean, X_df, X_scale, Z):
    """Return mixture predictive mean and variance when component moments exist."""

    mean, df, scale = component_features(params, X_mean, X_df, X_scale)
    component_mean = mean
    component_variance = jnp.where(df > 2.0, df / (df - 2.0) * scale**2, jnp.nan)

    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * component_mean, axis=1)
    pred_second = jnp.sum(weights * (component_variance + component_mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance


def _positive(value):
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
