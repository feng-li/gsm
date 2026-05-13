"""Split-t mixture kernel for MATLAB's asymmetric Student-t model."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp
from jax.scipy.special import betaln

from gsm.links import inverse_link
from gsm.models.mixture import log_mixture_weights


@dataclass(frozen=True)
class SplitTMixtureParams:
    """Regression coefficients for a covariate-dependent split-t mixture."""

    mean_coef: jnp.ndarray
    df_coef: jnp.ndarray
    scale_coef: jnp.ndarray
    skewness_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: SplitTMixtureParams, X_mean, X_df, X_scale, X_skewness):
    """Return split-t feature values for each observation and component."""

    mean = inverse_link(X_mean @ params.mean_coef.T, "identity")
    df = inverse_link(X_df @ params.df_coef.T, "log")
    scale = inverse_link(X_scale @ params.scale_coef.T, "log")
    skewness = inverse_link(X_skewness @ params.skewness_coef.T, "log")
    return mean, _positive(df), _positive(scale), _positive(skewness)


def component_log_prob(y, mean, df, scale, skewness):
    """Asymmetric Student-t log density per observation and component."""

    y = jnp.asarray(y).reshape((-1, 1))
    left = y <= mean
    sign = jnp.where(left, 1.0, skewness**2)
    standardized = ((y - mean) ** 2) / (scale**2 * sign)
    return (
        jnp.log(2.0)
        + 0.5 * (1.0 + df) * jnp.log(df / (df + standardized))
        - jnp.log(scale)
        - 0.5 * jnp.log(df)
        - betaln(0.5 * df, 0.5)
        - jnp.log1p(skewness)
    )


def responsibilities(params: SplitTMixtureParams, y, X_mean, X_df, X_scale, X_skewness, Z):
    """Return posterior component responsibilities."""

    mean, df, scale, skewness = component_features(params, X_mean, X_df, X_scale, X_skewness)
    logits = (
        log_mixture_weights(params.gating_coef, Z)
        + component_log_prob(y, mean, df, scale, skewness)
    )
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(
    params: SplitTMixtureParams,
    y,
    X_mean,
    X_df,
    X_scale,
    X_skewness,
    Z,
):
    """Return pointwise marginal log likelihoods."""

    mean, df, scale, skewness = component_features(params, X_mean, X_df, X_scale, X_skewness)
    comp_lp = component_log_prob(y, mean, df, scale, skewness)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(params: SplitTMixtureParams, y, X_mean, X_df, X_scale, X_skewness, Z):
    """Marginal log likelihood with mixture allocations integrated out."""

    return jnp.sum(log_prob_observations(params, y, X_mean, X_df, X_scale, X_skewness, Z))


def predict_mean_variance(params: SplitTMixtureParams, X_mean, X_df, X_scale, X_skewness, Z):
    """Return mixture predictive mean and variance when component moments exist."""

    mean, df, scale, skewness = component_features(params, X_mean, X_df, X_scale, X_skewness)
    beta_term = jnp.exp(betaln(0.5 * df, 0.5))
    post_c = 2.0 * jnp.sqrt(df) * scale * (skewness - 1.0) / ((df - 1.0) * beta_term)
    component_mean = jnp.where(df > 1.0, mean + post_c, jnp.nan)
    component_variance = (
        ((1.0 + skewness**3) / (1.0 + skewness))
        * (df / (df - 2.0))
        * scale**2
        - post_c**2
    )
    component_variance = jnp.where(df > 2.0, component_variance, jnp.nan)

    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * component_mean, axis=1)
    pred_second = jnp.sum(weights * (component_variance + component_mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance


def _positive(value):
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
