"""FEBAMA softmax gating and log predictive score kernels."""

import jax.numpy as jnp
from jax.nn import logsumexp


def add_intercept(features):
    """Prepend an intercept column to a feature matrix."""

    features = _as_feature_matrix(features)
    return jnp.concatenate([jnp.ones((features.shape[0], 1)), features], axis=1)


def linear_predictors(beta, features):
    """Return FEBAMA linear predictors with the last component as baseline."""

    features = _as_feature_matrix(features)
    beta = _as_beta_matrix(beta)
    if beta.shape[1] != features.shape[1]:
        raise ValueError("beta columns must match feature columns")
    eta = features @ beta.T
    baseline = jnp.zeros((features.shape[0], 1), dtype=eta.dtype)
    return jnp.concatenate([eta, baseline], axis=1)


def log_weights(beta, features):
    """Return log softmax weights for all components."""

    eta = linear_predictors(beta, features)
    return eta - logsumexp(eta, axis=1, keepdims=True)


def logscore(lpd, features, beta, sum: bool = True):
    """Return FEBAMA log predictive score."""

    lpd = _as_lpd_matrix(lpd)
    log_w = log_weights(beta, features)
    if log_w.shape != lpd.shape:
        raise ValueError("lpd columns must equal beta rows plus one baseline")
    pointwise = logsumexp(log_w + lpd, axis=1)
    if sum:
        return jnp.sum(pointwise)
    return pointwise


def responsibilities(lpd, features, beta):
    """Return posterior model responsibilities for each observation."""

    lpd = _as_lpd_matrix(lpd)
    logits = log_weights(beta, features) + lpd
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def _as_feature_matrix(features):
    features = jnp.asarray(features)
    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if not jnp.issubdtype(features.dtype, jnp.floating):
        features = features.astype(jnp.float64)
    return features


def _as_lpd_matrix(lpd):
    lpd = jnp.asarray(lpd)
    if lpd.ndim != 2:
        raise ValueError("lpd must be a 2D matrix")
    if lpd.shape[1] < 2:
        raise ValueError("lpd must contain at least two component columns")
    if not jnp.issubdtype(lpd.dtype, jnp.floating):
        lpd = lpd.astype(jnp.float64)
    return lpd


def _as_beta_matrix(beta):
    beta = jnp.asarray(beta)
    if beta.ndim == 1:
        beta = beta.reshape((1, -1))
    if beta.ndim != 2:
        raise ValueError("beta must be a 1D or 2D coefficient array")
    if beta.shape[0] < 1:
        raise ValueError("beta must contain one row per non-baseline component")
    if not jnp.issubdtype(beta.dtype, jnp.floating):
        beta = beta.astype(jnp.float64)
    return beta
