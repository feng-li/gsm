"""Beta-binomial mixture kernel for MATLAB's BetaBin model."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp
from jax.scipy.special import gammaln

from gsm.links import inverse_link
from gsm.models.mixture import log_mixture_weights


@dataclass(frozen=True)
class BetaBinMixtureParams:
    """Regression coefficients for a covariate-dependent beta-binomial mixture."""

    mean_coef: jnp.ndarray
    dispersion_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params: BetaBinMixtureParams, X_mean, X_dispersion):
    """Return success probabilities and precision values for each component."""

    probability = inverse_link(X_mean @ params.mean_coef.T, "logit")
    dispersion = inverse_link(X_dispersion @ params.dispersion_coef.T, "log")
    return _probability(probability), _positive(dispersion)


def component_log_prob(y, probability, dispersion):
    """Beta-binomial log PMF per observation and component.

    The response convention is ``y[:, 0]`` successes and ``y[:, 1]`` trials.
    ``dispersion`` follows MATLAB's ``phi`` precision parameterization.
    """

    successes, trials = _split_response(y)
    probability = _probability(probability)
    dispersion = _positive(dispersion)
    failures = trials - successes
    alpha = dispersion * probability
    beta = dispersion * (1.0 - probability)
    log_density = (
        gammaln(trials + 1.0)
        - gammaln(failures + 1.0)
        - gammaln(successes + 1.0)
        + gammaln(successes + alpha)
        + gammaln(failures + beta)
        + gammaln(dispersion)
        - gammaln(alpha)
        - gammaln(beta)
        - gammaln(trials + dispersion)
    )
    return jnp.where(_support(successes, trials), log_density, -jnp.inf)


def responsibilities(params: BetaBinMixtureParams, y, X_mean, X_dispersion, Z):
    """Return posterior component responsibilities."""

    probability, dispersion = component_features(params, X_mean, X_dispersion)
    logits = log_mixture_weights(params.gating_coef, Z) + component_log_prob(
        y, probability, dispersion
    )
    return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def log_prob_observations(params: BetaBinMixtureParams, y, X_mean, X_dispersion, Z):
    """Return pointwise marginal log likelihoods."""

    probability, dispersion = component_features(params, X_mean, X_dispersion)
    comp_lp = component_log_prob(y, probability, dispersion)
    log_w = log_mixture_weights(params.gating_coef, Z)
    return logsumexp(log_w + comp_lp, axis=1)


def log_prob(*args):
    """Return either a simple beta-binomial log PMF sum or mixture log likelihood."""

    if isinstance(args[0], BetaBinMixtureParams):
        params, y, X_mean, X_dispersion, Z = args
        return jnp.sum(log_prob_observations(params, y, X_mean, X_dispersion, Z))
    y, probability, dispersion = args
    return jnp.sum(component_log_prob(y, probability, dispersion))


def predict_mean_variance(params: BetaBinMixtureParams, X_mean, X_dispersion, Z, trials=None):
    """Return mixture predictive success-count mean and variance."""

    probability, dispersion = component_features(params, X_mean, X_dispersion)
    trials = _prediction_trials(probability.shape[0], trials, probability.dtype)
    component_mean = trials * probability
    component_variance = (
        trials
        * probability
        * (1.0 - probability)
        * (trials + dispersion)
        / (1.0 + dispersion)
    )
    weights = jnp.exp(log_mixture_weights(params.gating_coef, Z))
    pred_mean = jnp.sum(weights * component_mean, axis=1)
    pred_second = jnp.sum(weights * (component_variance + component_mean**2), axis=1)
    pred_variance = pred_second - pred_mean**2
    return pred_mean, pred_variance


def _split_response(y):
    y = jnp.asarray(y)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("beta-binomial response must have columns [successes, trials]")
    successes = y[:, 0:1]
    trials = y[:, 1:2]
    return successes, trials


def _support(successes, trials):
    return (
        (successes >= 0.0)
        & (trials >= 0.0)
        & (successes <= trials)
        & jnp.isclose(successes, jnp.round(successes))
        & jnp.isclose(trials, jnp.round(trials))
    )


def _prediction_trials(n_obs: int, trials, dtype):
    if trials is None:
        return jnp.ones((n_obs, 1), dtype=dtype)
    trials = jnp.asarray(trials, dtype=dtype)
    if trials.ndim == 2 and trials.shape[1] == 2:
        trials = trials[:, 1:2]
    else:
        trials = trials.reshape((-1, 1))
    return trials


def _probability(value):
    value = jnp.asarray(value)
    eps = jnp.finfo(value.dtype).eps
    return jnp.clip(value, eps, 1.0 - eps)


def _positive(value):
    value = jnp.asarray(value)
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)
