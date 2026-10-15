"""Public FEBAMA workflow helpers for precomputed log predictive densities."""

from dataclasses import dataclass
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
from jax.nn import logsumexp
import numpy as np

from gsm.febama.data import LpdFeatures, SeriesData
from gsm.febama.distributions import log_prob_matrix
from gsm.febama.features import compute_tsfeatures
from gsm.febama.inference import (
    FebamaMapResult,
    FebamaVbResult,
    fit_map,
    fit_vb,
    sample_febama_beta_posterior,
)
from gsm.febama.scoring import add_intercept, logscore, log_weights


@dataclass(frozen=True)
class FebamaFit:
    """Fitted FEBAMA gating model."""

    method: str
    beta: np.ndarray
    add_intercept: bool
    result: FebamaMapResult | FebamaVbResult
    posterior: object | None = None
    n_predictive_samples: int = 100
    seed: int = 123
    model_names: tuple[str, ...] | None = None
    feature_names: tuple[str, ...] | None = None
    feature_mean: np.ndarray | None = None
    feature_sd: np.ndarray | None = None


@dataclass(frozen=True)
class FebamaScore:
    """Predictive log-score summary for a fitted FEBAMA model."""

    total: float
    pointwise: np.ndarray
    n_posterior_samples: int = 0


def prepare_lpd_features(
    lpd,
    features,
    model_names=None,
    feature_names=None,
    response=None,
    origin=None,
    date=None,
    lead=None,
) -> LpdFeatures:
    """Build a validated ``LpdFeatures`` object from arrays."""

    return LpdFeatures(
        lpd=lpd,
        features=features,
        response=response,
        model_names=model_names,
        feature_names=feature_names,
        origin=origin,
        date=date,
        lead=lead,
    )


def compute_lpd_features(
    data,
    forecasters: Iterable[Callable],
    feature_names: Iterable[str] | None = None,
    *,
    model_names: Iterable[str] | None = None,
    start: int = 25,
    max_origins: int | None = None,
    horizon: int = 1,
    feature_window: int | None = None,
    frequency: int | None = 1,
    feature_function: Callable | None = None,
    precomputed_features=None,
    forecast_cache: dict | None = None,
) -> LpdFeatures:
    """Build marginal LPD rows for each rolling origin and horizon step.

    An optional caller-owned cache retains predictions and raw origin features
    for reuse in forecast metrics, without refitting experts.
    """

    y, dates = _series_values_and_dates(data)
    forecaster_tuple = tuple(forecasters)
    if len(forecaster_tuple) < 2:
        raise ValueError("at least two forecasters are required")
    names = _model_names(model_names, forecaster_tuple)
    feature_names_tuple = None if feature_names is None else tuple(feature_names)

    start = int(start)
    horizon = int(horizon)
    if start < 2:
        raise ValueError("start must leave at least two historical observations")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if feature_window is not None and int(feature_window) < 2:
        raise ValueError("feature_window must be at least two")
    if max_origins is not None and int(max_origins) < 1:
        raise ValueError("max_origins must be positive")

    stop = y.shape[0] - horizon + 1
    if max_origins is not None:
        stop = min(stop, start + int(max_origins))
    if stop <= start:
        raise ValueError("not enough observations for the requested rolling origins")

    origins = np.arange(start, stop, dtype=int)
    feature_matrix = _precomputed_feature_matrix(
        precomputed_features,
        origins.shape[0],
        feature_names_tuple,
    )
    if feature_matrix is not None and feature_names_tuple is None:
        feature_names_tuple = tuple(f"feature_{idx}" for idx in range(feature_matrix.shape[1]))
    if feature_matrix is None and feature_names_tuple is None:
        raise ValueError("feature_names are required when features are computed")

    lpd_rows = []
    feature_rows = []
    response_rows = []
    for row_idx, origin in enumerate(origins):
        history = y[:origin]
        actual = y[origin : origin + horizon]
        predictions = [forecaster(history, horizon) for forecaster in forecaster_tuple]
        lpd_rows.append(np.asarray(log_prob_matrix(actual, predictions), dtype=float))
        response_rows.extend(actual)

        if feature_matrix is None:
            feature_history = history if feature_window is None else history[-int(feature_window) :]
            feature_values = _compute_feature_row(
                feature_history,
                feature_names_tuple,
                feature_function,
                frequency,
            )
        else:
            feature_values = feature_matrix[row_idx]
        feature_rows.extend([feature_values] * horizon)
        if forecast_cache is not None:
            forecast_cache[int(origin)] = (tuple(predictions), np.asarray(feature_values).copy())

    return LpdFeatures(
        lpd=np.concatenate(lpd_rows, axis=0),
        features=np.asarray(feature_rows, dtype=float),
        response=np.asarray(response_rows, dtype=float),
        model_names=names,
        feature_names=feature_names_tuple,
        origin=np.repeat(origins, horizon),
        date=None if dates is None else tuple(
            dates[origin + step] for origin in origins for step in range(horizon)
        ),
        lead=np.tile(np.arange(1, horizon + 1), len(origins)),
    )


def fit_febama(
    lpd_features: LpdFeatures,
    fit_method: str = "map",
    add_intercept: bool = True,
    initial_beta=None,
    active_mask=None,
    coefficient_prior_scale: float = 10.0,
    max_iter: int = 1000,
    learning_rate: float = 1e-2,
    tol: float = 1e-6,
    n_elbo_samples: int = 8,
    n_restarts: int = 1,
    seed: int = 123,
    posterior_init_log_std: float = -5.0,
    n_validation_samples: int = 64,
    n_predictive_samples: int = 100,
    validation_interval: int = 25,
) -> FebamaFit:
    """Fit FEBAMA gating coefficients for precomputed component log densities."""

    if n_predictive_samples < 0:
        raise ValueError("n_predictive_samples must be non-negative")
    method = fit_method.lower()
    features = _features_for_model(lpd_features.features, add_intercept)
    if method == "map":
        result = fit_map(
            lpd_features.lpd,
            features,
            initial_beta=initial_beta,
            active_mask=active_mask,
            coefficient_prior_scale=coefficient_prior_scale,
            max_iter=max_iter,
        )
        posterior = None
    elif method == "vb":
        result = fit_vb(
            lpd_features.lpd,
            features,
            initial_beta=initial_beta,
            active_mask=active_mask,
            coefficient_prior_scale=coefficient_prior_scale,
            max_iter=max_iter,
            learning_rate=learning_rate,
            tol=tol,
            n_elbo_samples=n_elbo_samples,
            n_restarts=n_restarts,
            seed=seed,
            posterior_init_log_std=posterior_init_log_std,
            n_validation_samples=n_validation_samples,
            validation_interval=validation_interval,
        )
        posterior = result.posterior
    else:
        raise ValueError("fit_method must be 'map' or 'vb'")
    return FebamaFit(
        method=method,
        beta=result.beta,
        add_intercept=add_intercept,
        result=result,
        posterior=posterior,
        n_predictive_samples=n_predictive_samples,
        seed=seed,
        model_names=None if lpd_features.model_names is None else tuple(lpd_features.model_names),
        feature_names=None if lpd_features.feature_names is None else tuple(lpd_features.feature_names),
        feature_mean=None if lpd_features.feature_mean is None else lpd_features.feature_mean.copy(),
        feature_sd=None if lpd_features.feature_sd is None else lpd_features.feature_sd.copy(),
    )


def compute_weights(
    fit: FebamaFit,
    features,
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return posterior-mean weights for VB, or fitted weights for MAP.

    Explicitly passing ``n_samples=0`` requests weights at mean coefficients.
    """

    return np.exp(compute_log_weights(fit, features, n_samples=n_samples, seed=seed))


def compute_log_weights(fit, features, n_samples=None, seed=None):
    """Return log posterior-mean weights without probability underflow."""

    n_samples = fit.n_predictive_samples if n_samples is None else n_samples
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if fit.posterior is not None and n_samples > 0:
        samples = sample_log_weights(
            fit, features, n_samples=n_samples, seed=fit.seed if seed is None else seed,
        )
        return np.asarray(logsumexp(samples, axis=0) - np.log(n_samples))
    return np.asarray(log_weights(fit.beta, _model_features(fit, features)))


def sample_weights(
    fit: FebamaFit,
    features,
    n_samples: int = 100,
    seed: int = 123,
) -> np.ndarray:
    """Return posterior-sampled FEBAMA weights for a fitted VB model."""

    return np.exp(sample_log_weights(fit, features, n_samples=n_samples, seed=seed))


def sample_log_weights(fit, features, n_samples=100, seed=123):
    """Return posterior-sampled log weights using a reusable compiled kernel."""

    if fit.posterior is None:
        raise ValueError("sample_weights requires a VB fit with a posterior")
    matrix = _model_features(fit, features)
    beta_samples = sample_febama_beta_posterior(
        fit.posterior,
        seed=seed,
        n_samples=n_samples,
    )
    return np.asarray(_sample_log_weights(jnp.asarray(beta_samples), jnp.asarray(matrix)))


_sample_log_weights = jax.jit(jax.vmap(log_weights, in_axes=(0, None)))


@jax.jit
def _sample_log_scores(beta_samples, features, lpd):
    log_w = _sample_log_weights(beta_samples, features)
    return logsumexp(log_w + lpd[None, :, :], axis=2)


def score_febama(
    lpd_features: LpdFeatures,
    fit: FebamaFit,
    n_samples: int | None = None,
    seed: int | None = None,
) -> FebamaScore:
    """Score the posterior predictive density, or MAP/plugin when requested."""

    features = _model_features(fit, lpd_features)
    order = _column_order(lpd_features.model_names, fit.model_names, "model")
    lpd = lpd_features.lpd if order is None else lpd_features.lpd[:, order]
    n_samples = fit.n_predictive_samples if n_samples is None else n_samples
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if fit.posterior is not None and n_samples > 0:
        beta_samples = sample_febama_beta_posterior(
            fit.posterior, seed=fit.seed if seed is None else seed, n_samples=n_samples,
        )
        scores = _sample_log_scores(jnp.asarray(beta_samples), jnp.asarray(features), jnp.asarray(lpd))
        pointwise = np.asarray(logsumexp(scores, axis=0) - np.log(n_samples))
    else:
        pointwise = np.asarray(logscore(lpd, features, fit.beta, sum=False))
        n_samples = 0
    return FebamaScore(
        total=float(np.sum(pointwise)), pointwise=pointwise, n_posterior_samples=n_samples,
    )


def _features_for_model(features, should_add_intercept: bool):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must be finite")
    if should_add_intercept:
        return np.asarray(add_intercept(features))
    return features


def _column_order(names, reference, label):
    if reference is None:
        return None
    if names is None or len(set(names)) != len(names) or set(names) != set(reference):
        raise ValueError(f"{label} names must match the fitted model")
    return [tuple(names).index(name) for name in reference]


def _model_features(fit, features):
    if isinstance(features, LpdFeatures):
        order = _column_order(features.feature_names, fit.feature_names, "feature")
        _check_scaling(fit, features, order)
        features = features.features if order is None else features.features[:, order]
    return _features_for_model(features, fit.add_intercept)


def _check_scaling(fit, data, order=None):
    for name in ("feature_mean", "feature_sd"):
        expected, actual = getattr(fit, name), getattr(data, name)
        if expected is not None and actual is not None:
            actual = actual if order is None else actual[order]
            if not np.allclose(expected, actual, rtol=1e-12, atol=1e-12):
                raise ValueError("feature scaling must match the fitted model")


def _series_values_and_dates(data) -> tuple[np.ndarray, tuple[str, ...] | None]:
    if isinstance(data, SeriesData):
        values = np.asarray(data.x, dtype=float).reshape((-1,))
        dates = data.date
    else:
        values = np.asarray(data, dtype=float).reshape((-1,))
        dates = None
    if values.size < 3:
        raise ValueError("data must contain at least three observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values")
    if dates is not None and len(dates) != values.shape[0]:
        raise ValueError("date length must match data length")
    return values, dates


def _model_names(model_names, forecasters) -> tuple[str, ...]:
    if model_names is not None:
        names = tuple(model_names)
        if len(names) != len(forecasters):
            raise ValueError("model_names length must match forecasters")
        return names
    return tuple(
        getattr(forecaster, "__name__", f"model_{idx}")
        for idx, forecaster in enumerate(forecasters)
    )


def _precomputed_feature_matrix(features, n_rows: int, feature_names):
    if features is None:
        return None
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("precomputed_features must be a 2D matrix")
    if matrix.shape[0] != n_rows:
        raise ValueError("precomputed_features rows must match rolling origins")
    if feature_names is not None and len(feature_names) != matrix.shape[1]:
        raise ValueError("feature_names length must match precomputed feature columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("precomputed_features must be finite")
    return matrix


def _compute_feature_row(history, feature_names, feature_function, frequency) -> np.ndarray:
    if feature_function is None:
        values = compute_tsfeatures(history, frequency=frequency, feature_names=feature_names)
    else:
        values = feature_function(history)
    if isinstance(values, dict):
        missing = [name for name in feature_names if name not in values]
        if missing:
            raise ValueError(f"missing feature values: {missing}")
        return np.asarray([values[name] for name in feature_names], dtype=float)
    row = np.asarray(values, dtype=float).reshape((-1,))
    if row.shape != (len(feature_names),):
        raise ValueError("feature_function output length must match feature_names")
    return row
