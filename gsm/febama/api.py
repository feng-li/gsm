"""Public FEBAMA workflow helpers for precomputed log predictive densities."""

from dataclasses import dataclass
from typing import Callable, Iterable

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


@dataclass(frozen=True)
class FebamaScore:
    """Predictive log-score summary for a fitted FEBAMA model."""

    total: float
    pointwise: np.ndarray


def prepare_lpd_features(
    lpd,
    features,
    model_names=None,
    feature_names=None,
    response=None,
    origin=None,
    date=None,
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
) -> LpdFeatures:
    """Build rolling component LPDs and features for FEBAMA fitting."""

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
        lpd_rows.append(np.asarray(log_prob_matrix(actual, predictions), dtype=float).sum(axis=0))
        response_rows.append(actual[0] if horizon == 1 else actual)

        if feature_matrix is None:
            feature_history = history if feature_window is None else history[-int(feature_window) :]
            feature_values = _compute_feature_row(
                feature_history,
                feature_names_tuple,
                feature_function,
                frequency,
            )
            feature_rows.append(feature_values)
        else:
            feature_rows.append(feature_matrix[row_idx])

    return LpdFeatures(
        lpd=np.asarray(lpd_rows, dtype=float),
        features=np.asarray(feature_rows, dtype=float),
        response=np.asarray(response_rows, dtype=float),
        model_names=names,
        feature_names=feature_names_tuple,
        origin=origins,
        date=None if dates is None else tuple(dates[idx] for idx in origins),
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
) -> FebamaFit:
    """Fit FEBAMA gating coefficients for precomputed component log densities."""

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
    )


def compute_weights(fit: FebamaFit, features) -> np.ndarray:
    """Return FEBAMA softmax weights for a fitted model."""

    matrix = _features_for_model(_raw_features(features), fit.add_intercept)
    return np.asarray(np.exp(log_weights(fit.beta, matrix)))


def sample_weights(
    fit: FebamaFit,
    features,
    n_samples: int = 100,
    seed: int = 123,
) -> np.ndarray:
    """Return posterior-sampled FEBAMA weights for a fitted VB model."""

    if fit.posterior is None:
        raise ValueError("sample_weights requires a VB fit with a posterior")
    matrix = _features_for_model(_raw_features(features), fit.add_intercept)
    beta_samples = sample_febama_beta_posterior(
        fit.posterior,
        seed=seed,
        n_samples=n_samples,
    )
    return np.asarray([np.exp(log_weights(beta, matrix)) for beta in beta_samples])


def score_febama(lpd_features: LpdFeatures, fit: FebamaFit) -> FebamaScore:
    """Score precomputed component log densities under fitted FEBAMA weights."""

    features = _features_for_model(lpd_features.features, fit.add_intercept)
    pointwise = np.asarray(logscore(lpd_features.lpd, features, fit.beta, sum=False))
    return FebamaScore(total=float(np.sum(pointwise)), pointwise=pointwise)


def _features_for_model(features, should_add_intercept: bool):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if should_add_intercept:
        return np.asarray(add_intercept(features))
    return features


def _raw_features(features):
    if isinstance(features, LpdFeatures):
        return features.features
    return features


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
