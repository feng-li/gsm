"""FEBAMA forecasting and forecast metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.special import logsumexp

from gsm.febama.api import (
    _check_scaling,
    _column_order,
    compute_log_weights,
    sample_log_weights,
)
from gsm.febama.data import LpdFeatures, SeriesData
from gsm.febama.distributions import PredictiveDistribution, log_prob_matrix
from gsm.febama.features import compute_tsfeatures, standardize_features


@dataclass(frozen=True)
class FebamaForecast:
    """One FEBAMA forecast result."""

    forecast: np.ndarray
    weights: np.ndarray
    log_score: float | None
    mase: float | None
    smape: float | None
    lpd: np.ndarray | None
    features: np.ndarray
    predictions: tuple[PredictiveDistribution, ...]
    date: tuple[str, ...] | None = None
    weight_samples: np.ndarray | None = None
    forecast_samples: np.ndarray | None = None
    log_score_samples: np.ndarray | None = None


@dataclass(frozen=True)
class FebamaPerformance:
    """Aggregate performance summary for FEBAMA forecast results."""

    n_forecasts: int
    n_scored_forecasts: int
    total_log_score: float | None
    mean_log_score: float | None
    mean_mase: float | None
    mean_smape: float | None
    total_log_score_samples: np.ndarray | None = None
    mean_log_score_samples: np.ndarray | None = None


def forecast_febama(
    data,
    fit,
    lpd_features: LpdFeatures,
    forecasters: Iterable[Callable],
    feature_names: Iterable[str] | None = None,
    *,
    horizon: int = 1,
    feature_window: int | None = None,
    frequency: int | None = 1,
    feature_function: Callable | None = None,
    n_weight_samples: int = 0,
    seed: int = 123,
    model_names: Iterable[str] | None = None,
    predictions: Iterable[PredictiveDistribution] | None = None,
    precomputed_features=None,
) -> FebamaForecast:
    """Combine expert horizon marginals using features known at the origin.

    Gating features are held fixed across the horizon. Future forecast means
    are never appended to the observed series or used to refit expert models.
    """

    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    n_weight_samples = int(n_weight_samples)
    if n_weight_samples < 0:
        raise ValueError("n_weight_samples must be nonnegative")
    if feature_window is not None and int(feature_window) < 2:
        raise ValueError("feature_window must be at least two")
    y, xx, dates = _series_parts(data)
    forecaster_tuple = tuple(forecasters)
    if len(forecaster_tuple) < 2:
        raise ValueError("at least two forecasters are required")
    if lpd_features.feature_mean is None or lpd_features.feature_sd is None:
        raise ValueError("lpd_features must be cleaned before forecasting")
    feature_order = _column_order(lpd_features.feature_names, fit.feature_names, "feature")
    _check_scaling(fit, lpd_features, feature_order)
    names = tuple(model_names) if model_names is not None else tuple(
        getattr(forecaster, "__name__", "") for forecaster in forecaster_tuple
    )
    model_order = _column_order(names, fit.model_names, "model")

    raw_feature_names = (
        tuple(feature_names) if feature_names is not None else lpd_features.feature_names
    )
    if raw_feature_names is None:
        raise ValueError("feature_names are required")

    history = y if feature_window is None else y[-int(feature_window):]
    if precomputed_features is None:
        raw_features = _feature_row(history, raw_feature_names, feature_function, frequency)
    else:
        raw_features = np.asarray(precomputed_features, dtype=float).reshape((1, -1))
        if raw_features.shape[1] != len(raw_feature_names):
            raise ValueError("precomputed_features columns must match feature_names")
    scaled_origin = standardize_features(
        raw_features, lpd_features.feature_mean, lpd_features.feature_sd,
        feature_names=raw_feature_names,
        reference_feature_names=lpd_features.feature_names,
    )
    scaled_features = np.repeat(scaled_origin, horizon, axis=0)
    if feature_order is not None:
        scaled_features = scaled_features[:, feature_order]
    predictions = (tuple(forecaster(y, horizon) for forecaster in forecaster_tuple)
                   if predictions is None else tuple(predictions))
    if len(predictions) != len(forecaster_tuple):
        raise ValueError("predictions must match the number of forecasters")
    if model_order is not None:
        predictions = tuple(predictions[idx] for idx in model_order)
    means = np.column_stack([_horizon_means(prediction, horizon) for prediction in predictions])
    weight_samples = None
    forecast_samples = None
    log_score_samples = None
    if n_weight_samples > 0 and getattr(fit, "posterior", None) is not None:
        log_weight_samples = sample_log_weights(
            fit,
            scaled_features,
            n_samples=n_weight_samples,
            seed=seed,
        )
        weight_samples = np.exp(log_weight_samples)
        log_w = logsumexp(log_weight_samples, axis=0) - np.log(n_weight_samples)
        weights = np.exp(log_w)
        forecast_samples = np.sum(weight_samples * means[None, :, :], axis=2)
    else:
        log_w = compute_log_weights(fit, scaled_features, seed=seed)
        weights = np.exp(log_w)
    forecast = np.sum(weights * means, axis=1)

    actual = None if xx is None else xx[:horizon]
    if xx is not None and actual.shape[0] != horizon:
        raise ValueError("data.xx must contain at least horizon observations")
    lpd = None
    log_score = None
    mase_value = None
    smape_value = None
    if actual is not None:
        lpd = np.asarray(log_prob_matrix(actual, predictions), dtype=float)
        pointwise = logsumexp(log_w + lpd, axis=1)
        log_score = float(np.sum(pointwise))
        if weight_samples is not None:
            sample_pointwise = logsumexp(log_weight_samples + lpd[None, :, :], axis=2)
            log_score_samples = np.sum(sample_pointwise, axis=1)
        mase_value = mase(actual, forecast, y)
        smape_value = smape(actual, forecast)

    return FebamaForecast(
        forecast=np.asarray(forecast, dtype=float),
        weights=np.asarray(weights, dtype=float),
        log_score=log_score,
        mase=mase_value,
        smape=smape_value,
        lpd=lpd,
        features=scaled_features,
        predictions=predictions,
        date=None if dates is None else dates[:horizon],
        weight_samples=weight_samples,
        forecast_samples=forecast_samples,
        log_score_samples=log_score_samples,
    )


def summarize_performance(forecasts: Iterable[FebamaForecast]) -> FebamaPerformance:
    """Summarize log-score, MASE, and sMAPE over FEBAMA forecast results."""

    forecast_tuple = tuple(forecasts)
    if not forecast_tuple:
        raise ValueError("at least one forecast is required")

    scored = [forecast for forecast in forecast_tuple if forecast.log_score is not None]
    if scored:
        log_scores = np.asarray([forecast.log_score for forecast in scored], dtype=float)
        total_log_score = float(np.sum(log_scores))
        mean_log_score = float(np.mean(log_scores))
    else:
        total_log_score = None
        mean_log_score = None
    total_samples, mean_samples = _sample_log_score_summary(scored)

    return FebamaPerformance(
        n_forecasts=len(forecast_tuple),
        n_scored_forecasts=len(scored),
        total_log_score=total_log_score,
        mean_log_score=mean_log_score,
        mean_mase=_mean_finite_metric(forecast_tuple, "mase"),
        mean_smape=_mean_finite_metric(forecast_tuple, "smape"),
        total_log_score_samples=total_samples,
        mean_log_score_samples=mean_samples,
    )


def mase(actual, forecast, insample) -> float:
    """Mean absolute scaled error using one-step naive in-sample scaling."""

    actual = np.asarray(actual, dtype=float).reshape((-1,))
    forecast = np.asarray(forecast, dtype=float).reshape((-1,))
    insample = np.asarray(insample, dtype=float).reshape((-1,))
    if actual.shape != forecast.shape:
        raise ValueError("actual and forecast must have the same shape")
    if insample.shape[0] < 2:
        return float("nan")
    scale = np.mean(np.abs(np.diff(insample)))
    if not np.isfinite(scale) or scale <= 0.0:
        return float("nan")
    return float(np.mean(np.abs(actual - forecast)) / scale)


def smape(actual, forecast) -> float:
    """Symmetric mean absolute percentage error on the 0-200 scale."""

    actual = np.asarray(actual, dtype=float).reshape((-1,))
    forecast = np.asarray(forecast, dtype=float).reshape((-1,))
    if actual.shape != forecast.shape:
        raise ValueError("actual and forecast must have the same shape")
    denom = np.abs(actual) + np.abs(forecast)
    values = np.zeros_like(denom, dtype=float)
    np.divide(
        200.0 * np.abs(actual - forecast),
        denom,
        out=values,
        where=denom > 0.0,
    )
    return float(np.mean(values))


def _series_parts(data) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...] | None]:
    if isinstance(data, SeriesData):
        y = np.asarray(data.x, dtype=float).reshape((-1,))
        xx = None if data.xx is None else np.asarray(data.xx, dtype=float).reshape((-1,))
        dates = _forecast_dates(data.date, y.shape[0], None if xx is None else xx.shape[0])
    else:
        y = np.asarray(data, dtype=float).reshape((-1,))
        xx = None
        dates = None
    if y.size < 1:
        raise ValueError("data must contain at least one in-sample observation")
    if not np.all(np.isfinite(y)):
        raise ValueError("data must contain only finite in-sample observations")
    if xx is not None and not np.all(np.isfinite(xx)):
        raise ValueError("data.xx must contain only finite observations")
    return y, xx, dates


def _forecast_dates(
    dates: tuple[str, ...] | None,
    n_history: int,
    n_future: int | None,
) -> tuple[str, ...] | None:
    if dates is None or n_future is None:
        return None
    if len(dates) >= n_history + n_future:
        return tuple(dates[n_history : n_history + n_future])
    return None


def _feature_row(history, feature_names, feature_function, frequency) -> np.ndarray:
    if feature_function is None:
        values = compute_tsfeatures(history, frequency=frequency, feature_names=feature_names)
    else:
        values = feature_function(history)
    if isinstance(values, dict):
        missing = [name for name in feature_names if name not in values]
        if missing:
            raise ValueError(f"missing feature values: {missing}")
        row = [values[name] for name in feature_names]
    else:
        row = values
    row = np.asarray(row, dtype=float).reshape((1, -1))
    if row.shape[1] != len(feature_names):
        raise ValueError("feature_function output length must match feature_names")
    if not np.all(np.isfinite(row)):
        raise ValueError("forecast features must be finite")
    return row


def _mean_finite_metric(forecasts: tuple[FebamaForecast, ...], name: str) -> float | None:
    values = [
        float(value)
        for value in (getattr(forecast, name) for forecast in forecasts)
        if value is not None and np.isfinite(value)
    ]
    if not values:
        return None
    return float(np.mean(values))


def _sample_log_score_summary(
    scored: list[FebamaForecast],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not scored or any(forecast.log_score_samples is None for forecast in scored):
        return None, None
    samples = []
    for forecast in scored:
        sample = np.asarray(forecast.log_score_samples, dtype=float).reshape((-1,))
        if sample.size < 1:
            raise ValueError("log_score_samples must not be empty")
        if samples and sample.shape != samples[0].shape:
            raise ValueError("log_score_samples must have matching shape")
        samples.append(sample)
    matrix = np.vstack(samples)
    return np.sum(matrix, axis=0), np.mean(matrix, axis=0)


def _horizon_means(prediction: PredictiveDistribution, horizon: int) -> np.ndarray:
    values = np.asarray(prediction.mean(), dtype=float).reshape((-1,))
    if values.size == 1:
        values = np.repeat(values, horizon)
    if values.shape != (horizon,):
        raise ValueError("forecaster means must match the forecast horizon")
    if not np.all(np.isfinite(values)):
        raise ValueError("forecaster means must be finite")
    return values
