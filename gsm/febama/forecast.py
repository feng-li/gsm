"""One-step FEBAMA forecasting and forecast metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.special import logsumexp

from gsm.febama.api import compute_weights, sample_weights
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
) -> FebamaForecast:
    """Produce a one-step FEBAMA forecast from fitted gating coefficients."""

    horizon = int(horizon)
    if horizon != 1:
        raise ValueError("only horizon=1 is currently implemented")
    n_weight_samples = int(n_weight_samples)
    if n_weight_samples < 0:
        raise ValueError("n_weight_samples must be nonnegative")
    y, xx, dates = _series_parts(data)
    forecaster_tuple = tuple(forecasters)
    if len(forecaster_tuple) < 2:
        raise ValueError("at least two forecasters are required")
    if lpd_features.feature_mean is None or lpd_features.feature_sd is None:
        raise ValueError("lpd_features must be cleaned before forecasting")

    raw_feature_names = (
        tuple(feature_names) if feature_names is not None else lpd_features.feature_names
    )
    if raw_feature_names is None:
        raise ValueError("feature_names are required")

    history = y if feature_window is None else y[-int(feature_window) :]
    raw_features = _feature_row(history, raw_feature_names, feature_function, frequency)
    scaled_features = standardize_features(
        raw_features,
        lpd_features.feature_mean,
        lpd_features.feature_sd,
        feature_names=raw_feature_names,
        reference_feature_names=lpd_features.feature_names,
    )
    weights = compute_weights(fit, scaled_features)

    predictions = tuple(forecaster(y, horizon) for forecaster in forecaster_tuple)
    means = np.column_stack(
        [np.asarray(prediction.mean(), dtype=float) for prediction in predictions]
    )
    forecast = np.sum(weights * means, axis=1)
    weight_samples = None
    forecast_samples = None
    log_score_samples = None
    if n_weight_samples > 0 and getattr(fit, "posterior", None) is not None:
        weight_samples = sample_weights(
            fit,
            scaled_features,
            n_samples=n_weight_samples,
            seed=seed,
        )
        forecast_samples = np.sum(weight_samples * means[None, :, :], axis=2)

    actual = None if xx is None else xx[:horizon]
    if xx is not None and actual.shape[0] != horizon:
        raise ValueError("data.xx must contain at least horizon observations")
    lpd = None
    log_score = None
    mase_value = None
    smape_value = None
    if actual is not None:
        lpd = np.asarray(log_prob_matrix(actual, predictions), dtype=float)
        pointwise = logsumexp(np.log(weights) + lpd, axis=1)
        log_score = float(np.sum(pointwise))
        if weight_samples is not None:
            sample_pointwise = logsumexp(
                np.log(weight_samples) + lpd[None, :, :],
                axis=2,
            )
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
        date=dates,
        weight_samples=weight_samples,
        forecast_samples=forecast_samples,
        log_score_samples=log_score_samples,
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
