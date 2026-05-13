"""Base forecasting adapters for FEBAMA."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
from scipy.stats import norm

from gsm.febama.distributions import PredictiveDistribution


def naive_fore(y, horizon: int, interval_level: float = 90.0) -> PredictiveDistribution:
    """Naive Gaussian forecast with the last observation as the mean."""

    y = _as_series(y)
    horizon = _validate_horizon(horizon)
    steps = np.arange(1, horizon + 1, dtype=float)
    sd = _innovation_sd(y) * np.sqrt(steps)
    mean = np.full(horizon, y[-1], dtype=float)
    return _gaussian_prediction(mean, sd)


def rw_drift_fore(y, horizon: int, interval_level: float = 90.0) -> PredictiveDistribution:
    """Random-walk-with-drift Gaussian forecast."""

    y = _as_series(y)
    horizon = _validate_horizon(horizon)
    steps = np.arange(1, horizon + 1, dtype=float)
    drift = 0.0 if y.size < 2 else (y[-1] - y[0]) / (y.size - 1)
    innovations = np.diff(y) - drift if y.size > 1 else np.asarray([0.0])
    sd = _residual_sd(innovations) * np.sqrt(steps)
    mean = y[-1] + steps * drift
    return _gaussian_prediction(mean, sd)


def ets_fore(
    y,
    horizon: int,
    interval_level: float = 90.0,
    season_length: int = 1,
    model: str = "ZZZ",
    **model_kwargs,
) -> PredictiveDistribution:
    """AutoETS Gaussian forecast using ``statsforecast``."""

    model_obj = _statsforecast_model(
        "AutoETS",
        season_length=season_length,
        model=model,
        **model_kwargs,
    )
    return _statsforecast_prediction(model_obj, y, horizon, interval_level)


def auto_arima_fore(
    y,
    horizon: int,
    interval_level: float = 90.0,
    season_length: int = 1,
    **model_kwargs,
) -> PredictiveDistribution:
    """AutoARIMA Gaussian forecast using ``statsforecast``."""

    model_obj = _statsforecast_model(
        "AutoARIMA",
        season_length=season_length,
        **model_kwargs,
    )
    return _statsforecast_prediction(model_obj, y, horizon, interval_level)


def garch_fore(
    y,
    horizon: int,
    interval_level: float = 90.0,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    **fit_kwargs,
) -> PredictiveDistribution:
    """GARCH Gaussian forecast using ``arch``."""

    return _arch_forecast(y, horizon, vol="GARCH", p=p, q=q, dist=dist, **fit_kwargs)


def egarch_fore(
    y,
    horizon: int,
    interval_level: float = 90.0,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    simulations: int = 1000,
    **fit_kwargs,
) -> PredictiveDistribution:
    """EGARCH Gaussian forecast using ``arch``."""

    method = "analytic" if horizon == 1 else "simulation"
    return _arch_forecast(
        y,
        horizon,
        vol="EGARCH",
        p=p,
        q=q,
        dist=dist,
        method=method,
        simulations=simulations,
        **fit_kwargs,
    )


def _statsforecast_model(class_name: str, **kwargs):
    if find_spec("statsforecast") is None:
        raise ImportError(f"{class_name} forecaster requires the 'statsforecast' package")
    from statsforecast import models

    model_cls = getattr(models, class_name)
    return model_cls(**kwargs)


def _statsforecast_prediction(model, y, horizon, interval_level):
    y = _as_series(y)
    horizon = _validate_horizon(horizon)
    level = int(interval_level)
    out = model.forecast(y=y, h=horizon, level=[level])
    mean = np.asarray(out["mean"], dtype=float)
    sd = _interval_sd(mean, out.get(f"lo-{level}"), interval_level)
    return _gaussian_prediction(mean, sd)


def _arch_forecast(
    y,
    horizon,
    vol,
    p,
    q,
    dist,
    method="analytic",
    simulations=1000,
    **fit_kwargs,
):
    if find_spec("arch") is None:
        raise ImportError(f"{vol} forecaster requires the 'arch' package")
    from arch import arch_model

    y = _as_series(y)
    horizon = _validate_horizon(horizon)
    fit_options = {"disp": "off", "show_warning": False}
    fit_options.update(fit_kwargs)
    model = arch_model(
        y,
        mean="Constant",
        vol=vol,
        p=p,
        q=q,
        dist=dist,
        rescale=False,
    )
    result = model.fit(**fit_options)
    forecast = result.forecast(
        horizon=horizon,
        method=method,
        simulations=simulations,
        reindex=False,
    )
    mean = np.asarray(forecast.mean, dtype=float)[-1]
    variance = np.asarray(forecast.variance, dtype=float)[-1]
    return PredictiveDistribution(
        "gaussian",
        {"mean": mean, "variance": _positive(variance)},
    )


def _interval_sd(mean, lower, interval_level):
    if lower is None:
        return np.full_like(mean, np.finfo(float).tiny)
    lower = np.asarray(lower, dtype=float)
    denom = norm.ppf(1.0 - float(interval_level) / 100.0)
    if not np.isfinite(denom) or denom == 0.0:
        raise ValueError("interval_level must define a finite lower-tail quantile")
    return _positive_scale((lower - mean) / denom)


def _gaussian_prediction(mean, sd):
    return PredictiveDistribution(
        "gaussian",
        {"mean": np.asarray(mean, dtype=float), "sd": _positive_scale(sd)},
    )


def _as_series(y):
    array = np.asarray(y, dtype=float).reshape((-1,))
    if array.size < 1:
        raise ValueError("time series must contain at least one observation")
    if not np.all(np.isfinite(array)):
        raise ValueError("time series must contain only finite values")
    return array


def _validate_horizon(horizon):
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return horizon


def _innovation_sd(y):
    if y.size < 2:
        return np.sqrt(np.finfo(float).tiny)
    return _residual_sd(np.diff(y))


def _residual_sd(values):
    values = np.asarray(values, dtype=float).reshape((-1,))
    if values.size < 2:
        return np.sqrt(np.finfo(float).tiny)
    return float(max(np.std(values, ddof=1), np.sqrt(np.finfo(float).tiny)))


def _positive(value):
    return np.maximum(np.asarray(value, dtype=float), np.finfo(float).tiny)


def _positive_scale(value):
    return np.maximum(np.asarray(value, dtype=float), np.sqrt(np.finfo(float).tiny))
