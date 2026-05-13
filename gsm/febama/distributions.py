"""Predictive distribution adapters for FEBAMA component forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
import numpy as np

from gsm.models import gamma, lognormal, poisson, splitnormal, splitt, studentt


ArrayLike = object


@dataclass(frozen=True)
class DistributionSpec:
    """Functions needed by FEBAMA for one predictive distribution."""

    log_prob: Callable[[ArrayLike, Mapping[str, ArrayLike]], jnp.ndarray]
    mean: Callable[[Mapping[str, ArrayLike]], jnp.ndarray]
    variance: Callable[[Mapping[str, ArrayLike]], jnp.ndarray]
    support: str = "real"


@dataclass(frozen=True)
class PredictiveDistribution:
    """A component predictive distribution returned by a FEBAMA forecaster."""

    name: str
    params: Mapping[str, ArrayLike]

    def log_prob(self, y) -> jnp.ndarray:
        return log_prob_from_prediction(y, self)

    def mean(self) -> jnp.ndarray:
        return get_distribution(self.name).mean(self.params).reshape((-1,))

    def variance(self) -> jnp.ndarray:
        return get_distribution(self.name).variance(self.params).reshape((-1,))


_REGISTRY: dict[str, DistributionSpec] = {}


def register_distribution(name: str, spec: DistributionSpec) -> None:
    """Register a predictive distribution adapter."""

    _REGISTRY[_normalize_name(name)] = spec


def get_distribution(name: str) -> DistributionSpec:
    """Return a registered distribution adapter."""

    normalized = _normalize_name(name)
    try:
        return _REGISTRY[normalized]
    except KeyError as exc:
        choices = ", ".join(available_distributions())
        raise ValueError(f"unknown FEBAMA distribution {name!r}; available: {choices}") from exc


def available_distributions() -> tuple[str, ...]:
    """Return registered distribution names."""

    return tuple(sorted(_REGISTRY))


def log_prob_from_prediction(y, prediction: PredictiveDistribution) -> jnp.ndarray:
    """Return pointwise log probabilities for one component prediction."""

    return get_distribution(prediction.name).log_prob(y, prediction.params).reshape((-1,))


def log_prob_matrix(
    y,
    predictions: tuple[PredictiveDistribution, ...] | list[PredictiveDistribution],
):
    """Stack component log probabilities into an ``n_obs`` by ``n_components`` matrix."""

    if len(predictions) < 1:
        raise ValueError("at least one prediction is required")
    columns = [log_prob_from_prediction(y, prediction) for prediction in predictions]
    return jnp.column_stack(columns)


def _gaussian_log_prob(y, params):
    mean = _param_column(params, "mean")
    variance = _variance(params)
    y = _column(y)
    return -0.5 * (jnp.log(2.0 * jnp.pi * variance) + ((y - mean) ** 2) / variance)


def _studentt_log_prob(y, params):
    return studentt.component_log_prob(
        y,
        _param_column(params, "mean"),
        _positive(_param_column(params, "df")),
        _scale(params),
    )


def _splitnormal_log_prob(y, params):
    return splitnormal.component_log_prob(
        y,
        _param_column(params, "mean"),
        _scale(params),
        _positive(_param_column(params, "skewness")),
    )


def _splitt_log_prob(y, params):
    return splitt.component_log_prob(
        y,
        _param_column(params, "mean"),
        _positive(_param_column(params, "df")),
        _scale(params),
        _positive(_param_column(params, "skewness")),
    )


def _lognormal_log_prob(y, params):
    return lognormal.component_log_prob(
        y,
        _param_column(params, "mean"),
        _scale(params),
        _string_param(params, "parameterization", "standard"),
    )


def _gamma_log_prob(y, params):
    first, second, parameterization = _gamma_params(params)
    return gamma.component_log_prob(y, first, second, parameterization)


def _poisson_log_prob(y, params):
    return poisson.component_log_prob(y, _mean_or_rate(params))


def _identity_mean(params):
    return _param_column(params, "mean")


def _gaussian_variance(params):
    return _variance(params)


def _studentt_variance(params):
    df = _positive(_param_column(params, "df"))
    scale = _scale(params)
    return jnp.where(df > 2.0, df / (df - 2.0) * scale**2, jnp.nan)


def _splitnormal_mean(params):
    mean = _param_column(params, "mean")
    scale = _scale(params)
    skewness = _positive(_param_column(params, "skewness"))
    return mean + jnp.sqrt(2.0 / jnp.pi) * scale * (skewness - 1.0)


def _splitnormal_variance(params):
    scale = _scale(params)
    skewness = _positive(_param_column(params, "skewness"))
    return scale**2 * (skewness + ((jnp.pi - 2.0) / jnp.pi) * (skewness - 1.0) ** 2)


def _splitt_mean(params):
    mean = _param_column(params, "mean")
    df = _positive(_param_column(params, "df"))
    scale = _scale(params)
    skewness = _positive(_param_column(params, "skewness"))
    beta_term = jnp.exp(splitt.betaln(0.5 * df, 0.5))
    post_c = 2.0 * jnp.sqrt(df) * scale * (skewness - 1.0) / ((df - 1.0) * beta_term)
    return jnp.where(df > 1.0, mean + post_c, jnp.nan)


def _splitt_variance(params):
    df = _positive(_param_column(params, "df"))
    scale = _scale(params)
    skewness = _positive(_param_column(params, "skewness"))
    beta_term = jnp.exp(splitt.betaln(0.5 * df, 0.5))
    post_c = 2.0 * jnp.sqrt(df) * scale * (skewness - 1.0) / ((df - 1.0) * beta_term)
    variance = (
        ((1.0 + skewness**3) / (1.0 + skewness))
        * (df / (df - 2.0))
        * scale**2
        - post_c**2
    )
    return jnp.where(df > 2.0, variance, jnp.nan)


def _lognormal_mean(params):
    mean = _param_column(params, "mean")
    scale = _scale(params)
    if _string_param(params, "parameterization", "standard") == "response":
        return mean
    return jnp.exp(mean + 0.5 * scale**2)


def _lognormal_variance(params):
    mean = _param_column(params, "mean")
    scale = _scale(params)
    if _string_param(params, "parameterization", "standard") == "response":
        return scale**2
    return jnp.exp(2.0 * mean + scale**2) * jnp.expm1(scale**2)


def _gamma_mean(params):
    first, second, parameterization = _gamma_params(params)
    if parameterization == "shape_scale":
        return first * second
    return first


def _gamma_variance(params):
    first, second, parameterization = _gamma_params(params)
    if parameterization == "shape_scale":
        return first * second**2
    return second


def _mean_or_rate(params):
    if "mean" in params:
        return _positive(_param_column(params, "mean"))
    return _positive(_param_column(params, "rate"))


def _gamma_params(params):
    if "shape" in params and "scale" in params:
        return (
            _positive(_param_column(params, "shape")),
            _positive(_param_column(params, "scale")),
            "shape_scale",
        )
    return (
        _positive(_param_column(params, "mean")),
        _positive(_param_column(params, "variance")),
        _string_param(params, "parameterization", "mean_variance"),
    )


def _variance(params):
    if "variance" in params:
        return _positive(_param_column(params, "variance"))
    return _scale(params) ** 2


def _scale(params):
    for name in ("scale", "sd", "sigma"):
        if name in params:
            return _positive(_param_column(params, name))
    raise KeyError("expected one of: scale, sd, sigma")


def _param_column(params, name):
    try:
        return _column(params[name])
    except KeyError as exc:
        raise KeyError(f"missing distribution parameter: {name}") from exc


def _string_param(params, name, default):
    value = params.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _column(value):
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.float64)
    return array.reshape((-1, 1))


def _positive(value):
    value = jnp.asarray(value)
    return jnp.maximum(value, jnp.finfo(value.dtype).tiny)


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


register_distribution(
    "gaussian",
    DistributionSpec(_gaussian_log_prob, _identity_mean, _gaussian_variance),
)
register_distribution(
    "studentt",
    DistributionSpec(_studentt_log_prob, _identity_mean, _studentt_variance),
)
register_distribution(
    "splitnormal",
    DistributionSpec(_splitnormal_log_prob, _splitnormal_mean, _splitnormal_variance),
)
register_distribution(
    "splitt",
    DistributionSpec(_splitt_log_prob, _splitt_mean, _splitt_variance),
)
register_distribution(
    "lognormal",
    DistributionSpec(_lognormal_log_prob, _lognormal_mean, _lognormal_variance, "positive"),
)
register_distribution(
    "gamma",
    DistributionSpec(_gamma_log_prob, _gamma_mean, _gamma_variance, "positive"),
)
register_distribution(
    "poisson",
    DistributionSpec(_poisson_log_prob, _mean_or_rate, _mean_or_rate, "count"),
)
