"""Public FEBAMA workflow helpers for precomputed log predictive densities."""

from dataclasses import dataclass

import numpy as np

from gsm.febama.data import LpdFeatures
from gsm.febama.inference import FebamaMapResult, fit_map
from gsm.febama.scoring import add_intercept, logscore, log_weights


@dataclass(frozen=True)
class FebamaFit:
    """Fitted FEBAMA gating model."""

    method: str
    beta: np.ndarray
    add_intercept: bool
    result: FebamaMapResult


@dataclass(frozen=True)
class FebamaScore:
    """Predictive log-score summary for a fitted FEBAMA model."""

    total: float
    pointwise: np.ndarray


def prepare_lpd_features(lpd, features, model_names=None, feature_names=None) -> LpdFeatures:
    """Build a validated ``LpdFeatures`` object from arrays."""

    return LpdFeatures(
        lpd=lpd,
        features=features,
        model_names=model_names,
        feature_names=feature_names,
    )


def fit_febama(
    lpd_features: LpdFeatures,
    fit_method: str = "map",
    add_intercept: bool = True,
    initial_beta=None,
    active_mask=None,
    coefficient_prior_scale: float = 10.0,
    max_iter: int = 1000,
) -> FebamaFit:
    """Fit FEBAMA gating coefficients for precomputed component log densities."""

    method = fit_method.lower()
    if method != "map":
        raise ValueError("only fit_method='map' is currently implemented")
    features = _features_for_model(lpd_features.features, add_intercept)
    result = fit_map(
        lpd_features.lpd,
        features,
        initial_beta=initial_beta,
        active_mask=active_mask,
        coefficient_prior_scale=coefficient_prior_scale,
        max_iter=max_iter,
    )
    return FebamaFit(
        method=method,
        beta=result.beta,
        add_intercept=add_intercept,
        result=result,
    )


def compute_weights(fit: FebamaFit, features) -> np.ndarray:
    """Return FEBAMA softmax weights for a fitted model."""

    matrix = _features_for_model(_raw_features(features), fit.add_intercept)
    return np.asarray(np.exp(log_weights(fit.beta, matrix)))


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
