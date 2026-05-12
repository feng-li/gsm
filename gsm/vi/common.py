"""Shared variational result and preprocessing helpers."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from gsm.data import standardize_covariates


@dataclass(frozen=True)
class VariationalResult:
    params: Any
    elbo_history: np.ndarray
    converged: bool
    posterior: Any | None = None
    responsibilities: np.ndarray | None = None
    predictive_mean: np.ndarray | None = None
    predictive_variance: np.ndarray | None = None


def apply_standardization(
    X: np.ndarray,
    method: int,
    c1: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    if X.shape[1] != c1.shape[0] or X.shape[1] != c2.shape[0]:
        raise ValueError("standardization constants do not match design matrix")
    if method == 0:
        return X

    use_column = ~(np.isnan(c1) | np.isnan(c2))
    if not np.any(use_column):
        return X

    if method == 1:
        X[:, use_column] = (X[:, use_column] - c1[use_column]) / c2[use_column]
    elif method == 2:
        scale = 2.0 / (c2[use_column] - c1[use_column])
        offset = 1.0 - scale * c2[use_column]
        X[:, use_column] = X[:, use_column] * scale + offset
    elif method == 3:
        X[:, use_column] = X[:, use_column] - c1[use_column]
    else:
        raise ValueError(f"unknown standardization method: {method}")
    return X


def standardize_designs(
    designs: dict[str, np.ndarray],
    method: int,
    standardization: Any | None = None,
) -> dict[str, np.ndarray]:
    """Fit or apply standardization to named feature design matrices."""

    if not method:
        return designs

    standardized: dict[str, np.ndarray] = {}
    for name, design in designs.items():
        if standardization is None:
            standardized[name], _, _ = standardize_covariates(design, method)
        else:
            standardized[name] = apply_standardization(
                design,
                method,
                getattr(standardization, f"{name}_c1"),
                getattr(standardization, f"{name}_c2"),
            )
    return standardized


def fit_design_standardization(
    designs: dict[str, np.ndarray],
    method: int,
) -> dict[str, np.ndarray]:
    """Return ``*_c1`` and ``*_c2`` kwargs for a standardization dataclass."""

    stats: dict[str, np.ndarray] = {}
    for name, design in designs.items():
        _, c1, c2 = standardize_covariates(design, method)
        stats[f"{name}_c1"] = c1
        stats[f"{name}_c2"] = c2
    return stats


def validate_positive_response(y: np.ndarray) -> None:
    if np.any(np.asarray(y).reshape(-1) <= 0.0):
        raise ValueError("LogNorm and LogNormRep require strictly positive responses")


def validate_unit_interval_response(y: np.ndarray) -> None:
    y = np.asarray(y).reshape(-1)
    if np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("BetaReg requires responses between 0 and 1")


def validate_count_response(y: np.ndarray) -> None:
    y = np.asarray(y).reshape(-1)
    if not np.all(np.isfinite(y)):
        raise ValueError("count models require finite responses")
    if np.any(y < 0.0):
        raise ValueError("count models require non-negative responses")
    if not np.allclose(y, np.round(y)):
        raise ValueError("count models require integer-valued responses")
