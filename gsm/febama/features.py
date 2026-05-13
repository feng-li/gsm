"""Feature extraction, cleaning, and table-loading helpers for FEBAMA."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from gsm.febama.data import LpdFeatures


SP500_TABLE3_FEATURES: tuple[str, ...] = (
    "alpha",
    "arch_acf",
    "arch_r2",
    "beta",
    "crossing_points",
    "diff1x_pacf5",
    "diff2_acf1",
    "diff2_acf10",
    "entropy",
    "garch_acf",
    "garch_r2",
    "nonlinearity",
    "trend",
    "unitroot_kpss",
    "x_acf1",
)


@dataclass(frozen=True)
class PrecomputedFeatureTable:
    """Precomputed rolling features with optional response and LPD columns."""

    features: np.ndarray
    feature_names: tuple[str, ...]
    response: np.ndarray | None = None
    date: tuple[str, ...] | None = None
    origin: tuple[str, ...] | None = None
    lpd_features: LpdFeatures | None = None


def clean_features(lpd_features: LpdFeatures, drop_constant: bool = True) -> LpdFeatures:
    """Drop unusable feature columns and store R-style scaling metadata."""

    features = np.asarray(lpd_features.features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    names = _feature_names(lpd_features.feature_names, features.shape[1])

    finite = np.all(np.isfinite(features), axis=0)
    mean = np.zeros(features.shape[1], dtype=float)
    sd = np.ones(features.shape[1], dtype=float)
    if features.shape[0] > 0 and np.any(finite):
        ddof = 1 if features.shape[0] > 1 else 0
        mean[finite] = np.mean(features[:, finite], axis=0)
        sd[finite] = np.std(features[:, finite], axis=0, ddof=ddof)
    scale_ok = np.isfinite(mean) & np.isfinite(sd) & (sd > 0.0)
    keep = finite & (scale_ok if drop_constant else np.isfinite(mean))

    kept = features[:, keep]
    kept_mean = mean[keep]
    kept_sd = sd[keep] if drop_constant else np.where(sd[keep] > 0.0, sd[keep], 1.0)
    scaled = standardize_features(kept, kept_mean, kept_sd)
    return LpdFeatures(
        lpd=lpd_features.lpd,
        features=scaled,
        feature_mean=kept_mean,
        feature_sd=kept_sd,
        model_names=lpd_features.model_names,
        feature_names=tuple(np.asarray(names, dtype=object)[keep].tolist()),
    )


def standardize_features(
    features,
    feature_mean,
    feature_sd,
    feature_names: Iterable[str] | None = None,
    reference_feature_names: Iterable[str] | None = None,
) -> np.ndarray:
    """Apply stored feature scaling, optionally aligning columns by name."""

    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D matrix")

    if reference_feature_names is not None:
        if feature_names is None:
            raise ValueError("feature_names are required when reference_feature_names are used")
        matrix = _align_columns(matrix, tuple(feature_names), tuple(reference_feature_names))

    mean = np.asarray(feature_mean, dtype=float)
    sd = np.asarray(feature_sd, dtype=float)
    if mean.shape != (matrix.shape[1],):
        raise ValueError("feature_mean length must match feature columns")
    if sd.shape != (matrix.shape[1],):
        raise ValueError("feature_sd length must match feature columns")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sd)):
        raise ValueError("feature_mean and feature_sd must be finite")
    if np.any(sd <= 0.0):
        raise ValueError("feature_sd must be positive")
    return (matrix - mean) / sd


def compute_tsfeatures(
    y,
    frequency: int | None = 1,
    feature_names: Iterable[str] | None = None,
    *,
    scale: bool = True,
    threads: int | None = 1,
    feature_functions: Iterable[Callable] | None = None,
) -> dict[str, float]:
    """Compute one row of ``tsfeatures`` values for a single series."""

    values = np.asarray(y, dtype=float)
    if values.ndim != 1:
        raise ValueError("y must be a 1D array")
    if values.size == 0:
        raise ValueError("y must contain at least one observation")
    if not np.all(np.isfinite(values)):
        raise ValueError("y must contain only finite values")

    try:
        import pandas as pd
        import tsfeatures
    except ImportError as exc:
        raise ImportError(
            "tsfeatures is required for compute_tsfeatures; install gsm[febama]."
        ) from exc

    ts = pd.DataFrame(
        {
            "unique_id": "series_0",
            "ds": np.arange(values.size, dtype=int),
            "y": values,
        }
    )
    kwargs = {"freq": frequency, "scale": scale, "threads": threads}
    if feature_functions is not None:
        kwargs["features"] = list(feature_functions)
    row = tsfeatures.tsfeatures(ts, **kwargs).iloc[0].to_dict()
    row.pop("unique_id", None)
    out = {name: float(value) for name, value in row.items() if _is_number(value)}

    if feature_names is None:
        return out
    requested = tuple(feature_names)
    missing = [name for name in requested if name not in out]
    if missing:
        raise ValueError(f"missing tsfeatures columns: {missing}")
    return {name: out[name] for name in requested}


def read_precomputed_feature_table(
    path,
    feature_columns: Iterable[str],
    *,
    lpd_columns: Iterable[str] | None = None,
    response_column: str | None = None,
    date_column: str | None = None,
    origin_column: str | None = None,
    model_names: Iterable[str] | None = None,
) -> PrecomputedFeatureTable:
    """Load a CSV feature table without hard-coded data paths or schemas."""

    feature_names = tuple(feature_columns)
    lpd_names = None if lpd_columns is None else tuple(lpd_columns)
    rows = _read_csv_rows(path)
    _require_columns(rows, feature_names, "feature")
    if lpd_names is not None:
        _require_columns(rows, lpd_names, "lpd")
    if response_column is not None:
        _require_columns(rows, (response_column,), "response")
    if date_column is not None:
        _require_columns(rows, (date_column,), "date")
    if origin_column is not None:
        _require_columns(rows, (origin_column,), "origin")

    features = _numeric_matrix(rows, feature_names)
    response = None if response_column is None else _numeric_vector(rows, response_column)
    date = None if date_column is None else tuple(row[date_column] for row in rows)
    origin = None if origin_column is None else tuple(row[origin_column] for row in rows)

    lpd_features = None
    if lpd_names is not None:
        lpd_features = LpdFeatures(
            lpd=_numeric_matrix(rows, lpd_names),
            features=features,
            model_names=None if model_names is None else tuple(model_names),
            feature_names=feature_names,
        )
    return PrecomputedFeatureTable(
        features=features,
        feature_names=feature_names,
        response=response,
        date=date,
        origin=origin,
        lpd_features=lpd_features,
    )


def _read_csv_rows(path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("feature table must contain at least one row")
    return rows


def _require_columns(rows: list[dict[str, str]], columns: Iterable[str], label: str) -> None:
    available = set(rows[0])
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _numeric_matrix(rows: list[dict[str, str]], columns: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[_as_float(row[column]) for column in columns] for row in rows], dtype=float)


def _numeric_vector(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([_as_float(row[column]) for row in rows], dtype=float)


def _as_float(value: str) -> float:
    if value == "":
        return np.nan
    return float(value)


def _feature_names(names: tuple[str, ...] | None, n_features: int) -> tuple[str, ...]:
    if names is None:
        return tuple(f"feature_{idx}" for idx in range(n_features))
    if len(names) != n_features:
        raise ValueError("feature_names length must match feature columns")
    return tuple(names)


def _align_columns(
    matrix: np.ndarray,
    feature_names: tuple[str, ...],
    reference_feature_names: tuple[str, ...],
) -> np.ndarray:
    if len(feature_names) != matrix.shape[1]:
        raise ValueError("feature_names length must match feature columns")
    index = {name: pos for pos, name in enumerate(feature_names)}
    missing = [name for name in reference_feature_names if name not in index]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return matrix[:, [index[name] for name in reference_feature_names]]


def _is_number(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
