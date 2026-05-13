"""Run a small FEBAMA example with live tsfeatures and simple forecasters."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp


PYTHON_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

from gsm.febama import (  # noqa: E402
    SeriesData,
    clean_features,
    compute_lpd_features,
    compute_weights,
    fit_febama,
    naive_fore,
    prepare_lpd_features,
    rw_drift_fore,
    score_febama,
    standardize_features,
)


DEFAULT_DATA_PATH = PYTHON_CODE_ROOT / "data" / "sp500_1990-2009_calendar.csv"
DEFAULT_FEATURES = ("x_acf1", "diff1_acf1", "entropy", "alpha", "beta", "unitroot_kpss")
MODEL_NAMES = ("naive", "rw_drift")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal FEBAMA example using rolling S&P 500 returns."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--value-column", default="Returns")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--start", type=int, default=150)
    parser.add_argument("--max-origins", type=int, default=10)
    parser.add_argument("--test-size", type=int, default=2)
    parser.add_argument("--feature-window", type=int, default=100)
    parser.add_argument("--prior-scale", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    args = parser.parse_args()

    y, dates = _read_series(args.data, args.value_column, args.date_column)
    lpd_features = compute_lpd_features(
        SeriesData(x=y, date=dates),
        forecasters=(naive_fore, rw_drift_fore),
        feature_names=tuple(args.features),
        model_names=MODEL_NAMES,
        start=args.start,
        max_origins=args.max_origins,
        feature_window=args.feature_window,
    )
    train_raw, test_raw = _train_test_split(lpd_features, args.test_size)

    train_clean = clean_features(train_raw)
    fit = fit_febama(
        train_clean,
        coefficient_prior_scale=args.prior_scale,
        max_iter=args.max_iter,
    )

    test_features = standardize_features(
        test_raw.features,
        train_clean.feature_mean,
        train_clean.feature_sd,
        feature_names=tuple(args.features),
        reference_feature_names=train_clean.feature_names,
    )
    test_data = prepare_lpd_features(
        test_raw.lpd,
        test_features,
        model_names=MODEL_NAMES,
        feature_names=train_clean.feature_names,
        response=test_raw.response,
        origin=test_raw.origin,
        date=test_raw.date,
    )
    test_score = score_febama(test_data, fit)
    equal_weight_score = _equal_weight_score(test_raw.lpd)
    weights = compute_weights(fit, test_data.features)

    print(f"data: {args.data}")
    print(f"rows: {y.shape[0]}")
    print(f"rolling origins: {lpd_features.lpd.shape[0]}")
    print(f"train origins: {train_raw.lpd.shape[0]}")
    print(f"test origins: {test_raw.lpd.shape[0]}")
    if test_raw.date is not None:
        print(f"test dates: {test_raw.date[0]} to {test_raw.date[-1]}")
    print(f"base models: {', '.join(MODEL_NAMES)}")
    print(f"requested features: {', '.join(args.features)}")
    print(f"kept features: {', '.join(train_clean.feature_names) or '(intercept only)'}")
    print(f"febama test log score: {test_score.total:.6f}")
    print(f"equal-weight test log score: {equal_weight_score:.6f}")
    print(f"improvement: {test_score.total - equal_weight_score:.6f}")
    print("average held-out weights:")
    for name, weight in zip(MODEL_NAMES, np.asarray(weights).mean(axis=0), strict=True):
        print(f"  {name}: {weight:.4f}")


def _read_series(
    path: Path,
    value_column: str,
    date_column: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values: list[float] = []
    dates: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("data file must have a header row")
        if value_column not in reader.fieldnames:
            raise ValueError(f"missing value column {value_column!r}")
        has_date = date_column in reader.fieldnames
        for row_number, row in enumerate(reader):
            value = float(row[value_column])
            if not np.isfinite(value):
                continue
            values.append(value)
            dates.append(row[date_column] if has_date else str(row_number))
    if len(values) < 3:
        raise ValueError("data file must contain at least three finite observations")
    return np.asarray(values, dtype=float), tuple(dates)


def _train_test_split(lpd_features, test_size: int):
    if test_size < 1:
        raise ValueError("test_size must be positive")
    if test_size >= lpd_features.lpd.shape[0]:
        raise ValueError("test_size must be smaller than the number of rolling origins")

    split = lpd_features.lpd.shape[0] - test_size
    return _slice_lpd_features(lpd_features, slice(None, split)), _slice_lpd_features(
        lpd_features,
        slice(split, None),
    )


def _slice_lpd_features(lpd_features, rows):
    indices = np.arange(lpd_features.lpd.shape[0])[rows]
    return prepare_lpd_features(
        lpd_features.lpd[rows],
        lpd_features.features[rows],
        model_names=lpd_features.model_names,
        feature_names=lpd_features.feature_names,
        response=None if lpd_features.response is None else lpd_features.response[rows],
        origin=None if lpd_features.origin is None else lpd_features.origin[rows],
        date=None
        if lpd_features.date is None
        else tuple(lpd_features.date[int(i)] for i in indices),
    )


def _equal_weight_score(lpd: np.ndarray) -> float:
    return float(np.sum(logsumexp(lpd, axis=1) - np.log(lpd.shape[1])))


if __name__ == "__main__":
    main()
