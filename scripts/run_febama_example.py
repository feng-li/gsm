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
    clean_features,
    compute_tsfeatures,
    compute_weights,
    fit_febama,
    log_prob_matrix,
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
    lpd, features, origin_dates = _rolling_lpd_features(
        y=y,
        dates=dates,
        start=args.start,
        max_origins=args.max_origins,
        feature_window=args.feature_window,
        feature_names=tuple(args.features),
    )
    train, test = _train_test_split(lpd, features, origin_dates, args.test_size)

    train_data = prepare_lpd_features(
        train["lpd"],
        train["features"],
        model_names=MODEL_NAMES,
        feature_names=tuple(args.features),
    )
    train_clean = clean_features(train_data)
    fit = fit_febama(
        train_clean,
        coefficient_prior_scale=args.prior_scale,
        max_iter=args.max_iter,
    )

    test_features = standardize_features(
        test["features"],
        train_clean.feature_mean,
        train_clean.feature_sd,
        feature_names=tuple(args.features),
        reference_feature_names=train_clean.feature_names,
    )
    test_data = prepare_lpd_features(
        test["lpd"],
        test_features,
        model_names=MODEL_NAMES,
        feature_names=train_clean.feature_names,
    )
    test_score = score_febama(test_data, fit)
    equal_weight_score = _equal_weight_score(test["lpd"])
    weights = compute_weights(fit, test_data.features)

    print(f"data: {args.data}")
    print(f"rows: {y.shape[0]}")
    print(f"rolling origins: {lpd.shape[0]}")
    print(f"train origins: {train['lpd'].shape[0]}")
    print(f"test origins: {test['lpd'].shape[0]}")
    print(f"test dates: {test['dates'][0]} to {test['dates'][-1]}")
    print(f"base models: {', '.join(MODEL_NAMES)}")
    print(f"requested features: {', '.join(args.features)}")
    print(f"kept features: {', '.join(train_clean.feature_names) or '(intercept only)'}")
    print(f"febama test log score: {test_score.total:.6f}")
    print(f"equal-weight test log score: {equal_weight_score:.6f}")
    print(f"improvement: {test_score.total - equal_weight_score:.6f}")
    print("average held-out weights:")
    for name, weight in zip(MODEL_NAMES, np.asarray(weights).mean(axis=0), strict=True):
        print(f"  {name}: {weight:.4f}")


def _read_series(path: Path, value_column: str, date_column: str) -> tuple[np.ndarray, tuple[str, ...]]:
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


def _rolling_lpd_features(
    y: np.ndarray,
    dates: tuple[str, ...],
    start: int,
    max_origins: int,
    feature_window: int,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if start < 2:
        raise ValueError("start must leave at least two historical observations")
    if max_origins < 2:
        raise ValueError("max_origins must be at least two")
    if feature_window < 2:
        raise ValueError("feature_window must be at least two")

    stop = min(y.shape[0], start + max_origins)
    if stop - start < 2:
        raise ValueError("not enough observations for the requested rolling origins")

    lpd_rows: list[np.ndarray] = []
    feature_rows: list[list[float]] = []
    origin_dates: list[str] = []
    for origin in range(start, stop):
        history = y[:origin]
        feature_history = history[-feature_window:]
        feature_values = compute_tsfeatures(
            feature_history,
            frequency=1,
            feature_names=feature_names,
        )
        predictions = (naive_fore(history, 1), rw_drift_fore(history, 1))
        lpd_row = np.asarray(log_prob_matrix(np.asarray([y[origin]]), predictions))[0]

        feature_rows.append([feature_values[name] for name in feature_names])
        lpd_rows.append(lpd_row)
        origin_dates.append(dates[origin])

    return (
        np.asarray(lpd_rows, dtype=float),
        np.asarray(feature_rows, dtype=float),
        tuple(origin_dates),
    )


def _train_test_split(
    lpd: np.ndarray,
    features: np.ndarray,
    dates: tuple[str, ...],
    test_size: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if test_size < 1:
        raise ValueError("test_size must be positive")
    if test_size >= lpd.shape[0]:
        raise ValueError("test_size must be smaller than the number of rolling origins")

    split = lpd.shape[0] - test_size
    train = {"lpd": lpd[:split], "features": features[:split], "dates": dates[:split]}
    test = {"lpd": lpd[split:], "features": features[split:], "dates": dates[split:]}
    return train, test


def _equal_weight_score(lpd: np.ndarray) -> float:
    return float(np.sum(logsumexp(lpd, axis=1) - np.log(lpd.shape[1])))


if __name__ == "__main__":
    main()
