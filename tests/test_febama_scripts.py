import numpy as np

from gsm.febama import (
    SeriesData,
    clean_features,
    compute_lpd_features,
    fit_febama,
    summarize_performance,
)
from scripts.run_febama_example import (
    FORECASTERS,
    MODEL_NAMES,
    PERFORMANCE_COLUMNS,
    _equal_weight_score,
    _forecast_holdout,
    _performance_row,
    _train_test_split,
)


def test_run_febama_example_fast_one_origin_summary_columns_are_finite():
    y = np.linspace(0.0, 1.0, 12)
    dates = tuple(f"date-{idx}" for idx in range(y.shape[0]))
    lpd_features = compute_lpd_features(
        SeriesData(x=y, date=dates),
        forecasters=FORECASTERS,
        feature_names=("last", "mean"),
        model_names=MODEL_NAMES,
        start=5,
        max_origins=4,
        feature_window=3,
        feature_function=_simple_features,
    )
    train_raw, test_raw = _train_test_split(lpd_features, test_size=1)
    train_clean = clean_features(train_raw)
    fit = fit_febama(train_clean, coefficient_prior_scale=100.0, max_iter=100)

    forecasts = _forecast_holdout(
        y,
        dates,
        test_raw.origin,
        test_raw.response,
        fit,
        train_clean,
        FORECASTERS,
        feature_names=train_clean.feature_names,
        feature_window=3,
        feature_function=_simple_features,
    )
    row = _performance_row(
        summarize_performance(forecasts),
        _equal_weight_score(test_raw.lpd),
    )

    assert tuple(row) == PERFORMANCE_COLUMNS
    assert row["n_forecasts"] == 1
    assert row["n_scored_forecasts"] == 1
    for name in PERFORMANCE_COLUMNS[2:]:
        assert np.isfinite(row[name])


def _simple_features(history):
    history = np.asarray(history, dtype=float)
    return {"last": history[-1], "mean": float(np.mean(history))}
