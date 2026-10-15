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


def test_example_reuses_cached_forecasts_and_features():
    calls = {"expert": 0, "features": 0}

    def expert(y, horizon):
        calls["expert"] += 1
        return FORECASTERS[0](y, horizon)

    def features(y):
        calls["features"] += 1
        return dict(_simple_features(y), constant=1.)

    cache = {}
    y = np.sin(np.arange(12.))
    data = compute_lpd_features(
        y, (expert, expert), ("last", "mean", "constant"), model_names=MODEL_NAMES,
        start=5, max_origins=4, feature_window=3, feature_function=features, forecast_cache=cache,
    )
    train, test = _train_test_split(data, 1)
    train = clean_features(train)
    fit = fit_febama(train)
    before = calls.copy()
    forecasts = _forecast_holdout(
        y, None, test.origin, test.response, fit, train, (expert, expert),
        feature_names=train.feature_names, feature_window=3, feature_function=features,
        forecast_cache=cache, raw_feature_names=data.feature_names,
    )
    assert calls == before
    np.testing.assert_allclose(forecasts[0].lpd, test.lpd)


def test_multistep_holdout_keeps_training_targets_before_test_origin():
    data = compute_lpd_features(
        np.sin(np.arange(14.)), FORECASTERS, ("last", "mean"),
        start=3, max_origins=7, horizon=3, feature_function=_simple_features,
    )
    train, test = _train_test_split(data, 2)
    assert np.max(train.origin + train.lead - 1) < np.min(test.origin)
    assert len(np.unique(test.origin)) == 2
