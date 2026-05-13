import numpy as np
import pytest

from gsm.febama import (
    FebamaForecast,
    PredictiveDistribution,
    SeriesData,
    clean_features,
    fit_febama,
    forecast_febama,
    mase,
    prepare_lpd_features,
    smape,
    summarize_performance,
)


def test_forecast_febama_returns_one_step_scores_and_weights():
    fit, lpd_features = _fitted_gate()
    data = SeriesData(
        x=np.linspace(0.0, 1.0, 10),
        xx=np.asarray([0.2]),
        date=tuple(f"date-{idx}" for idx in range(11)),
    )

    forecast = forecast_febama(
        data,
        fit,
        lpd_features,
        forecasters=(_zero_forecaster, _one_forecaster),
        feature_names=("level",),
        feature_function=_level_feature,
    )

    assert forecast.forecast.shape == (1,)
    assert forecast.weights.shape == (1, 2)
    assert forecast.lpd.shape == (1, 2)
    assert forecast.features.shape == (1, 1)
    assert forecast.date == ("date-10",)
    assert len(forecast.predictions) == 2
    np.testing.assert_allclose(forecast.weights.sum(axis=1), np.ones(1))
    assert np.isfinite(forecast.log_score)
    assert np.isfinite(forecast.mase)
    assert np.isfinite(forecast.smape)


def test_forecast_febama_without_holdout_skips_scores():
    fit, lpd_features = _fitted_gate()

    forecast = forecast_febama(
        np.linspace(0.0, 1.0, 10),
        fit,
        lpd_features,
        forecasters=(_zero_forecaster, _one_forecaster),
        feature_names=("level",),
        feature_function=_level_feature,
    )

    assert forecast.lpd is None
    assert forecast.log_score is None
    assert forecast.mase is None
    assert forecast.smape is None
    assert forecast.weight_samples is None
    assert forecast.forecast_samples is None
    assert forecast.log_score_samples is None
    assert forecast.forecast.shape == (1,)


def test_forecast_febama_recurses_across_multi_step_horizon():
    fit, lpd_features = _fitted_gate()
    data = SeriesData(
        x=np.linspace(0.0, 1.0, 10),
        xx=np.asarray([0.2, 0.3, 0.4]),
        date=tuple(f"date-{idx}" for idx in range(13)),
    )

    forecast = forecast_febama(
        data,
        fit,
        lpd_features,
        forecasters=(_zero_forecaster, _one_forecaster),
        feature_names=("level",),
        feature_function=_last_feature,
        horizon=3,
    )

    assert forecast.forecast.shape == (3,)
    assert forecast.weights.shape == (3, 2)
    assert forecast.lpd.shape == (3, 2)
    assert forecast.features.shape == (3, 1)
    assert forecast.date == ("date-10", "date-11", "date-12")
    assert len(forecast.predictions) == 2
    assert forecast.predictions[0].mean().shape == (3,)
    np.testing.assert_allclose(forecast.weights.sum(axis=1), np.ones(3))
    expected_second_feature = (
        forecast.forecast[0] - lpd_features.feature_mean[0]
    ) / lpd_features.feature_sd[0]
    np.testing.assert_allclose(forecast.features[1, 0], expected_second_feature)
    assert np.isfinite(forecast.log_score)
    assert np.isfinite(forecast.mase)
    assert np.isfinite(forecast.smape)


def test_forecast_febama_vb_returns_posterior_sampled_forecasts():
    fit, lpd_features = _fitted_vb_gate()
    data = SeriesData(x=np.linspace(0.0, 1.0, 10), xx=np.asarray([0.2, 0.3, 0.4]))

    forecast = forecast_febama(
        data,
        fit,
        lpd_features,
        forecasters=(_zero_forecaster, _one_forecaster),
        feature_names=("level",),
        feature_function=_last_feature,
        horizon=3,
        n_weight_samples=4,
        seed=123,
    )

    assert forecast.weight_samples.shape == (4, 3, 2)
    assert forecast.forecast_samples.shape == (4, 3)
    assert forecast.log_score_samples.shape == (4,)
    np.testing.assert_allclose(forecast.weight_samples.sum(axis=2), np.ones((4, 3)))
    assert np.isfinite(forecast.forecast_samples).all()
    assert np.isfinite(forecast.log_score_samples).all()


def test_forecast_febama_map_keeps_sample_fields_empty():
    fit, lpd_features = _fitted_gate()

    forecast = forecast_febama(
        np.linspace(0.0, 1.0, 10),
        fit,
        lpd_features,
        forecasters=(_zero_forecaster, _one_forecaster),
        feature_names=("level",),
        feature_function=_level_feature,
        n_weight_samples=3,
    )

    assert forecast.weight_samples is None
    assert forecast.forecast_samples is None
    assert forecast.log_score_samples is None


def test_forecast_febama_validates_current_scope_and_cleaning():
    fit, lpd_features = _fitted_gate()
    raw = prepare_lpd_features(
        lpd_features.lpd,
        lpd_features.features,
        model_names=lpd_features.model_names,
        feature_names=lpd_features.feature_names,
    )

    with pytest.raises(ValueError, match="horizon"):
        forecast_febama(
            np.linspace(0.0, 1.0, 10),
            fit,
            lpd_features,
            forecasters=(_zero_forecaster, _one_forecaster),
            feature_names=("level",),
            feature_function=_level_feature,
            horizon=0,
        )
    with pytest.raises(ValueError, match="cleaned"):
        forecast_febama(
            np.linspace(0.0, 1.0, 10),
            fit,
            raw,
            forecasters=(_zero_forecaster, _one_forecaster),
            feature_names=("level",),
            feature_function=_level_feature,
        )
    with pytest.raises(ValueError, match="n_weight_samples"):
        forecast_febama(
            np.linspace(0.0, 1.0, 10),
            fit,
            lpd_features,
            forecasters=(_zero_forecaster, _one_forecaster),
            feature_names=("level",),
            feature_function=_level_feature,
            n_weight_samples=-1,
        )


def test_febama_forecast_metrics_validate_shapes_and_zero_denominator():
    np.testing.assert_allclose(smape([0.0], [0.0]), 0.0)
    assert np.isnan(mase([1.0], [1.0], [2.0, 2.0]))
    with pytest.raises(ValueError, match="same shape"):
        smape([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="same shape"):
        mase([1.0, 2.0], [1.0], [1.0, 2.0])


def test_summarize_performance_aggregates_scored_forecasts():
    forecasts = (
        _summary_forecast(log_score=1.0, mase=2.0, smape=10.0),
        _summary_forecast(log_score=3.0, mase=np.nan, smape=20.0),
        _summary_forecast(log_score=None, mase=None, smape=None),
    )

    summary = summarize_performance(forecasts)

    assert summary.n_forecasts == 3
    assert summary.n_scored_forecasts == 2
    np.testing.assert_allclose(summary.total_log_score, 4.0)
    np.testing.assert_allclose(summary.mean_log_score, 2.0)
    np.testing.assert_allclose(summary.mean_mase, 2.0)
    np.testing.assert_allclose(summary.mean_smape, 15.0)
    assert summary.total_log_score_samples is None
    assert summary.mean_log_score_samples is None


def test_summarize_performance_aggregates_sampled_log_scores():
    forecasts = (
        _summary_forecast(log_score=1.0, log_score_samples=[0.8, 1.2]),
        _summary_forecast(log_score=2.0, log_score_samples=[1.5, 2.5]),
    )

    summary = summarize_performance(forecasts)

    np.testing.assert_allclose(summary.total_log_score_samples, [2.3, 3.7])
    np.testing.assert_allclose(summary.mean_log_score_samples, [1.15, 1.85])


def test_summarize_performance_validates_inputs():
    with pytest.raises(ValueError, match="at least one forecast"):
        summarize_performance([])
    with pytest.raises(ValueError, match="matching shape"):
        summarize_performance(
            (
                _summary_forecast(log_score=1.0, log_score_samples=[1.0, 2.0]),
                _summary_forecast(log_score=2.0, log_score_samples=[1.0]),
            )
        )


def _fitted_gate():
    cleaned = _gate_features()
    fit = fit_febama(cleaned, coefficient_prior_scale=100.0)
    return fit, cleaned


def _fitted_vb_gate():
    cleaned = _gate_features()
    fit = fit_febama(
        cleaned,
        fit_method="vb",
        coefficient_prior_scale=100.0,
        max_iter=20,
        learning_rate=0.05,
        n_elbo_samples=3,
        seed=123,
    )
    return fit, cleaned


def _gate_features():
    x = np.linspace(-1.0, 1.0, 9)
    lpd = np.column_stack([2.0 * x, np.zeros_like(x)])
    raw = prepare_lpd_features(
        lpd,
        x[:, None],
        model_names=("zero", "one"),
        feature_names=("level",),
    )
    return clean_features(raw)


def _level_feature(history):
    return {"level": 1.0}


def _last_feature(history):
    return {"level": float(np.asarray(history)[-1])}


def _zero_forecaster(y, horizon):
    return PredictiveDistribution(
        "gaussian",
        {"mean": np.zeros(horizon), "sd": np.full(horizon, 0.5)},
    )


def _one_forecaster(y, horizon):
    return PredictiveDistribution(
        "gaussian",
        {"mean": np.ones(horizon), "sd": np.full(horizon, 0.5)},
    )


def _summary_forecast(
    log_score,
    mase=1.0,
    smape=1.0,
    log_score_samples=None,
):
    return FebamaForecast(
        forecast=np.zeros(1),
        weights=np.ones((1, 1)),
        log_score=log_score,
        mase=mase,
        smape=smape,
        lpd=None,
        features=np.zeros((1, 1)),
        predictions=(),
        log_score_samples=None
        if log_score_samples is None
        else np.asarray(log_score_samples, dtype=float),
    )
