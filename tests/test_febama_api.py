import numpy as np
import pytest

from gsm.febama import (
    SeriesData,
    compute_lpd_features,
    compute_weights,
    fit_febama,
    naive_fore,
    prepare_lpd_features,
    rw_drift_fore,
    sample_weights,
    score_febama,
)
from gsm.febama.scoring import add_intercept, logscore


def _workflow_data():
    x = np.linspace(-2.0, 2.0, 11)
    features = x[:, None]
    lpd = np.column_stack([2.0 * x, np.zeros_like(x)])
    return prepare_lpd_features(lpd, features, model_names=("signal", "baseline"))


def test_febama_public_workflow_fits_scores_and_computes_weights():
    lpd_features = _workflow_data()

    fit = fit_febama(lpd_features, coefficient_prior_scale=100.0)
    weights = compute_weights(fit, lpd_features)
    score = score_febama(lpd_features, fit)

    assert fit.method == "map"
    assert fit.add_intercept is True
    assert fit.beta.shape == (1, 2)
    assert weights.shape == lpd_features.lpd.shape
    assert score.pointwise.shape == (lpd_features.lpd.shape[0],)
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(lpd_features.lpd.shape[0]))
    assert np.isfinite(score.total)
    assert score.total > _zero_beta_score(lpd_features)


def test_febama_public_fit_can_skip_intercept():
    lpd_features = _workflow_data()
    features_with_intercept = prepare_lpd_features(
        lpd_features.lpd,
        add_intercept(lpd_features.features),
        model_names=lpd_features.model_names,
    )

    fit = fit_febama(
        features_with_intercept,
        add_intercept=False,
        coefficient_prior_scale=100.0,
    )
    weights = compute_weights(fit, features_with_intercept.features)

    assert fit.beta.shape == (1, 2)
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(lpd_features.lpd.shape[0]))


def test_febama_public_api_rejects_unknown_fit_method():
    lpd_features = _workflow_data()

    with pytest.raises(ValueError, match="fit_method"):
        fit_febama(lpd_features, fit_method="sgld")


def test_febama_prepare_lpd_features_validates_shapes():
    with pytest.raises(ValueError, match="same number of rows"):
        prepare_lpd_features(np.ones((3, 2)), np.ones((2, 1)))


def test_febama_compute_weights_validates_feature_shape():
    lpd_features = _workflow_data()
    fit = fit_febama(lpd_features, coefficient_prior_scale=100.0)

    with pytest.raises(ValueError, match="features must be a 2D matrix"):
        compute_weights(fit, np.ones(3))


def test_febama_public_vb_workflow_samples_weights():
    lpd_features = _workflow_data()

    fit = fit_febama(
        lpd_features,
        fit_method="vb",
        coefficient_prior_scale=100.0,
        max_iter=30,
        learning_rate=0.05,
        n_elbo_samples=4,
        seed=123,
    )
    weights = sample_weights(fit, lpd_features, n_samples=3, seed=123)

    assert fit.method == "vb"
    assert fit.posterior is not None
    assert fit.result.elbo_history.shape[0] >= 1
    assert weights.shape == (3, lpd_features.lpd.shape[0], lpd_features.lpd.shape[1])
    np.testing.assert_allclose(weights.sum(axis=2), np.ones(weights.shape[:2]))
    assert np.isfinite(weights).all()


def test_febama_sample_weights_requires_vb_fit():
    lpd_features = _workflow_data()
    fit = fit_febama(lpd_features, coefficient_prior_scale=100.0)

    with pytest.raises(ValueError, match="VB fit"):
        sample_weights(fit, lpd_features, n_samples=2)


def test_vb_scores_integrate_densities_and_weights():
    from gsm.febama.api import FebamaFit
    from gsm.febama.inference import FebamaVbPosterior
    from scipy.integrate import quad
    from scipy.special import expit
    from scipy.stats import norm

    posterior = FebamaVbPosterior(np.array([[2.]]), np.log(np.array([[3.]])),
                                 np.ones((1, 1), dtype=bool))
    fit = FebamaFit("vb", posterior.mean, True, None, posterior)
    data = prepare_lpd_features(np.array([[0., -5.]]), np.empty((1, 0)))
    score = score_febama(data, fit, n_samples=8192, seed=7)
    weights = compute_weights(fit, data.features, n_samples=8192, seed=7)
    expected_weight = quad(lambda z: expit(2 + 3*z) * norm.pdf(z), -12, 12)[0]
    expected_score = np.log(expected_weight + (1-expected_weight)*np.exp(-5))
    np.testing.assert_allclose(score.total, expected_score, atol=.02)
    np.testing.assert_allclose(score.total, np.log(weights[0, 0]+weights[0, 1]*np.exp(-5)))
    assert score.n_posterior_samples == 8192
    plugin = score_febama(data, fit, n_samples=0)
    assert plugin.n_posterior_samples == 0
    assert abs(plugin.total-score.total) > .1


def test_febama_compute_lpd_features_builds_rolling_training_data():
    y = np.linspace(1.0, 2.0, 12)
    dates = tuple(f"date-{idx}" for idx in range(y.shape[0]))

    lpd_features = compute_lpd_features(
        SeriesData(x=y, date=dates),
        forecasters=(naive_fore, rw_drift_fore),
        feature_names=("last", "mean"),
        model_names=("naive", "drift"),
        start=5,
        max_origins=4,
        feature_window=3,
        feature_function=_simple_features,
    )

    assert lpd_features.lpd.shape == (4, 2)
    assert lpd_features.features.shape == (4, 2)
    assert lpd_features.model_names == ("naive", "drift")
    assert lpd_features.feature_names == ("last", "mean")
    np.testing.assert_allclose(lpd_features.response, y[5:9])
    np.testing.assert_array_equal(lpd_features.origin, np.asarray([5, 6, 7, 8]))
    assert lpd_features.date == dates[5:9]
    np.testing.assert_allclose(
        lpd_features.features[0],
        np.asarray([y[4], np.mean(y[2:5])]),
    )
    assert np.isfinite(lpd_features.lpd).all()


def test_febama_compute_lpd_features_accepts_precomputed_features():
    y = np.linspace(1.0, 2.0, 8)
    features = np.arange(6.0).reshape((3, 2))

    lpd_features = compute_lpd_features(
        y,
        forecasters=(naive_fore, rw_drift_fore),
        feature_names=("a", "b"),
        start=3,
        max_origins=3,
        precomputed_features=features,
    )

    np.testing.assert_allclose(lpd_features.features, features)
    assert lpd_features.feature_names == ("a", "b")
    np.testing.assert_array_equal(lpd_features.origin, np.asarray([3, 4, 5]))


def test_febama_compute_lpd_features_validates_inputs():
    with pytest.raises(ValueError, match="at least two forecasters"):
        compute_lpd_features([1.0, 2.0, 3.0], forecasters=(naive_fore,))
    with pytest.raises(ValueError, match="feature_names are required"):
        compute_lpd_features(
            [1.0, 2.0, 3.0, 4.0],
            forecasters=(naive_fore, rw_drift_fore),
            start=2,
            feature_function=_simple_features,
        )


def _zero_beta_score(lpd_features):
    features = add_intercept(lpd_features.features)
    beta = np.zeros((lpd_features.lpd.shape[1] - 1, features.shape[1]))
    return float(logscore(lpd_features.lpd, features, beta, sum=True))


def _simple_features(history):
    history = np.asarray(history, dtype=float)
    return {"last": history[-1], "mean": float(np.mean(history))}


def test_multistep_training_scores_the_same_marginals_as_forecasting():
    from gsm.febama import PredictiveDistribution, clean_features, forecast_febama

    def negative(history, horizon):
        return PredictiveDistribution("gaussian", {"mean": np.full(horizon, -1.), "sd": .25})

    def positive(history, horizon):
        return PredictiveDistribution("gaussian", {"mean": np.full(horizon, 1.), "sd": .25})

    history, actual = np.array([0., 1., 2.]), np.array([-1., 1.])
    data = compute_lpd_features(
        np.r_[history, actual], (negative, positive), ("x",), start=3, horizon=2,
        precomputed_features=np.zeros((1, 1)),
    )
    np.testing.assert_array_equal(data.origin, [3, 3])
    np.testing.assert_array_equal(data.lead, [1, 2])
    np.testing.assert_array_equal(data.response, actual)
    assert data.lpd.shape == (2, 2)
    clean = clean_features(data)
    fit = fit_febama(clean, active_mask=np.zeros((1, 1), dtype=bool))
    forecast = forecast_febama(
        SeriesData(history, actual), fit, clean, (negative, positive), horizon=2,
        feature_names=("x",), feature_function=lambda y: {"x": 0.},
    )
    np.testing.assert_allclose(score_febama(clean, fit).total, forecast.log_score)


def test_named_columns_are_aligned_to_the_fit():
    from dataclasses import replace

    data = prepare_lpd_features(
        [[0., -2.], [-1., 0.], [0., -1.]], [[1., 0.], [0., 1.], [2., 3.]],
        model_names=("a", "b"), feature_names=("x", "z"),
    )
    fit = fit_febama(data)
    swapped = replace(
        data, features=data.features[:, ::-1], feature_names=("z", "x"),
        lpd=data.lpd[:, ::-1], model_names=("b", "a"),
    )
    np.testing.assert_allclose(score_febama(data, fit).pointwise, score_febama(swapped, fit).pointwise)
    np.testing.assert_allclose(compute_weights(fit, data), compute_weights(fit, swapped))
    with pytest.raises(ValueError, match="model names"):
        score_febama(replace(data, model_names=("a", "unknown")), fit)
    with pytest.raises(ValueError, match="feature names"):
        compute_weights(fit, replace(data, feature_names=("x", "unknown")))


def test_fitted_scaling_is_preserved_and_checked():
    from dataclasses import replace
    from gsm.febama import clean_features

    data = clean_features(_workflow_data())
    fit = fit_febama(data)
    np.testing.assert_array_equal(fit.feature_mean, data.feature_mean)
    np.testing.assert_array_equal(fit.feature_sd, data.feature_sd)
    with pytest.raises(ValueError, match="scaling"):
        compute_weights(fit, replace(data, feature_mean=data.feature_mean + 1.))


def test_posterior_weight_kernel_is_compiled_once(monkeypatch):
    import jax
    import gsm.febama.api as api
    from gsm.febama.inference import FebamaVbPosterior

    traces = []

    @jax.jit
    def kernel(beta, features):
        traces.append(True)
        return jax.vmap(api.log_weights, in_axes=(0, None))(beta, features)

    monkeypatch.setattr(api, "_sample_log_weights", kernel)
    posterior = FebamaVbPosterior(np.zeros((1, 2)), np.full((1, 2), -2.), np.ones((1, 2), bool))
    fit = api.FebamaFit("vb", posterior.mean, True, None, posterior)
    for _ in range(3):
        sample_weights(fit, np.ones((3, 1)), n_samples=8)
    assert len(traces) == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_lpd_container_rejects_contaminating_values(bad):
    with pytest.raises(ValueError, match="NaN or positive infinity"):
        prepare_lpd_features([[0., bad]], [[1.]])
