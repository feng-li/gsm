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
