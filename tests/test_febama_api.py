import numpy as np
import pytest

from gsm.febama import (
    compute_weights,
    fit_febama,
    prepare_lpd_features,
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

    with pytest.raises(ValueError, match="only fit_method='map'"):
        fit_febama(lpd_features, fit_method="vb")


def test_febama_prepare_lpd_features_validates_shapes():
    with pytest.raises(ValueError, match="same number of rows"):
        prepare_lpd_features(np.ones((3, 2)), np.ones((2, 1)))


def test_febama_compute_weights_validates_feature_shape():
    lpd_features = _workflow_data()
    fit = fit_febama(lpd_features, coefficient_prior_scale=100.0)

    with pytest.raises(ValueError, match="features must be a 2D matrix"):
        compute_weights(fit, np.ones(3))


def _zero_beta_score(lpd_features):
    features = add_intercept(lpd_features.features)
    beta = np.zeros((lpd_features.lpd.shape[1] - 1, features.shape[1]))
    return float(logscore(lpd_features.lpd, features, beta, sum=True))
