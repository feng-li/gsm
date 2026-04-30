import numpy as np

from gsm.config import FitConfig, GaussianMixtureSetting
from gsm.data import Dataset
from gsm.evaluation import (
    chronological_holdout_indices,
    fit_heldout_gaussian_mixture_lpds,
    predictive_log_score,
)
from gsm.variational import fit_variational


def test_chronological_holdout_indices():
    train_idx, test_idx = chronological_holdout_indices(10, 0.3)

    np.testing.assert_array_equal(train_idx, np.arange(7))
    np.testing.assert_array_equal(test_idx, np.arange(7, 10))


def test_heldout_gaussian_mixture_lpds_smoke():
    y = np.asarray([-2.1, -2.0, -1.8, -1.9, 1.8, 2.0, 2.1, 2.2])[:, None]
    x = np.linspace(-1.0, 1.0, len(y))
    X = np.column_stack([np.ones(len(y)), x])
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const", "x"))
    setting = GaussianMixtureSetting(
        n_components=1,
        covs=((0, 1), (0,)),
        covs_mix=(0, 1),
        standardize=2,
    )
    fit = FitConfig(
        seed=123,
        max_iter=20,
        learning_rate=0.02,
        n_restarts=1,
        tol=0.0,
        n_elbo_samples=3,
        n_predictive_samples=5,
    )

    result = fit_heldout_gaussian_mixture_lpds(dataset, setting, fit, test_size=0.25)

    np.testing.assert_array_equal(result.train_indices, np.arange(6))
    np.testing.assert_array_equal(result.test_indices, np.arange(6, 8))
    assert result.train_score.pointwise.shape == (6,)
    assert result.test_score.pointwise.shape == (2,)
    assert np.isfinite(result.train_score.elpd)
    assert np.isfinite(result.test_score.elpd)
    assert result.train_score.n_posterior_samples == 5
    assert result.test_score.n_posterior_samples == 5


def test_predictive_log_score_can_fall_back_to_posterior_mean():
    y = np.asarray([-1.0, 0.0, 1.0])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = GaussianMixtureSetting(
        n_components=1,
        covs=((0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )
    fit = FitConfig(seed=123, max_iter=5, learning_rate=0.01, n_elbo_samples=2)
    result = fit_variational(dataset, setting, fit)

    score = predictive_log_score(result, dataset, setting, n_samples=0)

    assert score.pointwise.shape == (3,)
    assert score.n_posterior_samples == 0
    assert np.isfinite(score.elpd)
