import numpy as np

from gsm.config import FitConfig, GaussianMixtureSetting
from gsm.data import Dataset
from gsm.variational import fit_variational


def test_gaussian_mixture_variational_smoke():
    y = np.asarray([-2.1, -1.9, -2.0, 1.8, 2.0, 2.2])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = GaussianMixtureSetting(
        n_components=2,
        covs=((0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )
    fit = FitConfig(seed=123, max_iter=50, learning_rate=0.03, n_restarts=1, tol=0.0)

    result = fit_variational(dataset, setting, fit)

    assert np.isfinite(result.elbo_history).all()
    assert result.elbo_history[-1] > result.elbo_history[0]
    assert result.responsibilities is not None
    assert result.responsibilities.shape == (len(y), 2)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), np.ones(len(y)))
    assert result.predictive_mean is not None
    assert result.predictive_variance is not None


def test_gaussian_mixture_variational_with_ard():
    y = np.asarray([-2.1, -1.9, -2.0, 1.8, 2.0, 2.2])[:, None]
    x = np.linspace(-1.0, 1.0, len(y))
    X = np.column_stack([np.ones(len(y)), x])
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const", "x"))
    setting = GaussianMixtureSetting(
        n_components=2,
        covs=((0, 1), (0, 1)),
        covs_mix=(0, 1),
        standardize=0,
    )
    fit = FitConfig(
        seed=123,
        max_iter=30,
        learning_rate=0.02,
        n_restarts=1,
        tol=0.0,
        use_ard=True,
    )

    result = fit_variational(dataset, setting, fit)

    assert np.isfinite(result.elbo_history).all()
    assert result.responsibilities is not None
    assert result.responsibilities.shape == (len(y), 2)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), np.ones(len(y)))
