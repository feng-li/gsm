import jax.numpy as jnp
import numpy as np

from gsm.config import FitConfig, SplitNormalMixtureSetting, sp500_splitnormal_mixture_setting
from gsm.data import Dataset
from gsm.models.splitnormal import (
    SplitNormalMixtureParams,
    component_log_prob,
    log_prob,
    predict_mean_variance,
)
from gsm.variational import fit_variational, sample_splitnormal_mixture_posterior


def test_sp500_splitnormal_setting_matches_matlab_asymnorm_layout():
    setting = sp500_splitnormal_mixture_setting()

    assert setting.model_name == "SplitNormal"
    assert setting.feature_names == ("Mean", "Sigma", "Skewness")
    assert setting.link_types == ("identity", "log", "log")
    assert setting.covs == ((0,), tuple(range(10)), tuple(range(10)))
    assert setting.covs_mix == tuple(range(10))
    assert setting.n_components == 1
    assert setting.standardize == 2


def test_symmetric_splitnormal_log_prob_matches_normal_density():
    y = jnp.asarray([-1.0, 0.0, 2.0])
    mean = jnp.zeros((3, 1))
    scale = jnp.full((3, 1), 2.0)
    skewness = jnp.ones((3, 1))

    actual = np.asarray(component_log_prob(y, mean, scale, skewness)).reshape(-1)
    expected = -0.5 * np.log(2.0 * np.pi) - np.log(2.0) - (np.asarray(y) ** 2) / 8.0

    np.testing.assert_allclose(actual, expected)


def test_splitnormal_predictive_moments_are_normal_when_symmetric():
    X = jnp.ones((4, 1))
    params = SplitNormalMixtureParams(
        mean_coef=jnp.array([[0.0]]),
        scale_coef=jnp.log(jnp.array([[2.0]])),
        skewness_coef=jnp.array([[0.0]]),
        gating_coef=jnp.zeros((0, 1)),
    )

    pred_mean, pred_variance = predict_mean_variance(params, X, X, X, X)

    np.testing.assert_allclose(pred_mean, np.zeros(4))
    np.testing.assert_allclose(pred_variance, np.full(4, 4.0))


def test_splitnormal_log_prob_is_finite():
    X = jnp.ones((3, 1))
    y = jnp.asarray([-1.0, 0.0, 1.0])
    params = SplitNormalMixtureParams(
        mean_coef=jnp.array([[-1.0], [1.0]]),
        scale_coef=jnp.log(jnp.array([[1.0], [1.0]])),
        skewness_coef=jnp.array([[0.0], [0.0]]),
        gating_coef=jnp.array([[0.0]]),
    )

    assert np.isfinite(float(log_prob(params, y, X, X, X, X)))


def test_splitnormal_variational_smoke():
    y = np.asarray([-2.1, -1.9, -2.0, 1.8, 2.0, 2.2])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = SplitNormalMixtureSetting(
        n_components=2,
        covs=((0,), (0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )
    fit = FitConfig(
        seed=123,
        max_iter=10,
        learning_rate=0.01,
        n_restarts=1,
        tol=0.0,
        n_elbo_samples=2,
    )

    result = fit_variational(dataset, setting, fit)

    assert np.isfinite(result.elbo_history).all()
    assert result.posterior is not None
    assert result.responsibilities is not None
    assert result.responsibilities.shape == (len(y), 2)
    samples = sample_splitnormal_mixture_posterior(result.posterior, seed=123, n_samples=3)
    assert samples["scale_coef"].shape == (3, 2, 1)
