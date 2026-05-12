import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

from gsm.config import FitConfig, StudentTMixtureSetting, studentt_mixture_setting
from gsm.data import Dataset
from gsm.evaluation import fit_heldout_model_lpds
from gsm.models.studentt import (
    StudentTMixtureParams,
    component_log_prob,
    log_prob,
    log_prob_observations,
    predict_mean_variance,
)
from gsm.variational import fit_variational, sample_studentt_mixture_posterior


def test_studentt_setting_matches_matlab_studt_defaults():
    setting = studentt_mixture_setting()

    assert setting.model_name == "StudT"
    assert setting.feature_names == ("Mean", "DF", "Scale")
    assert setting.link_types == ("identity", "log", "log")
    assert setting.covs == ((0,), (0,), (0,))
    assert setting.covs_mix == (0,)
    assert setting.n_components == 2
    assert setting.standardize == 1


def test_studentt_component_log_prob_matches_matlab_formula():
    y = jnp.asarray([-1.0, 0.0, 2.0])
    mean = jnp.zeros((3, 1))
    df = jnp.full((3, 1), 5.0)
    scale = jnp.full((3, 1), 2.0)

    actual = np.asarray(component_log_prob(y, mean, df, scale)).reshape(-1)

    z = np.asarray(y) / 2.0
    expected = (
        gammaln(3.0)
        - gammaln(2.5)
        - 0.5 * np.log(5.0 * np.pi)
        - np.log(2.0)
        - 3.0 * np.log1p((z**2) / 5.0)
    )
    np.testing.assert_allclose(actual, expected)


def test_studentt_predictive_moments():
    X = jnp.ones((4, 1))
    params = StudentTMixtureParams(
        mean_coef=jnp.asarray([[0.5]]),
        df_coef=jnp.log(jnp.asarray([[5.0]])),
        scale_coef=jnp.log(jnp.asarray([[2.0]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    pred_mean, pred_variance = predict_mean_variance(params, X, X, X, X)

    np.testing.assert_allclose(pred_mean, np.full(4, 0.5))
    np.testing.assert_allclose(pred_variance, np.full(4, 5.0 / 3.0 * 4.0))


def test_studentt_log_prob_sums_pointwise_values():
    X = jnp.ones((3, 1))
    y = jnp.asarray([-1.0, 0.0, 1.0])
    params = StudentTMixtureParams(
        mean_coef=jnp.asarray([[-1.0], [1.0]]),
        df_coef=jnp.log(jnp.asarray([[8.0], [8.0]])),
        scale_coef=jnp.log(jnp.asarray([[1.0], [1.0]])),
        gating_coef=jnp.asarray([[0.0]]),
    )

    pointwise = log_prob_observations(params, y, X, X, X, X)

    np.testing.assert_allclose(log_prob(params, y, X, X, X, X), jnp.sum(pointwise))


def test_studentt_variational_smoke():
    y = np.asarray([-2.1, -1.9, -2.0, 1.8, 2.0, 2.2])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = StudentTMixtureSetting(
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
    samples = sample_studentt_mixture_posterior(result.posterior, seed=123, n_samples=3)
    assert samples["df_coef"].shape == (3, 2, 1)


def test_studentt_heldout_lpds_smoke():
    y = np.asarray([-2.1, -2.0, -1.8, -1.9, 1.8, 2.0, 2.1, 2.2])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = StudentTMixtureSetting(
        n_components=1,
        covs=((0,), (0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )
    fit = FitConfig(
        seed=123,
        max_iter=5,
        learning_rate=0.01,
        n_restarts=1,
        tol=0.0,
        n_elbo_samples=2,
        n_predictive_samples=3,
    )

    result = fit_heldout_model_lpds(dataset, setting, fit, test_size=0.25)

    assert result.train_score.pointwise.shape == (6,)
    assert result.test_score.pointwise.shape == (2,)
    assert result.test_score.n_posterior_samples == 3
    assert np.isfinite(result.test_score.elpd)
