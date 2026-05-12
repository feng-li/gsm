import jax.numpy as jnp
from jax.scipy.special import gammaln
import numpy as np

from gsm.config import (
    FitConfig,
    GammaMixtureSetting,
    GammaRepMixtureSetting,
    gamma_mixture_setting,
    gamma_rep_mixture_setting,
)
from gsm.data import Dataset
from gsm.evaluation import fit_heldout_model_lpds
from gsm.models.gamma import (
    GammaMixtureParams,
    component_log_prob,
    log_prob,
    log_prob_observations,
    predict_mean_variance,
)
from gsm.variational import fit_variational, sample_gamma_mixture_posterior


def test_gamma_settings_match_matlab_and_rep_defaults():
    setting = gamma_mixture_setting()
    rep_setting = gamma_rep_mixture_setting()

    assert setting.model_name == "Gamma"
    assert setting.feature_names == ("Mean", "Variance")
    assert setting.link_types == ("log", "log")
    assert setting.prior_mean_feat == (361.0, 176.377**2)
    assert setting.prior_std_feat == (100.0, 100.0**2)
    assert setting.parameterization == "mean_variance"
    assert rep_setting.model_name == "GammaRep"
    assert rep_setting.feature_names == ("Shape", "Scale")
    assert rep_setting.parameterization == "shape_scale"


def test_gamma_component_log_prob_matches_matlab_formula():
    y = jnp.asarray([1.5, 3.0])
    mean = jnp.asarray([[2.0], [4.0]])
    variance = jnp.asarray([[1.0], [2.0]])

    actual = component_log_prob(y, mean, variance)

    shape = mean**2 / variance
    scale = variance / mean
    expected = (
        (shape - 1.0) * jnp.log(y.reshape((-1, 1)))
        - y.reshape((-1, 1)) / scale
        - shape * jnp.log(scale)
        - gammaln(shape)
    )
    np.testing.assert_allclose(actual, expected)


def test_gamma_rep_shape_scale_parameterization_moments():
    X = jnp.ones((3, 1))
    shape = 3.0
    scale = 2.0
    params = GammaMixtureParams(
        mean_coef=jnp.log(jnp.asarray([[shape]])),
        variance_coef=jnp.log(jnp.asarray([[scale]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    pred_mean, pred_variance = predict_mean_variance(
        params,
        X,
        X,
        X,
        parameterization="shape_scale",
    )

    np.testing.assert_allclose(pred_mean, np.full(3, shape * scale))
    np.testing.assert_allclose(pred_variance, np.full(3, shape * scale**2))


def test_gamma_nonpositive_y_returns_negative_infinity():
    mean = jnp.ones((2, 1))
    variance = jnp.ones((2, 1))

    actual = component_log_prob(jnp.asarray([0.0, -1.0]), mean, variance)

    assert np.isneginf(np.asarray(actual)).all()


def test_gamma_log_prob_sums_pointwise_values():
    X = jnp.ones((3, 1))
    y = jnp.asarray([1.0, 2.0, 3.0])
    params = GammaMixtureParams(
        mean_coef=jnp.log(jnp.asarray([[2.0]])),
        variance_coef=jnp.log(jnp.asarray([[1.0]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    pointwise = log_prob_observations(params, y, X, X, X)

    np.testing.assert_allclose(log_prob(params, y, X, X, X), jnp.sum(pointwise))


def test_gamma_rep_variational_smoke():
    y = np.asarray([1.0, 1.2, 1.4, 3.0, 3.2, 3.5])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = GammaRepMixtureSetting(
        n_components=2,
        covs=((0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )
    fit = FitConfig(
        seed=123,
        max_iter=20,
        learning_rate=0.01,
        n_restarts=1,
        tol=0.0,
        n_elbo_samples=3,
    )

    result = fit_variational(dataset, setting, fit)

    assert np.isfinite(result.elbo_history).all()
    assert result.posterior is not None
    assert result.responsibilities is not None
    assert result.responsibilities.shape == (len(y), 2)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), np.ones(len(y)))
    samples = sample_gamma_mixture_posterior(result.posterior, seed=123, n_samples=4)
    assert samples["mean_coef"].shape == (4, 2, 1)
    assert samples["variance_coef"].shape == (4, 2, 1)


def test_gamma_variational_rejects_nonpositive_response():
    y = np.asarray([0.0, 1.0, 2.0])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = GammaMixtureSetting(
        n_components=1,
        covs=((0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )

    try:
        fit_variational(dataset, setting, FitConfig(max_iter=1, n_elbo_samples=1))
    except ValueError as exc:
        assert "strictly positive responses" in str(exc)
    else:
        raise AssertionError("expected nonpositive gamma response to be rejected")


def test_gamma_heldout_lpds_smoke():
    y = np.asarray([0.8, 1.0, 1.3, 1.7, 2.4, 3.0, 3.8, 4.4])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = GammaMixtureSetting(
        n_components=1,
        covs=((0,), (0,)),
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
