import jax.numpy as jnp
import numpy as np

from gsm.config import (
    FitConfig,
    LogNormalMixtureSetting,
    LogNormalRepMixtureSetting,
    lognormal_mixture_setting,
    lognormal_rep_mixture_setting,
)
from gsm.data import Dataset
from gsm.models.lognormal import (
    LogNormalMixtureParams,
    component_log_prob,
    log_prob,
    predict_mean_variance,
)
from gsm.variational import fit_variational, sample_lognormal_mixture_posterior


def test_lognormal_settings_match_matlab_defaults():
    setting = lognormal_mixture_setting()
    rep_setting = lognormal_rep_mixture_setting()

    assert setting.model_name == "LogNorm"
    assert setting.feature_names == ("Mean", "Scale")
    assert setting.link_types == ("log", "log")
    assert setting.prior_mean_feat == (np.log(361.0), np.log(176.377))
    assert setting.parameterization == "standard"
    assert rep_setting.model_name == "LogNormRep"
    assert rep_setting.prior_mean_feat == (361.0, 176.377)
    assert rep_setting.parameterization == "response"


def test_lognormal_component_log_prob_matches_formula():
    y = jnp.array([1.5, 3.0])
    mean = jnp.array([[0.2], [0.2]])
    scale = jnp.array([[0.7], [0.7]])

    actual = component_log_prob(y, mean, scale)
    expected = (
        -0.5 * ((jnp.log(y).reshape((-1, 1)) - mean) / scale) ** 2
        - jnp.log(scale)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - jnp.log(y).reshape((-1, 1))
    )

    np.testing.assert_allclose(actual, expected)


def test_lognormal_response_parameterization_moments():
    X = jnp.ones((3, 1))
    params = LogNormalMixtureParams(
        mean_coef=jnp.log(jnp.array([[2.0]])),
        scale_coef=jnp.log(jnp.array([[0.5]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    pred_mean, pred_variance = predict_mean_variance(
        params,
        X,
        X,
        X,
        parameterization="response",
    )

    np.testing.assert_allclose(pred_mean, np.full(3, 2.0))
    np.testing.assert_allclose(pred_variance, np.full(3, 0.25))
    assert np.isfinite(
        component_log_prob(
            jnp.array([2.0]),
            2.0,
            0.5,
            parameterization="response",
        )
    )


def test_lognormal_nonpositive_y_returns_negative_infinity():
    mean = jnp.ones((2, 1))
    scale = jnp.ones((2, 1))

    actual = component_log_prob(jnp.array([0.0, -1.0]), mean, scale)

    assert np.isneginf(np.asarray(actual)).all()


def test_lognormal_rep_variational_smoke():
    y = np.asarray([1.0, 1.2, 1.4, 3.0, 3.2, 3.5])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = LogNormalRepMixtureSetting(
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
    samples = sample_lognormal_mixture_posterior(result.posterior, seed=123, n_samples=4)
    assert samples["mean_coef"].shape == (4, 2, 1)
    assert samples["scale_coef"].shape == (4, 2, 1)


def test_lognormal_variational_rejects_nonpositive_response():
    y = np.asarray([0.0, 1.0, 2.0])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = LogNormalMixtureSetting(
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
        raise AssertionError("expected nonpositive lognormal response to be rejected")


def test_lognormal_log_prob_is_finite_for_positive_y():
    X = jnp.ones((3, 1))
    y = jnp.array([1.0, 2.0, 3.0])
    params = LogNormalMixtureParams(
        mean_coef=jnp.array([[0.0]]),
        scale_coef=jnp.array([[0.0]]),
        gating_coef=jnp.zeros((0, 1)),
    )

    assert np.isfinite(float(log_prob(params, y, X, X, X)))
