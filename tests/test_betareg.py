import jax.numpy as jnp
from jax.scipy.special import betaln
import numpy as np
import pytest

from gsm.config import BetaRegMixtureSetting, FitConfig, rajan_betareg_mixture_setting
from gsm.data import Dataset, load_csv_dataset, subset_dataset
from gsm.evaluation import fit_heldout_model_lpds
from gsm.models.betareg import (
    BetaRegMixtureParams,
    component_log_prob,
    log_prob,
    log_prob_observations,
    predict_mean_variance,
)
from gsm.variational import fit_variational, prepare_betareg_mixture_inputs


def test_rajan_betareg_setting_matches_matlab_defaults():
    setting = rajan_betareg_mixture_setting()

    assert setting.model_name == "BetaReg"
    assert not hasattr(setting, "data_file_name")
    assert setting.feature_names == ("Mean", "Disp")
    assert setting.link_types == ("logit", "log")
    assert setting.covs == (tuple(range(5)), tuple(range(5)))
    assert setting.covs_mix == tuple(range(5))
    assert setting.on_trial == (tuple(range(1, 5)), tuple(range(1, 5)))
    assert setting.on_trial_mix == tuple(range(1, 5))
    assert setting.add_constant is False
    assert setting.n_components == 2
    assert setting.standardize == 2
    assert setting.prior_mean_feat == (0.367, 2.2649)
    assert setting.prior_std_feat == (0.1, 10.0)
    assert setting.prior_shrink == ("unitinfo", "unitinfo")
    assert setting.prior_shrink_mix == 100.0


def test_betareg_component_log_prob_matches_beta_formula():
    y = jnp.asarray([0.2, 0.7])
    mean = jnp.asarray([[0.3], [0.6]])
    dispersion = jnp.asarray([[5.0], [8.0]])

    actual = component_log_prob(y, mean, dispersion)

    alpha = dispersion * mean
    beta = dispersion * (1.0 - mean)
    expected = (
        (alpha - 1.0) * jnp.log(y.reshape((-1, 1)))
        + (beta - 1.0) * jnp.log1p(-y.reshape((-1, 1)))
        - betaln(alpha, beta)
    )
    np.testing.assert_allclose(actual, expected)


def test_betareg_component_log_prob_clips_exact_boundaries_like_matlab():
    y = jnp.asarray([0.0, 1.0])
    mean = jnp.asarray([[0.3], [0.6]])
    dispersion = jnp.asarray([[5.0], [8.0]])

    actual = component_log_prob(y, mean, dispersion)

    clipped_y = jnp.asarray([0.001, 0.999]).reshape((-1, 1))
    alpha = dispersion * mean
    beta = dispersion * (1.0 - mean)
    expected = (
        (alpha - 1.0) * jnp.log(clipped_y)
        + (beta - 1.0) * jnp.log1p(-clipped_y)
        - betaln(alpha, beta)
    )
    np.testing.assert_allclose(actual, expected)


def test_betareg_mixture_predictive_moments_intercept_only():
    X = jnp.ones((4, 1))
    mean = 0.25
    dispersion = 9.0
    params = BetaRegMixtureParams(
        mean_coef=jnp.asarray([[jnp.log(mean / (1.0 - mean))]]),
        dispersion_coef=jnp.log(jnp.asarray([[dispersion]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    pred_mean, pred_variance = predict_mean_variance(params, X, X, X)

    np.testing.assert_allclose(pred_mean, np.full(4, mean))
    np.testing.assert_allclose(
        pred_variance,
        np.full(4, mean * (1.0 - mean) / (1.0 + dispersion)),
    )


def test_betareg_log_prob_sums_pointwise_values():
    X = jnp.ones((4, 1))
    y = jnp.asarray([0.2, 0.3, 0.6, 0.7])
    params = BetaRegMixtureParams(
        mean_coef=jnp.asarray([[-1.0], [1.0]]),
        dispersion_coef=jnp.log(jnp.asarray([[8.0], [8.0]])),
        gating_coef=jnp.asarray([[0.0]]),
    )

    pointwise = log_prob_observations(params, y, X, X, X)

    np.testing.assert_allclose(log_prob(params, y, X, X, X), jnp.sum(pointwise))


def test_betareg_rejects_response_outside_unit_interval():
    dataset = Dataset(
        y=np.asarray([-0.1, 0.5, 1.1])[:, None],
        X=np.ones((3, 1)),
        y_name="y",
        x_names=("Const",),
    )
    setting = BetaRegMixtureSetting(
        n_components=1,
        covs=((0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        prepare_betareg_mixture_inputs(dataset, setting)


def test_rajan_betareg_variational_smoke():
    dataset = load_csv_dataset(
        "data/Rajan.csv",
        response_column="debtratio",
        add_constant=False,
    )
    dataset = subset_dataset(dataset, np.arange(80))
    setting = rajan_betareg_mixture_setting(n_components=1)
    fit = FitConfig(
        seed=123,
        max_iter=5,
        learning_rate=0.01,
        n_restarts=1,
        tol=0.0,
        n_elbo_samples=2,
        n_predictive_samples=3,
    )

    result = fit_variational(dataset, setting, fit)

    assert result.responsibilities.shape == (80, 1)
    assert result.predictive_mean.shape == (80,)
    assert result.predictive_variance.shape == (80,)
    assert np.all(result.predictive_mean > 0.0)
    assert np.all(result.predictive_mean < 1.0)
    assert np.isfinite(result.elbo_history[-1])


def test_betareg_heldout_lpds_smoke():
    y = np.asarray([0.12, 0.18, 0.25, 0.35, 0.55, 0.65, 0.78, 0.86])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = BetaRegMixtureSetting(
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
