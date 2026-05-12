import jax.numpy as jnp
from jax.scipy.special import gammaln
import numpy as np

from gsm.config import (
    FitConfig,
    GenPoissonAltMixtureSetting,
    GenPoissonMixtureSetting,
    genpoisson_alt_mixture_setting,
    mdvisits_genpoisson_mixture_setting,
)
from gsm.data import Dataset
from gsm.evaluation import fit_heldout_model_lpds
from gsm.links import inverse_link, link
from gsm.models.genpoisson import (
    GenPoissonMixtureParams,
    component_log_prob,
    log_prob,
    log_prob_observations,
    predict_mean_variance,
)
from gsm.variational import (
    fit_variational,
    sample_genpoisson_mixture_posterior,
)


def _count_dataset(y):
    y = np.asarray(y, dtype=float)[:, None]
    X = np.ones((len(y), 1))
    return Dataset(y=y, X=X, y_name="y", x_names=("Const",))


def test_genpoisson_settings_match_matlab_defaults():
    standard = mdvisits_genpoisson_mixture_setting()
    alternative = genpoisson_alt_mixture_setting()

    assert standard.model_name == "GenPois"
    assert standard.feature_names == ("Mean", "Disp")
    assert standard.link_types == ("log", "log")
    assert standard.covs == (tuple(range(8)), tuple(range(8)))
    assert standard.covs_mix == tuple(range(8))
    assert standard.prior_mean_feat == (2.5891, 0.5778)
    assert standard.prior_std_feat == (10.0, 1.0)
    assert standard.parameterization == "standard"
    assert alternative.model_name == "GenPoisAlt"
    assert alternative.link_types == ("log", "log1")
    assert alternative.covs == (tuple(range(7)), tuple(range(7)))
    assert alternative.parameterization == "alternative"


def test_log1_link_round_trips():
    value = jnp.asarray([1.25, 2.0, 5.0])

    actual = inverse_link(link(value, "log1"), "log1")

    np.testing.assert_allclose(actual, value)


def test_genpoisson_component_log_prob_matches_matlab_formula():
    y = jnp.asarray([0.0, 3.0])
    mean = jnp.asarray([[2.0], [4.0]])
    dispersion = jnp.asarray([[0.5], [0.25]])

    actual = component_log_prob(y, mean, dispersion)

    y_col = y.reshape((-1, 1))
    expected = (
        y_col * jnp.log(mean)
        - y_col * jnp.log1p(dispersion * mean)
        + (y_col - 1.0) * jnp.log1p(dispersion * y_col)
        - mean * (1.0 + dispersion * y_col) / (1.0 + dispersion * mean)
        - gammaln(y_col + 1.0)
    )
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(log_prob(y, mean, dispersion), jnp.sum(expected))


def test_genpoisson_alt_component_log_prob_matches_matlab_formula():
    y = jnp.asarray([0.0, 3.0])
    mean = jnp.asarray([[2.0], [4.0]])
    dispersion = jnp.asarray([[1.5], [3.0]])

    actual = component_log_prob(
        y,
        mean,
        dispersion,
        parameterization="alternative",
    )

    y_col = y.reshape((-1, 1))
    expected = (
        jnp.log(mean)
        + (y_col - 1.0) * jnp.log(mean + (dispersion - 1.0) * y_col)
        - y_col * jnp.log(dispersion)
        - (mean + (dispersion - 1.0) * y_col) / dispersion
        - gammaln(y_col + 1.0)
    )
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        log_prob(y, mean, dispersion, parameterization="alternative"),
        jnp.sum(expected),
    )


def test_genpoisson_invalid_support_returns_negative_infinity():
    mean = jnp.ones((2, 1))
    dispersion = jnp.ones((2, 1))

    actual = component_log_prob(jnp.asarray([-1.0, 1.5]), mean, dispersion)

    assert np.isneginf(np.asarray(actual)).all()


def test_genpoisson_predictive_moments():
    X = jnp.ones((2, 1))
    standard = GenPoissonMixtureParams(
        mean_coef=jnp.log(jnp.asarray([[2.0]])),
        dispersion_coef=jnp.log(jnp.asarray([[0.5]])),
        gating_coef=jnp.zeros((0, 1)),
    )
    alternative = GenPoissonMixtureParams(
        mean_coef=jnp.log(jnp.asarray([[2.0]])),
        dispersion_coef=jnp.log(jnp.asarray([[1.0]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    standard_mean, standard_variance = predict_mean_variance(standard, X, X, X)
    alt_mean, alt_variance = predict_mean_variance(
        alternative,
        X,
        X,
        X,
        parameterization="alternative",
    )

    np.testing.assert_allclose(standard_mean, np.full(2, 2.0))
    np.testing.assert_allclose(standard_variance, np.full(2, 8.0))
    np.testing.assert_allclose(alt_mean, np.full(2, 2.0))
    np.testing.assert_allclose(alt_variance, np.full(2, 8.0))


def test_genpoisson_log_prob_sums_pointwise_values():
    X = jnp.ones((3, 1))
    y = jnp.asarray([0.0, 1.0, 2.0])
    params = GenPoissonMixtureParams(
        mean_coef=jnp.log(jnp.asarray([[1.5]])),
        dispersion_coef=jnp.log(jnp.asarray([[0.25]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    pointwise = log_prob_observations(params, y, X, X, X)

    np.testing.assert_allclose(log_prob(params, y, X, X, X), jnp.sum(pointwise))


def test_genpoisson_variational_smoke():
    dataset = _count_dataset([0, 1, 1, 2, 4, 7])
    setting = GenPoissonMixtureSetting(
        n_components=2,
        covs=((0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )
    fit = FitConfig(
        seed=123,
        max_iter=15,
        learning_rate=0.01,
        n_restarts=1,
        tol=0.0,
        n_elbo_samples=2,
    )

    result = fit_variational(dataset, setting, fit)

    assert np.isfinite(result.elbo_history).all()
    assert result.posterior is not None
    assert result.responsibilities is not None
    assert result.responsibilities.shape == (6, 2)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), np.ones(6))
    samples = sample_genpoisson_mixture_posterior(result.posterior, seed=123, n_samples=3)
    assert samples["mean_coef"].shape == (3, 2, 1)
    assert samples["dispersion_coef"].shape == (3, 2, 1)


def test_genpoisson_alt_variational_smoke():
    dataset = _count_dataset([0, 1, 1, 2, 4, 7])
    setting = GenPoissonAltMixtureSetting(
        n_components=2,
        covs=((0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )
    fit = FitConfig(
        seed=123,
        max_iter=15,
        learning_rate=0.01,
        n_restarts=1,
        tol=0.0,
        n_elbo_samples=2,
    )

    result = fit_variational(dataset, setting, fit)

    assert np.isfinite(result.elbo_history).all()
    assert result.posterior is not None
    assert result.responsibilities is not None
    assert result.responsibilities.shape == (6, 2)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), np.ones(6))


def test_genpoisson_heldout_lpds_smoke():
    dataset = _count_dataset([0, 1, 1, 2, 3, 5, 8, 13])
    standard = GenPoissonMixtureSetting(
        n_components=1,
        covs=((0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )
    alternative = GenPoissonAltMixtureSetting(
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

    standard_result = fit_heldout_model_lpds(dataset, standard, fit, test_size=0.25)
    alternative_result = fit_heldout_model_lpds(dataset, alternative, fit, test_size=0.25)

    assert standard_result.test_score.pointwise.shape == (2,)
    assert alternative_result.test_score.pointwise.shape == (2,)
    assert standard_result.test_score.n_posterior_samples == 3
    assert alternative_result.test_score.n_posterior_samples == 3
    assert np.isfinite(standard_result.test_score.elpd)
    assert np.isfinite(alternative_result.test_score.elpd)
