import jax.numpy as jnp
from jax.scipy.special import gammaln
import numpy as np

from gsm.config import (
    FitConfig,
    NegBinMixtureSetting,
    PoissonMixtureSetting,
    mdvisits_negbin_mixture_setting,
    mdvisits_poisson_mixture_setting,
)
from gsm.data import Dataset
from gsm.evaluation import fit_heldout_model_lpds
from gsm.models.negbin import (
    NegBinMixtureParams,
    component_log_prob as negbin_component_log_prob,
    log_prob as negbin_log_prob,
    log_prob_observations as negbin_log_prob_observations,
    predict_mean_variance as negbin_predict_mean_variance,
)
from gsm.models.poisson import (
    PoissonMixtureParams,
    component_log_prob as poisson_component_log_prob,
    log_prob as poisson_log_prob,
    log_prob_observations as poisson_log_prob_observations,
    predict_mean_variance as poisson_predict_mean_variance,
)
from gsm.variational import (
    fit_variational,
    sample_negbin_mixture_posterior,
    sample_poisson_mixture_posterior,
)


def _count_dataset(y):
    y = np.asarray(y, dtype=float)[:, None]
    X = np.ones((len(y), 1))
    return Dataset(y=y, X=X, y_name="y", x_names=("Const",))


def test_count_settings_match_mdvisits_matlab_defaults():
    poisson = mdvisits_poisson_mixture_setting()
    negbin = mdvisits_negbin_mixture_setting()

    assert poisson.model_name == "Pois"
    assert poisson.feature_names == ("Mean",)
    assert poisson.link_types == ("log",)
    assert poisson.covs == (tuple(range(8)),)
    assert poisson.covs_mix == tuple(range(8))
    assert poisson.prior_mean_feat == (2.5891,)
    assert poisson.prior_std_feat == (10.0,)
    assert negbin.model_name == "NegBin"
    assert negbin.feature_names == ("Mean", "Disp")
    assert negbin.link_types == ("log", "log")
    assert negbin.covs == (tuple(range(8)), tuple(range(8)))
    assert negbin.prior_mean_feat == (2.5891, 0.4951)
    assert negbin.prior_std_feat == (10.0, 10.0)


def test_poisson_component_log_prob_matches_matlab_formula():
    y = jnp.asarray([0.0, 3.0])
    mean = jnp.asarray([[1.5], [2.0]])

    actual = poisson_component_log_prob(y, mean)

    expected = y.reshape((-1, 1)) * jnp.log(mean) - mean - gammaln(y.reshape((-1, 1)) + 1.0)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(poisson_log_prob(y, mean), jnp.sum(expected))


def test_negbin_component_log_prob_matches_matlab_formula():
    y = jnp.asarray([0.0, 3.0])
    mean = jnp.asarray([[1.5], [2.0]])
    dispersion = jnp.asarray([[0.7], [1.2]])

    actual = negbin_component_log_prob(y, mean, dispersion)

    y_col = y.reshape((-1, 1))
    fraction = mean / (mean + dispersion)
    expected = (
        -gammaln(y_col + 1.0)
        + gammaln(y_col + dispersion)
        - gammaln(dispersion)
        + y_col * jnp.log(fraction)
        + dispersion * jnp.log1p(-fraction)
    )
    np.testing.assert_allclose(actual, expected)


def test_count_model_predictive_moments():
    X = jnp.ones((3, 1))
    poisson_params = PoissonMixtureParams(
        mean_coef=jnp.log(jnp.asarray([[2.0]])),
        gating_coef=jnp.zeros((0, 1)),
    )
    negbin_params = NegBinMixtureParams(
        mean_coef=jnp.log(jnp.asarray([[2.0]])),
        dispersion_coef=jnp.log(jnp.asarray([[4.0]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    poisson_mean, poisson_variance = poisson_predict_mean_variance(poisson_params, X, X)
    negbin_mean, negbin_variance = negbin_predict_mean_variance(negbin_params, X, X, X)

    np.testing.assert_allclose(poisson_mean, np.full(3, 2.0))
    np.testing.assert_allclose(poisson_variance, np.full(3, 2.0))
    np.testing.assert_allclose(negbin_mean, np.full(3, 2.0))
    np.testing.assert_allclose(negbin_variance, np.full(3, 3.0))


def test_count_component_invalid_support_returns_negative_infinity():
    mean = jnp.ones((2, 1))
    dispersion = jnp.ones((2, 1))
    y = jnp.asarray([-1.0, 1.5])

    poisson_actual = poisson_component_log_prob(y, mean)
    negbin_actual = negbin_component_log_prob(y, mean, dispersion)

    assert np.isneginf(np.asarray(poisson_actual)).all()
    assert np.isneginf(np.asarray(negbin_actual)).all()


def test_count_log_prob_sums_pointwise_values():
    X = jnp.ones((3, 1))
    y = jnp.asarray([0.0, 1.0, 2.0])
    poisson_params = PoissonMixtureParams(
        mean_coef=jnp.log(jnp.asarray([[1.5]])),
        gating_coef=jnp.zeros((0, 1)),
    )
    negbin_params = NegBinMixtureParams(
        mean_coef=jnp.log(jnp.asarray([[1.5]])),
        dispersion_coef=jnp.log(jnp.asarray([[2.0]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    poisson_pointwise = poisson_log_prob_observations(poisson_params, y, X, X)
    negbin_pointwise = negbin_log_prob_observations(negbin_params, y, X, X, X)

    np.testing.assert_allclose(
        poisson_log_prob(poisson_params, y, X, X),
        jnp.sum(poisson_pointwise),
    )
    np.testing.assert_allclose(
        negbin_log_prob(negbin_params, y, X, X, X),
        jnp.sum(negbin_pointwise),
    )


def test_count_variational_rejects_invalid_responses():
    fractional = _count_dataset([0.0, 1.5, 2.0])
    negative = _count_dataset([0.0, -1.0, 2.0])
    setting = PoissonMixtureSetting(
        n_components=1,
        covs=((0,),),
        covs_mix=(0,),
        standardize=0,
    )

    try:
        fit_variational(fractional, setting, FitConfig(max_iter=1, n_elbo_samples=1))
    except ValueError as exc:
        assert "integer-valued" in str(exc)
    else:
        raise AssertionError("expected fractional count response to be rejected")

    try:
        fit_variational(negative, setting, FitConfig(max_iter=1, n_elbo_samples=1))
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("expected negative count response to be rejected")


def test_poisson_variational_smoke():
    dataset = _count_dataset([0, 1, 1, 2, 3, 5])
    setting = PoissonMixtureSetting(
        n_components=2,
        covs=((0,),),
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
    samples = sample_poisson_mixture_posterior(result.posterior, seed=123, n_samples=3)
    assert samples["mean_coef"].shape == (3, 2, 1)


def test_negbin_variational_smoke():
    dataset = _count_dataset([0, 1, 1, 2, 4, 6])
    setting = NegBinMixtureSetting(
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
    samples = sample_negbin_mixture_posterior(result.posterior, seed=123, n_samples=3)
    assert samples["mean_coef"].shape == (3, 2, 1)
    assert samples["dispersion_coef"].shape == (3, 2, 1)


def test_count_heldout_lpds_smoke():
    dataset = _count_dataset([0, 1, 1, 2, 3, 5, 8, 13])
    poisson = PoissonMixtureSetting(
        n_components=1,
        covs=((0,),),
        covs_mix=(0,),
        standardize=0,
    )
    negbin = NegBinMixtureSetting(
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

    poisson_result = fit_heldout_model_lpds(dataset, poisson, fit, test_size=0.25)
    negbin_result = fit_heldout_model_lpds(dataset, negbin, fit, test_size=0.25)

    assert poisson_result.test_score.pointwise.shape == (2,)
    assert negbin_result.test_score.pointwise.shape == (2,)
    assert poisson_result.test_score.n_posterior_samples == 3
    assert negbin_result.test_score.n_posterior_samples == 3
    assert np.isfinite(poisson_result.test_score.elpd)
    assert np.isfinite(negbin_result.test_score.elpd)
