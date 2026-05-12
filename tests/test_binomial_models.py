from pathlib import Path

import jax.numpy as jnp
from jax.scipy.special import gammaln
import numpy as np

from gsm.config import (
    BetaBinMixtureSetting,
    BinomialMixtureSetting,
    FitConfig,
    betabin_mixture_setting,
    binomial_mixture_setting,
)
from gsm.data import Dataset, load_binomial_csv_dataset
from gsm.evaluation import fit_heldout_model_lpds
from gsm.models.betabinomial import (
    BetaBinMixtureParams,
    component_log_prob as betabin_component_log_prob,
    log_prob as betabin_log_prob,
    log_prob_observations as betabin_log_prob_observations,
    predict_mean_variance as betabin_predict_mean_variance,
)
from gsm.models.binomial import (
    BinomialMixtureParams,
    component_log_prob as binomial_component_log_prob,
    log_prob as binomial_log_prob,
    log_prob_observations as binomial_log_prob_observations,
    predict_mean_variance as binomial_predict_mean_variance,
)
from gsm.variational import (
    fit_variational,
    sample_betabin_mixture_posterior,
    sample_binomial_mixture_posterior,
)


def _binomial_dataset(y):
    y = np.asarray(y, dtype=float)
    X = np.ones((y.shape[0], 1))
    return Dataset(y=y, X=X, y_name="successes,trials", x_names=("Const",))


def test_binomial_settings_match_matlab_defaults():
    binomial = binomial_mixture_setting()
    betabin = betabin_mixture_setting()

    assert binomial.model_name == "Bin"
    assert binomial.feature_names == ("Mean",)
    assert binomial.link_types == ("logit",)
    assert binomial.covs == (tuple(range(7)),)
    assert binomial.covs_mix == tuple(range(7))
    assert binomial.prior_mean_feat == (0.5,)
    assert binomial.prior_std_feat == (10.0,)
    assert betabin.model_name == "BetaBin"
    assert betabin.feature_names == ("Mean", "Disp")
    assert betabin.link_types == ("logit", "log")
    assert betabin.covs == (tuple(range(8)), tuple(range(8)))
    assert betabin.covs_mix == tuple(range(7))
    assert betabin.prior_mean_feat == (0.5, 1.0)
    assert betabin.prior_std_feat == (10.0, 10.0)


def test_load_binomial_csv_dataset_uses_successes_trials_convention(tmp_path: Path):
    path = tmp_path / "binomial.csv"
    path.write_text(
        "successes,trials,x\n"
        "1,5,0.1\n"
        "3,7,0.2\n",
    )

    dataset = load_binomial_csv_dataset(path, add_constant=True)

    assert dataset.y_name == "successes,trials"
    assert dataset.x_names == ("Const", "x")
    np.testing.assert_allclose(dataset.y, np.asarray([[1.0, 5.0], [3.0, 7.0]]))
    np.testing.assert_allclose(dataset.X[:, 0], np.ones(2))


def test_binomial_component_log_prob_matches_matlab_formula():
    y = jnp.asarray([[2.0, 5.0], [7.0, 10.0]])
    probability = jnp.asarray([[0.3], [0.8]])

    actual = binomial_component_log_prob(y, probability)

    successes = y[:, 0:1]
    trials = y[:, 1:2]
    failures = trials - successes
    expected = (
        gammaln(trials + 1.0)
        - gammaln(failures + 1.0)
        - gammaln(successes + 1.0)
        + successes * jnp.log(probability)
        + failures * jnp.log1p(-probability)
    )
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(binomial_log_prob(y, probability), jnp.sum(expected))


def test_betabin_component_log_prob_matches_matlab_formula():
    y = jnp.asarray([[2.0, 5.0], [7.0, 10.0]])
    probability = jnp.asarray([[0.3], [0.8]])
    dispersion = jnp.asarray([[4.0], [2.0]])

    actual = betabin_component_log_prob(y, probability, dispersion)

    successes = y[:, 0:1]
    trials = y[:, 1:2]
    failures = trials - successes
    expected = (
        gammaln(trials + 1.0)
        - gammaln(failures + 1.0)
        - gammaln(successes + 1.0)
        + gammaln(successes + dispersion * probability)
        + gammaln(failures + dispersion * (1.0 - probability))
        + gammaln(dispersion)
        - gammaln(dispersion * probability)
        - gammaln(dispersion * (1.0 - probability))
        - gammaln(trials + dispersion)
    )
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        betabin_log_prob(y, probability, dispersion),
        jnp.sum(expected),
    )


def test_binomial_invalid_support_returns_negative_infinity():
    probability = jnp.ones((3, 1)) * 0.5
    dispersion = jnp.ones((3, 1))
    y = jnp.asarray([[-1.0, 5.0], [6.0, 5.0], [1.5, 5.0]])

    binomial_actual = binomial_component_log_prob(y, probability)
    betabin_actual = betabin_component_log_prob(y, probability, dispersion)

    assert np.isneginf(np.asarray(binomial_actual)).all()
    assert np.isneginf(np.asarray(betabin_actual)).all()


def test_binomial_predictive_moments():
    X = jnp.ones((2, 1))
    y = jnp.asarray([[2.0, 5.0], [7.0, 10.0]])
    binomial_params = BinomialMixtureParams(
        mean_coef=jnp.asarray([[0.0]]),
        gating_coef=jnp.zeros((0, 1)),
    )
    betabin_params = BetaBinMixtureParams(
        mean_coef=jnp.asarray([[0.0]]),
        dispersion_coef=jnp.log(jnp.asarray([[4.0]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    binomial_mean, binomial_variance = binomial_predict_mean_variance(
        binomial_params, X, X, trials=y
    )
    betabin_mean, betabin_variance = betabin_predict_mean_variance(
        betabin_params, X, X, X, trials=y
    )

    np.testing.assert_allclose(binomial_mean, np.asarray([2.5, 5.0]))
    np.testing.assert_allclose(binomial_variance, np.asarray([1.25, 2.5]))
    np.testing.assert_allclose(betabin_mean, np.asarray([2.5, 5.0]))
    np.testing.assert_allclose(betabin_variance, np.asarray([2.25, 7.0]))


def test_binomial_log_prob_sums_pointwise_values():
    X = jnp.ones((3, 1))
    y = jnp.asarray([[0.0, 5.0], [2.0, 5.0], [5.0, 5.0]])
    binomial_params = BinomialMixtureParams(
        mean_coef=jnp.asarray([[0.0]]),
        gating_coef=jnp.zeros((0, 1)),
    )
    betabin_params = BetaBinMixtureParams(
        mean_coef=jnp.asarray([[0.0]]),
        dispersion_coef=jnp.log(jnp.asarray([[2.0]])),
        gating_coef=jnp.zeros((0, 1)),
    )

    binomial_pointwise = binomial_log_prob_observations(binomial_params, y, X, X)
    betabin_pointwise = betabin_log_prob_observations(betabin_params, y, X, X, X)

    np.testing.assert_allclose(
        binomial_log_prob(binomial_params, y, X, X),
        jnp.sum(binomial_pointwise),
    )
    np.testing.assert_allclose(
        betabin_log_prob(betabin_params, y, X, X, X),
        jnp.sum(betabin_pointwise),
    )


def test_binomial_variational_rejects_invalid_response():
    setting = BinomialMixtureSetting(
        n_components=1,
        covs=((0,),),
        covs_mix=(0,),
        standardize=0,
    )
    invalid = _binomial_dataset([[6.0, 5.0], [1.0, 5.0]])

    try:
        fit_variational(invalid, setting, FitConfig(max_iter=1, n_elbo_samples=1))
    except ValueError as exc:
        assert "successes <= trials" in str(exc)
    else:
        raise AssertionError("expected invalid binomial response to be rejected")


def test_binomial_variational_smoke():
    dataset = _binomial_dataset(
        [[0, 10], [1, 10], [2, 10], [7, 10], [8, 10], [10, 10]]
    )
    setting = BinomialMixtureSetting(
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
    samples = sample_binomial_mixture_posterior(result.posterior, seed=123, n_samples=3)
    assert samples["mean_coef"].shape == (3, 2, 1)


def test_betabin_variational_smoke():
    dataset = _binomial_dataset(
        [[0, 10], [1, 10], [2, 10], [7, 10], [8, 10], [10, 10]]
    )
    setting = BetaBinMixtureSetting(
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
    samples = sample_betabin_mixture_posterior(result.posterior, seed=123, n_samples=3)
    assert samples["mean_coef"].shape == (3, 2, 1)
    assert samples["dispersion_coef"].shape == (3, 2, 1)


def test_binomial_heldout_lpds_smoke():
    dataset = _binomial_dataset(
        [[0, 10], [1, 10], [2, 10], [3, 10], [7, 10], [8, 10], [9, 10], [10, 10]]
    )
    binomial = BinomialMixtureSetting(
        n_components=1,
        covs=((0,),),
        covs_mix=(0,),
        standardize=0,
    )
    betabin = BetaBinMixtureSetting(
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

    binomial_result = fit_heldout_model_lpds(dataset, binomial, fit, test_size=0.25)
    betabin_result = fit_heldout_model_lpds(dataset, betabin, fit, test_size=0.25)

    assert binomial_result.test_score.pointwise.shape == (2,)
    assert betabin_result.test_score.pointwise.shape == (2,)
    assert binomial_result.test_score.n_posterior_samples == 3
    assert betabin_result.test_score.n_posterior_samples == 3
    assert np.isfinite(binomial_result.test_score.elpd)
    assert np.isfinite(betabin_result.test_score.elpd)
