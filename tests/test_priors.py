import jax.numpy as jnp
import numpy as np

from gsm.config import GaussianMixtureSetting
from gsm.priors import (
    build_gating_prior,
    build_gaussian_coefficient_prior,
    build_gaussian_mixture_priors,
    convert_prior_to_link_scale,
    gaussian_matrix_log_prob,
)
from gsm.variational import GaussianMixtureInputs


def test_convert_prior_to_link_scale_matches_identity_and_log_matlab_cases():
    mean, std = convert_prior_to_link_scale(0.0, 10.0, "identity")
    assert mean == 0.0
    assert std == 10.0

    log_mean, log_std = convert_prior_to_link_scale(1.0, 10.0, "log")
    expected_std = np.sqrt(np.log(101.0))
    expected_mean = -0.5 * np.log(101.0)
    np.testing.assert_allclose(log_mean, expected_mean)
    np.testing.assert_allclose(log_std, expected_std)


def test_convert_prior_to_link_scale_logit_returns_finite_positive_scale():
    mean, std = convert_prior_to_link_scale(0.5, 0.1, "logit")

    assert np.isfinite(mean)
    assert np.isfinite(std)
    assert std > 0.0


def test_feature_prior_sets_link_scale_intercept_and_numeric_shrinkage():
    x = np.arange(5.0)
    design = np.column_stack([np.ones_like(x), x])

    prior = build_gaussian_coefficient_prior(
        design,
        prior_mean_feat=1.0,
        prior_std_feat=10.0,
        link_type="log",
        shrinkage=100.0,
    )

    np.testing.assert_allclose(prior.mean, [-0.5 * np.log(101.0), 0.0])
    np.testing.assert_allclose(prior.covariance[0, 0], np.log(101.0))
    np.testing.assert_allclose(prior.covariance[1, 1], 100.0)
    np.testing.assert_array_equal(prior.constant_mask, [True, False])


def test_gating_unit_info_prior_matches_zellner_g_prior():
    z = np.column_stack([np.ones(4), np.arange(4.0)])

    prior = build_gating_prior(z, n_components=3, shrinkage="UnitInfo")

    assert prior is not None
    np.testing.assert_allclose(prior.mean, np.zeros(2))
    np.testing.assert_allclose(prior.covariance, 4 * np.linalg.inv(z.T @ z))
    np.testing.assert_allclose(prior.precision, np.linalg.inv(prior.covariance))


def test_gaussian_feature_unit_info_uses_heterogauss_expected_hessian():
    x = np.arange(1.0, 5.0)
    design = np.column_stack([np.ones_like(x), x])
    inputs = GaussianMixtureInputs(
        y=np.zeros_like(x),
        X_mean=design,
        X_variance=design,
        Z=design,
    )
    setting = GaussianMixtureSetting(
        n_components=2,
        covs=((0, 1), (0, 1)),
        covs_mix=(0, 1),
        standardize=0,
        prior_shrink=("UnitInfo", "UnitInfo"),
    )

    priors = build_gaussian_mixture_priors(inputs, setting)

    variance_at_prior_mean = np.exp(-0.5 * np.log(101.0))
    sum_x2 = np.sum(x**2)
    np.testing.assert_allclose(
        priors.mean.covariance[1, 1],
        len(x) * variance_at_prior_mean / sum_x2,
    )
    np.testing.assert_allclose(
        priors.log_variance.covariance[1, 1],
        2.0 * len(x) / sum_x2,
    )


def test_gaussian_matrix_log_prob_includes_matlab_normalizing_constants():
    design = np.ones((3, 1))
    prior = build_gaussian_coefficient_prior(
        design,
        prior_mean_feat=0.0,
        prior_std_feat=2.0,
        link_type="identity",
        shrinkage=100.0,
    )
    value = jnp.asarray([[0.0], [1.0]])

    log_prob = gaussian_matrix_log_prob(value, prior)
    expected = (
        -0.5 * np.log(2.0 * np.pi * 4.0)
        - 0.5 * np.log(2.0 * np.pi * 4.0)
        - 0.5 * (1.0**2 / 4.0)
    )
    np.testing.assert_allclose(log_prob, expected)
