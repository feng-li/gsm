import jax.numpy as jnp
import numpy as np

from gsm.config import hetero_gaussian_setting
from gsm.models.gaussian_mixture import (
    GaussianMixtureParams,
    log_mixture_weights,
    log_prob,
    predict_mean_variance,
)


def test_hetero_gaussian_setting_matches_matlab_defaults():
    setting = hetero_gaussian_setting()

    assert setting.model_name == "HeteroGauss"
    assert setting.feature_names == ("Mean", "Variance")
    assert setting.link_types == ("identity", "log")
    assert setting.covs == ((0,), (0,))
    assert setting.covs_mix == (0,)
    assert setting.n_components == 2


def test_log_mixture_weights_reference_component():
    Z = jnp.ones((3, 1))
    gating_coef = jnp.array([[0.0]])

    log_weights = log_mixture_weights(gating_coef, Z)

    np.testing.assert_allclose(np.exp(log_weights), np.full((3, 2), 0.5))


def test_gaussian_mixture_predictive_moments_intercept_only():
    X = jnp.ones((4, 1))
    params = GaussianMixtureParams(
        mean_coef=jnp.array([[0.0], [2.0]]),
        log_variance_coef=jnp.log(jnp.array([[1.0], [1.0]])),
        gating_coef=jnp.array([[0.0]]),
    )

    pred_mean, pred_variance = predict_mean_variance(params, X, X, X)

    np.testing.assert_allclose(pred_mean, np.full(4, 1.0))
    np.testing.assert_allclose(pred_variance, np.full(4, 2.0))


def test_gaussian_mixture_log_prob_is_finite():
    X = jnp.ones((4, 1))
    y = jnp.array([0.0, 0.5, 1.5, 2.0])
    params = GaussianMixtureParams(
        mean_coef=jnp.array([[0.0], [2.0]]),
        log_variance_coef=jnp.log(jnp.array([[1.0], [1.0]])),
        gating_coef=jnp.array([[0.0]]),
    )

    assert np.isfinite(float(log_prob(params, y, X, X, X)))
