import jax.numpy as jnp
import numpy as np
import pytest

from gsm.febama.inference import (
    active_beta_vector,
    fit_map,
    fit_vb,
    log_posterior,
    replace_active_beta,
    sample_febama_beta_posterior,
)
from gsm.febama.scoring import add_intercept


def _training_data():
    x = np.linspace(-2.0, 2.0, 9)
    features = np.asarray(add_intercept(x[:, None]))
    lpd = np.column_stack([2.0 * x, np.zeros_like(x)])
    return lpd, features


def test_febama_map_returns_finite_coefficients_and_improves_objective():
    lpd, features = _training_data()

    result = fit_map(lpd, features, coefficient_prior_scale=100.0)

    assert result.success
    assert result.beta.shape == (1, 2)
    assert result.active_mask.shape == (1, 2)
    assert np.isfinite(result.beta).all()
    assert result.objective > result.initial_objective
    assert result.beta[0, 1] > 0.0


def test_febama_map_keeps_all_features_active_by_default():
    lpd, features = _training_data()

    result = fit_map(lpd, features, coefficient_prior_scale=100.0)

    assert result.active_mask.dtype == bool
    assert result.active_mask.all()


def test_febama_active_mask_only_updates_active_coefficients():
    lpd, features = _training_data()
    initial_beta = np.asarray([[99.0, 0.0]])
    active_mask = np.asarray([[False, True]])

    result = fit_map(
        lpd,
        features,
        initial_beta=initial_beta,
        active_mask=active_mask,
        coefficient_prior_scale=100.0,
    )

    assert result.success
    assert result.beta[0, 0] == 0.0
    assert result.beta[0, 1] > 0.0


def test_febama_active_beta_pack_and_replace_roundtrip():
    beta = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = np.asarray([[True, False, True], [False, True, False]])

    packed = active_beta_vector(beta, mask)
    replaced = replace_active_beta(beta, mask, packed)

    np.testing.assert_allclose(packed, np.asarray([1.0, 3.0, 5.0]))
    np.testing.assert_allclose(
        replaced,
        np.asarray([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]]),
    )


def test_febama_log_posterior_includes_gaussian_prior_penalty():
    lpd, features = _training_data()
    beta = jnp.asarray([[0.0, 1.0]])

    weak_prior = log_posterior(lpd, features, beta, coefficient_prior_scale=100.0)
    strong_prior = log_posterior(lpd, features, beta, coefficient_prior_scale=0.5)

    assert float(strong_prior) < float(weak_prior)
    with pytest.raises(ValueError, match="positive"):
        log_posterior(lpd, features, beta, coefficient_prior_scale=0.0)


def test_febama_map_validates_shapes():
    lpd, features = _training_data()

    with pytest.raises(ValueError, match="initial_beta shape"):
        fit_map(lpd, features, initial_beta=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="active_mask shape"):
        fit_map(lpd, features, active_mask=np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="same number of rows"):
        fit_map(lpd[:-1], features)


def test_febama_vb_returns_finite_posterior_and_history():
    lpd, features = _training_data()

    result = fit_vb(
        lpd,
        features,
        coefficient_prior_scale=100.0,
        max_iter=40,
        learning_rate=0.05,
        n_elbo_samples=4,
        seed=321,
    )

    assert result.success
    assert result.beta.shape == (1, 2)
    assert result.posterior.mean.shape == (1, 2)
    assert result.posterior.log_std.shape == (1, 2)
    assert result.active_mask.shape == (1, 2)
    assert result.elbo_history.shape[0] >= 1
    assert np.isfinite(result.elbo_history).all()
    assert np.isfinite(result.posterior.mean).all()
    assert np.isfinite(result.posterior.log_std).all()


def test_febama_vb_respects_active_mask_and_samples_coefficients():
    lpd, features = _training_data()
    active_mask = np.asarray([[False, True]])

    result = fit_vb(
        lpd,
        features,
        active_mask=active_mask,
        coefficient_prior_scale=100.0,
        max_iter=20,
        learning_rate=0.05,
        n_elbo_samples=3,
        seed=123,
    )
    samples = sample_febama_beta_posterior(result.posterior, seed=123, n_samples=5)

    assert result.beta[0, 0] == 0.0
    assert samples.shape == (5, 1, 2)
    np.testing.assert_allclose(samples[:, 0, 0], np.zeros(5))
    assert np.isfinite(samples).all()


def test_febama_vb_validates_sampling_count():
    lpd, features = _training_data()
    result = fit_vb(lpd, features, max_iter=5, n_elbo_samples=2)

    with pytest.raises(ValueError, match="n_samples must be positive"):
        sample_febama_beta_posterior(result.posterior, n_samples=0)
