import jax.numpy as jnp
from jax.nn import logsumexp
import numpy as np
import pytest

from gsm.febama.data import LpdFeatures
from gsm.febama.scoring import (
    add_intercept,
    linear_predictors,
    logscore,
    log_weights,
    responsibilities,
)


def test_febama_log_weights_use_last_component_as_baseline():
    features = jnp.asarray([[1.0, -1.0], [1.0, 0.5], [1.0, 2.0]])
    beta = jnp.asarray([[0.25, -0.5], [-0.1, 0.2]])

    actual_eta = linear_predictors(beta, features)
    actual_log_weights = log_weights(beta, features)

    expected_eta = jnp.column_stack([features @ beta.T, jnp.zeros(3)])
    expected_log_weights = expected_eta - logsumexp(expected_eta, axis=1, keepdims=True)

    np.testing.assert_allclose(actual_eta, expected_eta)
    np.testing.assert_allclose(actual_log_weights, expected_log_weights)
    np.testing.assert_allclose(np.exp(np.asarray(actual_log_weights)).sum(axis=1), np.ones(3))


def test_febama_logscore_matches_manual_logsumexp_calculation():
    features = jnp.asarray([[1.0, -1.0], [1.0, 0.5], [1.0, 2.0]])
    beta = jnp.asarray([[0.25, -0.5], [-0.1, 0.2]])
    lpd = jnp.log(
        jnp.asarray(
            [
                [0.20, 0.30, 0.50],
                [0.10, 0.70, 0.20],
                [0.60, 0.15, 0.25],
            ]
        )
    )

    actual = logscore(lpd, features, beta, sum=False)

    eta = jnp.column_stack([features @ beta.T, jnp.zeros(3)])
    log_w = eta - logsumexp(eta, axis=1, keepdims=True)
    expected = logsumexp(log_w + lpd, axis=1)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(logscore(lpd, features, beta), jnp.sum(expected))


def test_febama_responsibilities_are_normalized():
    features = jnp.asarray([[1.0, 0.0], [1.0, 1.0]])
    beta = jnp.asarray([[0.2, -0.1], [0.4, 0.3]])
    lpd = jnp.log(jnp.asarray([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]]))

    resp = responsibilities(lpd, features, beta)

    assert resp.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(resp).sum(axis=1), np.ones(2))


def test_febama_add_intercept_prepends_ones():
    features = jnp.asarray([[2.0, 3.0], [4.0, 5.0]])

    actual = add_intercept(features)

    np.testing.assert_allclose(actual, jnp.asarray([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]]))


def test_febama_scoring_validates_shapes():
    features = jnp.ones((3, 2))
    beta = jnp.ones((1, 3))
    lpd = jnp.ones((3, 2))

    with pytest.raises(ValueError, match="beta columns"):
        logscore(lpd, features, beta)

    with pytest.raises(ValueError, match="at least two"):
        logscore(jnp.ones((3, 1)), features, jnp.ones((1, 2)))


def test_febama_lpd_features_container_validates_shapes():
    good = LpdFeatures(
        lpd=np.ones((3, 2)),
        features=np.ones((3, 4)),
        model_names=("a", "b"),
    )

    assert good.lpd.shape == (3, 2)
    with pytest.raises(ValueError, match="same number of rows"):
        LpdFeatures(lpd=np.ones((3, 2)), features=np.ones((2, 4)))
    with pytest.raises(ValueError, match="model_names"):
        LpdFeatures(lpd=np.ones((3, 2)), features=np.ones((3, 4)), model_names=("a",))
