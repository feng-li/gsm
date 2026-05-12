import jax.numpy as jnp
import numpy as np

from gsm.models.exphazard import (
    ExpHazardMixtureParams,
    component_log_prob as exphazard_component_log_prob,
    log_prob as exphazard_log_prob,
    log_prob_observations as exphazard_log_prob_observations,
    predict_survival_probability as exphazard_predict_survival_probability,
    responsibilities as exphazard_responsibilities,
)
from gsm.models.weibull_survival import (
    WeibullSurvivalMixtureParams,
    component_log_prob as weibull_component_log_prob,
    log_prob as weibull_log_prob,
    log_prob_observations as weibull_log_prob_observations,
    predict_survival_probability as weibull_predict_survival_probability,
    responsibilities as weibull_responsibilities,
)


def test_exphazard_component_log_prob_matches_archive_formula():
    t0 = jnp.array([0.0, 1.0])
    t = jnp.array([1.0, 3.0])
    event = jnp.array([0.0, 1.0])
    rate = jnp.array([[0.5], [0.5]])

    actual = exphazard_component_log_prob(t, t0, event, rate)

    increment = rate * (t.reshape((-1, 1)) - t0.reshape((-1, 1)))
    expected = jnp.array([[-0.5], [jnp.log1p(-jnp.exp(-1.0))]])
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(exphazard_log_prob(t, t0, event, rate), jnp.sum(expected))
    np.testing.assert_allclose(
        exphazard_predict_survival_probability(
            ExpHazardMixtureParams(jnp.log(jnp.array([[0.5]])), jnp.zeros((0, 1))),
            t,
            t0,
            jnp.ones((2, 1)),
            jnp.ones((2, 1)),
        ),
        jnp.exp(-increment).reshape((-1,)),
    )


def test_weibull_component_log_prob_matches_archive_formula():
    t0 = jnp.array([0.0, 1.0])
    t = jnp.array([1.0, 3.0])
    event = jnp.array([0.0, 1.0])
    rate = jnp.array([[0.5], [0.5]])
    shape = jnp.array([[2.0], [2.0]])

    actual = weibull_component_log_prob(t, t0, event, rate, shape)

    increment = rate * (t.reshape((-1, 1)) ** shape - t0.reshape((-1, 1)) ** shape)
    expected = jnp.array([[-0.5], [jnp.log1p(-jnp.exp(-4.0))]])
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        weibull_log_prob(t, t0, event, rate, shape),
        jnp.sum(expected),
    )
    np.testing.assert_allclose(
        weibull_predict_survival_probability(
            WeibullSurvivalMixtureParams(
                rate_coef=jnp.log(jnp.array([[0.5]])),
                shape_coef=jnp.log(jnp.array([[2.0]])),
                gating_coef=jnp.zeros((0, 1)),
            ),
            t,
            t0,
            jnp.ones((2, 1)),
            jnp.ones((2, 1)),
            jnp.ones((2, 1)),
        ),
        jnp.exp(-increment).reshape((-1,)),
    )


def test_survival_component_invalid_support_returns_negative_infinity():
    t0 = jnp.array([1.0, 0.0, 0.0])
    t = jnp.array([0.5, 1.0, 1.0])
    event = jnp.array([0.0, 0.5, 2.0])
    rate = jnp.ones((3, 1))
    shape = jnp.ones((3, 1))

    exp_actual = exphazard_component_log_prob(t, t0, event, rate)
    weibull_actual = weibull_component_log_prob(t, t0, event, rate, shape)

    assert np.isneginf(np.asarray(exp_actual)).all()
    assert np.isneginf(np.asarray(weibull_actual)).all()


def test_survival_mixture_responsibilities_and_log_prob_shapes():
    t0 = jnp.array([0.0, 1.0, 2.0])
    t = jnp.array([1.0, 2.0, 4.0])
    event = jnp.array([0.0, 1.0, 0.0])
    X = jnp.ones((3, 1))
    Z = jnp.ones((3, 1))
    exp_params = ExpHazardMixtureParams(
        rate_coef=jnp.log(jnp.array([[0.5], [1.0]])),
        gating_coef=jnp.zeros((1, 1)),
    )
    weibull_params = WeibullSurvivalMixtureParams(
        rate_coef=jnp.log(jnp.array([[0.5], [1.0]])),
        shape_coef=jnp.log(jnp.array([[1.0], [1.5]])),
        gating_coef=jnp.zeros((1, 1)),
    )

    exp_resp = exphazard_responsibilities(exp_params, t, t0, event, X, Z)
    weibull_resp = weibull_responsibilities(weibull_params, t, t0, event, X, X, Z)

    assert exp_resp.shape == (3, 2)
    assert weibull_resp.shape == (3, 2)
    np.testing.assert_allclose(exp_resp.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(weibull_resp.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(
        exphazard_log_prob(exp_params, t, t0, event, X, Z),
        jnp.sum(exphazard_log_prob_observations(exp_params, t, t0, event, X, Z)),
    )
    np.testing.assert_allclose(
        weibull_log_prob(weibull_params, t, t0, event, X, X, Z),
        jnp.sum(
            weibull_log_prob_observations(weibull_params, t, t0, event, X, X, Z)
        ),
    )


def test_survival_event_log_probability_is_stable_for_large_increment():
    t0 = jnp.array([0.0])
    t = jnp.array([1000.0])
    event = jnp.array([1.0])
    rate = jnp.ones((1, 1))
    shape = jnp.ones((1, 1))

    assert np.isfinite(float(exphazard_component_log_prob(t, t0, event, rate)[0, 0]))
    assert np.isfinite(float(weibull_component_log_prob(t, t0, event, rate, shape)[0, 0]))
