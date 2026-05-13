import jax.numpy as jnp
import numpy as np
import pytest

from gsm.febama import FebamaConfig, PredictiveDistribution
from gsm.febama.distributions import (
    available_distributions,
    get_distribution,
    log_prob_matrix,
)


def test_febama_distribution_registry_contains_core_choices():
    names = available_distributions()

    assert "gaussian" in names
    assert "studentt" in names
    assert "splitnormal" in names
    assert "splitt" in names
    assert get_distribution("student_t") is get_distribution("studentt")


def test_febama_distribution_registry_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown FEBAMA distribution"):
        get_distribution("not_a_distribution")

    with pytest.raises(ValueError, match="unknown FEBAMA distribution"):
        FebamaConfig(
            model_names=("a", "b"),
            distribution_names=("gaussian", "not_a_distribution"),
        )


def test_febama_gaussian_prediction_matches_r_dnorm_path():
    y = jnp.asarray([-1.0, 0.0, 2.0])
    prediction = PredictiveDistribution(
        "gaussian",
        {"mean": jnp.zeros(3), "sd": jnp.full(3, 2.0)},
    )

    actual = np.asarray(prediction.log_prob(y))
    expected = -0.5 * np.log(2.0 * np.pi * 4.0) - (np.asarray(y) ** 2) / 8.0

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(prediction.mean(), np.zeros(3))
    np.testing.assert_allclose(prediction.variance(), np.full(3, 4.0))


def test_febama_multiple_real_valued_distributions_produce_lpd_matrix():
    y = jnp.asarray([-1.0, 0.0, 1.0])
    predictions = (
        PredictiveDistribution("gaussian", {"mean": 0.0, "sd": 1.0}),
        PredictiveDistribution("studentt", {"mean": 0.0, "scale": 1.0, "df": 5.0}),
        PredictiveDistribution(
            "splitnormal",
            {"mean": 0.0, "scale": 1.0, "skewness": 1.2},
        ),
        PredictiveDistribution(
            "splitt",
            {"mean": 0.0, "scale": 1.0, "df": 7.0, "skewness": 0.8},
        ),
    )

    lpd = log_prob_matrix(y, predictions)

    assert lpd.shape == (3, 4)
    assert np.isfinite(np.asarray(lpd)).all()


def test_febama_distribution_support_is_enforced_by_adapters():
    y_bad_positive = jnp.asarray([0.0, -1.0])
    lognormal_pred = PredictiveDistribution(
        "lognormal",
        {"mean": 0.0, "scale": 1.0},
    )
    gamma_pred = PredictiveDistribution(
        "gamma",
        {"mean": 1.0, "variance": 2.0},
    )
    poisson_pred = PredictiveDistribution(
        "poisson",
        {"mean": 1.5},
    )

    assert np.isneginf(np.asarray(lognormal_pred.log_prob(y_bad_positive))).all()
    assert np.isneginf(np.asarray(gamma_pred.log_prob(y_bad_positive))).all()
    assert np.isneginf(np.asarray(poisson_pred.log_prob(jnp.asarray([-1.0, 1.5])))).all()


def test_febama_config_validates_component_lengths():
    config = FebamaConfig(
        model_names=("gauss", "t"),
        distribution_names=("gaussian", "studentt"),
    )

    assert config.add_intercept is True
    with pytest.raises(ValueError, match="at least two"):
        FebamaConfig(model_names=("gauss",), distribution_names=("gaussian",))
    with pytest.raises(ValueError, match="must match"):
        FebamaConfig(model_names=("gauss", "t"), distribution_names=("gaussian",))
