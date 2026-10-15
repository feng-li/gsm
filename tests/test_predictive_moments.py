import jax.numpy as jnp
import numpy as np
import pytest

from gsm.config import (
    FitConfig,
    LogNormalMixtureSetting,
    LogNormalRepMixtureSetting,
    SplitTMixtureSetting,
    StudentTMixtureSetting,
)
from gsm.data import Dataset
from gsm.variational import fit_variational
from gsm.vi.common import guard_studentt_moments


@pytest.mark.parametrize("setting, expected", [
    (StudentTMixtureSetting(n_components=1, covs=((0,),) * 3, covs_mix=(0,)), "undefined"),
    (SplitTMixtureSetting(n_components=1, covs=((0,),) * 4, covs_mix=(0,)), "undefined"),
    (LogNormalMixtureSetting(n_components=1), "infinite"),
    (LogNormalRepMixtureSetting(n_components=1), "finite"),
])
def test_posterior_moment_existence_is_not_decided_by_finite_draws(monkeypatch, setting, expected):
    import gsm.vi.engine as engine

    def optimize(fit, initialize, objective, build_result):
        q = initialize()
        q["log_std"] = {name: jnp.full_like(value, -5.) for name, value in q["mean"].items()}
        return build_result(q, np.array([0.]), False)

    monkeypatch.setattr(engine, "optimize_restarts", optimize)
    data = Dataset(np.array([[1.], [2.], [3.]]), np.ones((3, 1)), "y", ("Const",))
    result = fit_variational(data, setting, FitConfig(n_predictive_samples=8))
    if expected == "undefined":
        assert np.isnan(result.predictive_mean).all()
        assert np.isnan(result.predictive_variance).all()
        assert result.predictive_moment_warning
    elif expected == "infinite":
        assert np.isposinf(result.predictive_mean).all()
        assert np.isposinf(result.predictive_variance).all()
        assert result.predictive_moment_warning
    else:
        assert np.isfinite(result.predictive_mean).all()
        assert np.isfinite(result.predictive_variance).all()
        assert result.predictive_moment_warning is None


def test_deterministic_studentt_df_has_finite_moments():
    from gsm.models.studentt import StudentTMixtureParams
    from gsm.vi.studentt import StudentTMixturePosterior

    params = StudentTMixtureParams(
        jnp.zeros((1, 1)), jnp.full((1, 1), np.log(10.)),
        jnp.zeros((1, 1)), jnp.zeros((0, 1)),
    )
    log_std = StudentTMixtureParams(
        jnp.full((1, 1), -5.), jnp.full((1, 1), -jnp.inf),
        jnp.full((1, 1), -5.), jnp.zeros((0, 1)),
    )
    mean, variance, warning = guard_studentt_moments(
        StudentTMixturePosterior(params, log_std), np.ones((2, 1)), np.zeros(2), np.full(2, 1.25),
    )
    np.testing.assert_allclose(mean, 0.)
    np.testing.assert_allclose(variance, 1.25)
    assert warning is None
