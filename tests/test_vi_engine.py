from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gsm.vi.common import VariationalResult
from gsm.vi.engine import (
    initialize_mean_field_variational_params,
    monte_carlo_variational_elbo,
    optax_maximize,
    optimize_restarts,
)


def _result(params, history, converged):
    return VariationalResult(params=params, elbo_history=history, converged=converged)


def _options(**kwargs):
    return SimpleNamespace(**dict(
        max_iter=kwargs.get("max_iter", 5), learning_rate=0.03, tol=0.0,
        n_elbo_samples=kwargs.get("n_elbo_samples", 2),
        n_validation_samples=kwargs.get("n_validation_samples", 32),
        n_restarts=kwargs.get("n_restarts", 1), seed=123,
    ))


def test_optimizer_history_describes_returned_parameters():
    objective = lambda p: -(p["x"] - 2.0) ** 2
    params, history, _ = optax_maximize({"x": jnp.array(0.0)}, objective, 1, 0.1, 0.0)
    assert params["x"] > 0
    assert len(history) == 2
    np.testing.assert_allclose(history[-1], objective(params))


def test_optimizer_retains_finite_parameters_after_invalid_step():
    objective = lambda p: jnp.log(p["x"])
    params, history, converged = optax_maximize(
        {"x": jnp.array(0.01)}, objective, 5, 1e308, 0.0,
    )
    assert np.isfinite(params["x"])
    assert np.isfinite(history).all()
    np.testing.assert_allclose(history[-1], objective(params))
    assert not converged


def test_optimizer_reports_actual_iterations_and_checkpoint_steps():
    diagnostics = {}
    params, history, _ = optax_maximize(
        {"x": jnp.array(0.)}, lambda p: -(p["x"] - 2.)**2,
        12, .03, 0., validation_interval=5, diagnostics=diagnostics,
    )
    assert diagnostics["n_iter"] == 12
    assert diagnostics["best_iteration"] == 12
    assert diagnostics["termination_reason"] == "maximum iterations reached"
    np.testing.assert_array_equal(diagnostics["elbo_iterations"], [0, 5, 10, 12])
    assert len(history) == 4
    np.testing.assert_allclose(history[-1], -(params["x"] - 2.)**2)


def test_optimizer_retains_actual_work_when_best_is_initial():
    diagnostics = {}
    _, history, _ = optax_maximize(
        {"x": jnp.array(0.)}, lambda p: -(p["x"] - 1.)**2, 7, .1, 0.,
        validation_objective=lambda p: -p["x"]**2, diagnostics=diagnostics,
    )
    assert diagnostics["n_iter"] == 7
    assert diagnostics["best_iteration"] == 0
    assert len(history) == 1


def test_convergence_describes_the_returned_checkpoint():
    diagnostics = {}
    params, _, converged = optax_maximize(
        {"x": jnp.array(0.)}, lambda p: -(p["x"] - 1.)**2, 250, .05, 1e-5,
        validation_objective=lambda p: (p["x"] - 1.)**2, diagnostics=diagnostics,
    )
    assert params["x"] == 0.
    assert not converged
    assert diagnostics["n_iter"] == 250
    assert diagnostics["termination_reason"] == "maximum iterations reached"


@pytest.mark.parametrize("max_iter, rate, tol", [(0, .1, 0.), (1, np.nan, 0.), (1, .1, -1.)])
def test_optimizer_rejects_invalid_controls(max_iter, rate, tol):
    with pytest.raises(ValueError):
        optax_maximize({"x": jnp.array(0.)}, lambda p: -p["x"]**2, max_iter, rate, tol)


def test_restart_selection_skips_nan_initialization():
    starts = iter([jnp.nan, 0.0])

    def initialize():
        return initialize_mean_field_variational_params({"x": jnp.array(next(starts))}, -1.)

    result = optimize_restarts(
        _options(n_restarts=2), initialize,
        lambda noise: lambda q: monte_carlo_variational_elbo(
            q, noise, lambda p: -0.5 * p["x"]**2,
        ), _result,
    )
    assert np.isfinite(result.elbo_history).all()
    assert np.isfinite(result.validation_elbo)
    assert np.isfinite(result.elbo_mc_error)


def test_all_failed_restarts_raise_clear_error():
    with pytest.raises(RuntimeError, match="non-finite"):
        optimize_restarts(
            _options(),
            lambda: initialize_mean_field_variational_params({"x": jnp.array(jnp.nan)}, -1.),
            lambda noise: lambda q: q["mean"]["x"], _result,
        )


def test_restarts_share_validation_draws(monkeypatch):
    import gsm.vi.engine as engine

    scores = []
    seeds = []

    def optimize(params, objective, **kwargs):
        reference = initialize_mean_field_variational_params({"x": jnp.array(0.)}, 0.)
        score = float(kwargs["validation_objective"](reference))
        scores.append(score)
        seeds.append(kwargs["seed"])
        return reference, np.array([score]), False

    monkeypatch.setattr(engine, "optax_maximize", optimize)
    optimize_restarts(
        _options(n_restarts=2),
        lambda: initialize_mean_field_variational_params({"x": jnp.array(0.)}, 0.),
        lambda noise: lambda q: monte_carlo_variational_elbo(
            q, noise, lambda p: -0.5 * p["x"]**2,
        ), _result,
    )
    assert scores[0] == scores[1]
    assert seeds[0] != seeds[1]


def test_single_sample_stochastic_vi_recovers_normal_target():
    result = optimize_restarts(
        _options(max_iter=500, n_elbo_samples=1, n_validation_samples=128),
        lambda: initialize_mean_field_variational_params({"x": jnp.array(1.5)}, 0.),
        lambda noise: lambda q: monte_carlo_variational_elbo(
            q, noise, lambda p: -0.5 * (p["x"]**2 + jnp.log(2 * jnp.pi)),
        ), _result,
    )
    mean = float(result.params["mean"]["x"])
    sd = float(jnp.exp(result.params["log_std"]["x"]))
    assert abs(mean) < 0.3
    assert 0.7 < sd < 1.3
    true_elbo = 0.5 * (1 + 2 * np.log(sd) - mean**2 - sd**2)
    assert abs(result.validation_elbo - true_elbo) < 6 * result.elbo_mc_error


def test_stochastic_optimizer_draws_new_noise_each_step():
    target = lambda p, noise: -(p["x"] - noise)**2
    params, _, _ = optax_maximize(
        {"x": jnp.array(3.)}, target, 500, 0.03, 0.,
        noise_sampler=lambda key: jax.random.normal(key), seed=7,
        validation_objective=lambda p: -p["x"]**2,
    )
    assert abs(float(params["x"])) < 0.1


def test_analytical_prior_recovers_normal_with_one_training_draw():
    from gsm.config import FitConfig
    from gsm.priors import (
        GaussianMixtureCoefficientPriors,
        build_gaussian_coefficient_prior,
        gaussian_matrix_log_prob,
    )
    from gsm.vi.engine import fit_mean_field_mixture_vb

    prior = build_gaussian_coefficient_prior(np.ones((3, 1)), 2., 2., "identity")

    def elbo(q, inputs, noise, **kwargs):
        return monte_carlo_variational_elbo(
            q, noise, lambda p: gaussian_matrix_log_prob(p["mean_coef"], prior),
        )

    result = fit_mean_field_mixture_vb(
        FitConfig(max_iter=250, learning_rate=.04, n_restarts=1, n_elbo_samples=1,
                  n_validation_samples=2, n_predictive_samples=0, tol=0.),
        None,
        lambda: initialize_mean_field_variational_params({"mean_coef": jnp.zeros((1, 1))}, -.7),
        elbo, GaussianMixtureCoefficientPriors(prior, None, None), _result,
    )
    np.testing.assert_allclose(result.params["mean"]["mean_coef"], [[2.]], atol=.01)
    np.testing.assert_allclose(jnp.exp(result.params["log_std"]["mean_coef"]), [[2.]], atol=.01)
    assert result.elbo_mc_error < 1e-10
