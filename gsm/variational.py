"""Minimal variational inference framework for the Gaussian mixture sketch.

The first inference path uses a collapsed-responsibility objective: allocations
are integrated out through ``logsumexp`` and the coefficient variational family
is currently a point mass. This is intentionally modest, but it gives us the
right API surface for replacing the old MATLAB Metropolis-Hastings/Newton path
with an ELBO-style JAX optimization loop.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import FitConfig, GaussianMixtureSetting, ModelConfig
from .data import Dataset, standardize_covariates
from .models.gaussian_mixture import (
    GaussianMixtureParams,
    log_prob as gaussian_log_prob,
    predict_mean_variance,
    responsibilities as gaussian_responsibilities,
)


@dataclass(frozen=True)
class GaussianMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_variance: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class VariationalResult:
    params: Any
    elbo_history: np.ndarray
    converged: bool
    responsibilities: np.ndarray | None = None
    predictive_mean: np.ndarray | None = None
    predictive_variance: np.ndarray | None = None


def prepare_gaussian_mixture_inputs(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
) -> GaussianMixtureInputs:
    """Build feature-specific design matrices from a loaded dataset."""

    X_mean = dataset.X[:, setting.covs[0]]
    X_variance = dataset.X[:, setting.covs[1]]
    Z = dataset.X[:, setting.covs_mix]

    if setting.standardize:
        X_mean, _, _ = standardize_covariates(X_mean, setting.standardize)
        X_variance, _, _ = standardize_covariates(X_variance, setting.standardize)
        Z, _, _ = standardize_covariates(Z, setting.standardize)

    return GaussianMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=X_mean,
        X_variance=X_variance,
        Z=Z,
    )


def initialize_gaussian_mixture_params(
    inputs: GaussianMixtureInputs,
    setting: GaussianMixtureSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the first VB optimizer."""

    y = np.asarray(inputs.y)
    n_components = setting.n_components
    quantiles = np.linspace(0.15, 0.85, n_components)
    means = np.quantile(y, quantiles)
    variance = max(float(np.var(y, ddof=1)), 1e-6)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = means
    log_variance_coef = np.zeros((n_components, inputs.X_variance.shape[1]))
    log_variance_coef[:, 0] = np.log(variance)
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "log_variance_coef": jnp.asarray(log_variance_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def tree_to_gaussian_params(tree: dict[str, jnp.ndarray]) -> GaussianMixtureParams:
    return GaussianMixtureParams(
        mean_coef=tree["mean_coef"],
        log_variance_coef=tree["log_variance_coef"],
        gating_coef=tree["gating_coef"],
    )


def gaussian_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: GaussianMixtureInputs,
    coefficient_prior_scale: float = 10.0,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO with Gaussian coefficient shrinkage."""

    params = tree_to_gaussian_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_variance = jnp.asarray(inputs.X_variance)
    Z = jnp.asarray(inputs.Z)

    log_likelihood = gaussian_log_prob(params, y, X_mean, X_variance, Z)
    log_prior = -0.5 * sum(
        jnp.sum((value / coefficient_prior_scale) ** 2) for value in param_tree.values()
    )
    return log_likelihood + log_prior


def fit_gaussian_mixture_vb(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the Gaussian mixture scaffold with JAX gradients and local Adam."""

    fit = fit or FitConfig()
    inputs = prepare_gaussian_mixture_inputs(dataset, setting)
    best_result: VariationalResult | None = None

    for restart in range(fit.n_restarts):
        params = initialize_gaussian_mixture_params(inputs, setting)
        if restart:
            params = _jitter_params(params, fit.seed + restart)
        params, history, converged = _optax_maximize(
            params,
            lambda p: gaussian_mixture_elbo(p, inputs),
            max_iter=fit.max_iter,
            learning_rate=fit.learning_rate,
            tol=fit.tol,
        )
        result = _build_result(params, inputs, history, converged)
        if best_result is None or result.elbo_history[-1] > best_result.elbo_history[-1]:
            best_result = result

    if best_result is None:
        raise RuntimeError("no variational optimization runs were executed")
    return best_result


def fit_variational(
    dataset: Dataset,
    model: ModelConfig | GaussianMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Dispatch to the available variational inference implementation."""

    if isinstance(model, GaussianMixtureSetting):
        return fit_gaussian_mixture_vb(dataset, model, fit)
    raise NotImplementedError(f"no variational implementation for model type {type(model)!r}")


def _optax_maximize(
    params: dict[str, jnp.ndarray],
    objective,
    max_iter: int,
    learning_rate: float,
    tol: float,
) -> tuple[dict[str, jnp.ndarray], np.ndarray, bool]:
    loss_and_grad = jax.value_and_grad(lambda p: -objective(p))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    history: list[float] = []
    converged = False

    for _ in range(1, max_iter + 1):
        loss, grad = loss_and_grad(params)
        history.append(float(-loss))
        if len(history) > 5 and abs(history[-1] - history[-2]) < tol:
            converged = True
            break

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

    return params, np.asarray(history), converged


def _build_result(
    param_tree: dict[str, jnp.ndarray],
    inputs: GaussianMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    params = tree_to_gaussian_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_variance = jnp.asarray(inputs.X_variance)
    Z = jnp.asarray(inputs.Z)
    resp = gaussian_responsibilities(params, y, X_mean, X_variance, Z)
    pred_mean, pred_var = predict_mean_variance(params, X_mean, X_variance, Z)
    return VariationalResult(
        params=params,
        elbo_history=history,
        converged=converged,
        responsibilities=np.asarray(resp),
        predictive_mean=np.asarray(pred_mean),
        predictive_variance=np.asarray(pred_var),
    )


def _jitter_params(params: dict[str, jnp.ndarray], seed: int) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: value + jnp.asarray(0.01 * rng.standard_normal(value.shape))
        for name, value in params.items()
    }
