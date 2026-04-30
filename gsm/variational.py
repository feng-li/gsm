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
class GaussianMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    X_variance_c1: np.ndarray
    X_variance_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray


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
    standardization: GaussianMixtureStandardization | None = None,
) -> GaussianMixtureInputs:
    """Build feature-specific design matrices from a loaded dataset."""

    X_mean, X_variance, Z = _raw_gaussian_mixture_designs(dataset, setting)

    if setting.standardize:
        if standardization is None:
            X_mean, _, _ = standardize_covariates(X_mean, setting.standardize)
            X_variance, _, _ = standardize_covariates(X_variance, setting.standardize)
            Z, _, _ = standardize_covariates(Z, setting.standardize)
        else:
            X_mean = _apply_standardization(
                X_mean,
                setting.standardize,
                standardization.X_mean_c1,
                standardization.X_mean_c2,
            )
            X_variance = _apply_standardization(
                X_variance,
                setting.standardize,
                standardization.X_variance_c1,
                standardization.X_variance_c2,
            )
            Z = _apply_standardization(
                Z,
                setting.standardize,
                standardization.Z_c1,
                standardization.Z_c2,
            )

    return GaussianMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=X_mean,
        X_variance=X_variance,
        Z=Z,
    )


def fit_gaussian_mixture_standardization(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
) -> GaussianMixtureStandardization:
    """Fit the model-specific scaling constants used by the design matrices."""

    X_mean, X_variance, Z = _raw_gaussian_mixture_designs(dataset, setting)
    _, X_mean_c1, X_mean_c2 = standardize_covariates(X_mean, setting.standardize)
    _, X_variance_c1, X_variance_c2 = standardize_covariates(X_variance, setting.standardize)
    _, Z_c1, Z_c2 = standardize_covariates(Z, setting.standardize)
    return GaussianMixtureStandardization(
        X_mean_c1=X_mean_c1,
        X_mean_c2=X_mean_c2,
        X_variance_c1=X_variance_c1,
        X_variance_c2=X_variance_c2,
        Z_c1=Z_c1,
        Z_c2=Z_c2,
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
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO with optional ARD coefficient shrinkage."""

    params = tree_to_gaussian_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_variance = jnp.asarray(inputs.X_variance)
    Z = jnp.asarray(inputs.Z)

    log_likelihood = gaussian_log_prob(params, y, X_mean, X_variance, Z)
    log_prior = (
        _coefficient_log_prior(
            param_tree["mean_coef"],
            inputs.X_mean,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
        )
        + _coefficient_log_prior(
            param_tree["log_variance_coef"],
            inputs.X_variance,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
        )
        + _coefficient_log_prior(
            param_tree["gating_coef"],
            inputs.Z,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
        )
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
            lambda p: gaussian_mixture_elbo(
                p,
                inputs,
                coefficient_prior_scale=fit.coefficient_prior_scale,
                use_ard=fit.use_ard,
                ard_shape=fit.ard_shape,
                ard_rate=fit.ard_rate,
            ),
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


def _raw_gaussian_mixture_designs(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs_mix],
    )


def _apply_standardization(
    X: np.ndarray,
    method: int,
    c1: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    if X.shape[1] != c1.shape[0] or X.shape[1] != c2.shape[0]:
        raise ValueError("standardization constants do not match design matrix")
    if method == 0:
        return X

    use_column = ~(np.isnan(c1) | np.isnan(c2))
    if not np.any(use_column):
        return X

    if method == 1:
        X[:, use_column] = (X[:, use_column] - c1[use_column]) / c2[use_column]
    elif method == 2:
        scale = 2.0 / (c2[use_column] - c1[use_column])
        offset = 1.0 - scale * c2[use_column]
        X[:, use_column] = X[:, use_column] * scale + offset
    elif method == 3:
        X[:, use_column] = X[:, use_column] - c1[use_column]
    else:
        raise ValueError(f"unknown standardization method: {method}")
    return X


def _coefficient_log_prior(
    value: jnp.ndarray,
    design_matrix: np.ndarray,
    coefficient_prior_scale: float,
    use_ard: bool,
    ard_shape: float,
    ard_rate: float,
) -> jnp.ndarray:
    if coefficient_prior_scale <= 0:
        raise ValueError("coefficient_prior_scale must be positive")
    if not use_ard:
        return -0.5 * jnp.sum((value / coefficient_prior_scale) ** 2)
    if ard_shape <= 0 or ard_rate <= 0:
        raise ValueError("ard_shape and ard_rate must be positive")
    if value.size == 0:
        return jnp.asarray(0.0, dtype=value.dtype)

    constant_mask = _constant_column_mask(design_matrix)
    if constant_mask.shape[0] != value.shape[-1]:
        raise ValueError("design matrix and coefficient matrix column counts differ")

    constant_weights = jnp.asarray(constant_mask, dtype=value.dtype)
    constant_prior = -0.5 * jnp.sum(
        ((value / coefficient_prior_scale) ** 2) * constant_weights
    )

    column_squares = jnp.sum(value**2, axis=0)
    group_size = value.shape[0]
    nonconstant_weights = 1.0 - constant_weights
    ard_prior = -jnp.sum(
        nonconstant_weights
        * (ard_shape + 0.5 * group_size)
        * jnp.log(ard_rate + 0.5 * column_squares)
    )
    return constant_prior + ard_prior


def _constant_column_mask(design_matrix: np.ndarray) -> np.ndarray:
    X = np.asarray(design_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("design matrix must be 2D")
    if X.shape[0] == 0:
        return np.zeros(X.shape[1], dtype=bool)
    return np.all(np.isclose(X, X[:1, :]), axis=0)


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
