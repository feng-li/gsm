"""MAP and VB inference for FEBAMA softmax gating coefficients."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import value_and_grad
import numpy as np
from scipy.optimize import minimize

from gsm.febama.scoring import logscore
from gsm.vi.engine import (
    initialize_mean_field_variational_params,
    monte_carlo_variational_elbo,
    optimize_restarts,
)


@dataclass(frozen=True)
class FebamaMapResult:
    """Result from MAP fitting of FEBAMA gating coefficients."""

    beta: np.ndarray
    active_mask: np.ndarray
    initial_objective: float
    objective: float
    success: bool
    message: str
    n_iter: int


@dataclass(frozen=True)
class FebamaVbPosterior:
    """Mean-field Gaussian posterior over FEBAMA gating coefficients."""

    mean: np.ndarray
    log_std: np.ndarray
    active_mask: np.ndarray


@dataclass(frozen=True)
class FebamaVbResult:
    """Result from mean-field VB fitting of FEBAMA gating coefficients."""

    beta: np.ndarray
    posterior: FebamaVbPosterior
    active_mask: np.ndarray
    elbo_history: np.ndarray
    converged: bool
    objective: float
    success: bool
    message: str
    n_iter: int


@dataclass(frozen=True)
class _FebamaVbFitOptions:
    max_iter: int
    learning_rate: float
    tol: float
    n_elbo_samples: int
    n_restarts: int
    seed: int


def fit_map(
    lpd,
    features,
    initial_beta=None,
    active_mask=None,
    coefficient_prior_scale: float = 10.0,
    max_iter: int = 1000,
) -> FebamaMapResult:
    """Fit FEBAMA gating coefficients by MAP/BFGS."""

    lpd_array, features_array = _validate_training_arrays(lpd, features)
    beta0 = _initial_beta(lpd_array, features_array, initial_beta)
    mask = _active_mask(beta0, active_mask)
    beta0 = np.where(mask, beta0, 0.0)
    par0 = active_beta_vector(beta0, mask)
    initial_objective = float(
        log_posterior(lpd_array, features_array, beta0, coefficient_prior_scale)
    )

    if par0.size == 0:
        return FebamaMapResult(
            beta=beta0,
            active_mask=mask,
            initial_objective=initial_objective,
            objective=initial_objective,
            success=True,
            message="no active coefficients",
            n_iter=0,
        )

    template = np.zeros_like(beta0)
    active_indices = jnp.asarray(np.flatnonzero(mask.ravel()), dtype=jnp.int32)

    def objective_and_grad(par):
        par = jnp.asarray(par)
        value, grad = value_and_grad(_active_log_posterior)(
            par,
            lpd_array,
            features_array,
            template,
            active_indices,
            coefficient_prior_scale,
        )
        return -float(value), -np.asarray(grad, dtype=float)

    opt = minimize(
        fun=lambda par: objective_and_grad(par)[0],
        x0=par0,
        jac=lambda par: objective_and_grad(par)[1],
        method="BFGS",
        options={"maxiter": max_iter},
    )
    beta = replace_active_beta(template, mask, opt.x)
    objective = float(log_posterior(lpd_array, features_array, beta, coefficient_prior_scale))
    return FebamaMapResult(
        beta=beta,
        active_mask=mask,
        initial_objective=initial_objective,
        objective=objective,
        success=bool(opt.success),
        message=str(opt.message),
        n_iter=int(opt.nit),
    )


def fit_vb(
    lpd,
    features,
    initial_beta=None,
    active_mask=None,
    coefficient_prior_scale: float = 10.0,
    max_iter: int = 1000,
    learning_rate: float = 1e-2,
    tol: float = 1e-6,
    n_elbo_samples: int = 8,
    n_restarts: int = 1,
    seed: int = 123,
    posterior_init_log_std: float = -5.0,
) -> FebamaVbResult:
    """Fit FEBAMA gating coefficients with a mean-field Gaussian posterior."""

    lpd_array, features_array = _validate_training_arrays(lpd, features)
    beta0 = _initial_beta(lpd_array, features_array, initial_beta)
    mask = _active_mask(beta0, active_mask)
    beta0 = np.where(mask, beta0, 0.0)
    active0 = active_beta_vector(beta0, mask)
    template = np.zeros_like(beta0)
    active_indices = jnp.asarray(np.flatnonzero(mask.ravel()), dtype=jnp.int32)

    if active0.size == 0:
        posterior = FebamaVbPosterior(
            mean=beta0,
            log_std=np.zeros_like(beta0),
            active_mask=mask,
        )
        objective = float(log_posterior(lpd_array, features_array, beta0, coefficient_prior_scale))
        return FebamaVbResult(
            beta=beta0,
            posterior=posterior,
            active_mask=mask,
            elbo_history=np.asarray([objective]),
            converged=True,
            objective=objective,
            success=True,
            message="no active coefficients",
            n_iter=0,
        )

    fit_options = _FebamaVbFitOptions(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        tol=float(tol),
        n_elbo_samples=int(n_elbo_samples),
        n_restarts=int(n_restarts),
        seed=int(seed),
    )

    def initialize_variational_params():
        mean_tree = {"active_beta": jnp.asarray(active0)}
        return initialize_mean_field_variational_params(mean_tree, posterior_init_log_std)

    def objective_for_noise(noise_tree):
        def objective(variational_tree):
            return monte_carlo_variational_elbo(
                variational_tree,
                noise_tree,
                lambda sample_tree: _active_log_posterior(
                    sample_tree["active_beta"],
                    lpd_array,
                    features_array,
                    template,
                    active_indices,
                    coefficient_prior_scale,
                ),
            )

        return objective

    def build_result(variational_tree, history, converged):
        mean_active = np.asarray(variational_tree["mean"]["active_beta"], dtype=float)
        log_std_active = np.asarray(variational_tree["log_std"]["active_beta"], dtype=float)
        beta_mean = replace_active_beta(template, mask, mean_active)
        log_std = replace_active_beta(template, mask, log_std_active)
        posterior = FebamaVbPosterior(
            mean=beta_mean,
            log_std=log_std,
            active_mask=mask,
        )
        history = np.asarray(history, dtype=float)
        finite = bool(np.all(np.isfinite(history)))
        return FebamaVbResult(
            beta=beta_mean,
            posterior=posterior,
            active_mask=mask,
            elbo_history=history,
            converged=bool(converged),
            objective=float(history[-1]),
            success=finite,
            message="converged" if converged else "maximum iterations reached",
            n_iter=int(history.shape[0]),
        )

    return optimize_restarts(
        fit_options,
        initialize_variational_params,
        objective_for_noise,
        build_result,
    )


def sample_febama_beta_posterior(
    posterior: FebamaVbPosterior,
    seed: int = 123,
    n_samples: int = 100,
) -> np.ndarray:
    """Draw full FEBAMA coefficient matrices from a fitted VB posterior."""

    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    mean = np.asarray(posterior.mean, dtype=float)
    log_std = np.asarray(posterior.log_std, dtype=float)
    mask = _active_mask(mean, posterior.active_mask)
    active_mean = mean[mask]
    if active_mean.size == 0:
        return np.repeat(mean[None, :, :], n_samples, axis=0)
    active_log_std = log_std[mask]
    noise = jax.random.normal(
        jax.random.PRNGKey(int(seed)),
        (n_samples, active_mean.size),
        dtype=jnp.asarray(active_mean).dtype,
    )
    active_samples = active_mean + np.exp(active_log_std) * np.asarray(noise)
    template = np.zeros_like(mean)
    return np.asarray(
        [replace_active_beta(template, mask, active_samples[idx]) for idx in range(n_samples)]
    )


def log_posterior(
    lpd,
    features,
    beta,
    coefficient_prior_scale: float = 10.0,
):
    """Return FEBAMA logscore plus an independent Gaussian coefficient prior."""

    scale = float(coefficient_prior_scale)
    if scale <= 0.0:
        raise ValueError("coefficient_prior_scale must be positive")
    beta = jnp.asarray(beta)
    return logscore(lpd, features, beta, sum=True) - 0.5 * jnp.sum((beta / scale) ** 2)


def active_beta_vector(beta, active_mask) -> np.ndarray:
    """Return active coefficients as a flat vector."""

    beta = np.asarray(beta, dtype=float)
    mask = _active_mask(beta, active_mask)
    return beta[mask]


def replace_active_beta(beta_template, active_mask, active_beta) -> np.ndarray:
    """Return a full coefficient matrix with inactive coefficients set to zero."""

    beta = np.zeros_like(np.asarray(beta_template, dtype=float))
    mask = _active_mask(beta, active_mask)
    active_beta = np.asarray(active_beta, dtype=float)
    if active_beta.size != int(np.sum(mask)):
        raise ValueError("active_beta length must match active_mask")
    beta[mask] = active_beta
    return beta


def _active_log_posterior(
    active_beta,
    lpd,
    features,
    beta_template,
    active_indices,
    coefficient_prior_scale,
):
    beta = _replace_active_beta_jax(beta_template, active_indices, active_beta)
    return log_posterior(lpd, features, beta, coefficient_prior_scale)


def _replace_active_beta_jax(beta_template, active_indices, active_beta):
    template = jnp.asarray(beta_template)
    flat = jnp.zeros(template.size, dtype=jnp.asarray(active_beta).dtype)
    flat = flat.at[active_indices].set(active_beta)
    return flat.reshape(template.shape)


def _validate_training_arrays(lpd, features):
    lpd_array = np.asarray(lpd, dtype=float)
    features_array = np.asarray(features, dtype=float)
    if lpd_array.ndim != 2:
        raise ValueError("lpd must be a 2D matrix")
    if features_array.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if lpd_array.shape[0] != features_array.shape[0]:
        raise ValueError("lpd and features must have the same number of rows")
    if lpd_array.shape[1] < 2:
        raise ValueError("lpd must contain at least two component columns")
    if not np.all(np.isfinite(features_array)):
        raise ValueError("features must be finite")
    if not np.all(np.any(np.isfinite(lpd_array), axis=1)):
        raise ValueError("each lpd row must contain at least one finite value")
    return lpd_array, features_array


def _initial_beta(lpd, features, initial_beta):
    n_nonbaseline = lpd.shape[1] - 1
    shape = (n_nonbaseline, features.shape[1])
    if initial_beta is None:
        return np.zeros(shape, dtype=float)
    beta = np.asarray(initial_beta, dtype=float)
    if beta.shape != shape:
        raise ValueError("initial_beta shape must be (n_components - 1, n_features)")
    return beta


def _active_mask(beta, active_mask):
    if active_mask is None:
        return np.ones_like(np.asarray(beta), dtype=bool)
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != np.asarray(beta).shape:
        raise ValueError("active_mask shape must match beta")
    return mask
