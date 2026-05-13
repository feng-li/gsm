"""MAP inference for FEBAMA softmax gating coefficients."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import value_and_grad
import numpy as np
from scipy.optimize import minimize

from gsm.febama.scoring import logscore


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
