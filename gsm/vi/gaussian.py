"""Variational inference for Gaussian mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import FitConfig, GaussianMixtureSetting
from gsm.data import Dataset
from gsm.models.gaussian import (
    GaussianMixtureParams,
    log_prob as gaussian_log_prob,
    predict_mean_variance,
    responsibilities as gaussian_responsibilities,
)
from gsm.priors import (
    GaussianMixtureCoefficientPriors,
    build_gaussian_mixture_priors,
    coefficient_log_prior_sum,
)
from gsm.vi.common import (
    VariationalResult,
    fit_design_standardization,
    standardize_designs,
)
from gsm.vi.engine import (
    fit_mean_field_mixture_vb,
    initialize_mean_field_variational_params,
    monte_carlo_variational_elbo,
    sample_posterior_tree,
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
class GaussianMixturePosterior:
    """Mean-field Gaussian posterior over Gaussian-mixture coefficients."""

    mean: GaussianMixtureParams
    log_std: GaussianMixtureParams




def prepare_gaussian_mixture_inputs(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
    standardization: GaussianMixtureStandardization | None = None,
) -> GaussianMixtureInputs:
    """Build feature-specific design matrices from a loaded dataset."""

    X_mean, X_variance, Z = _raw_gaussian_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {"X_mean": X_mean, "X_variance": X_variance, "Z": Z},
        setting.standardize,
        standardization,
    )

    return GaussianMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=designs["X_mean"],
        X_variance=designs["X_variance"],
        Z=designs["Z"],
    )



def fit_gaussian_mixture_standardization(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
) -> GaussianMixtureStandardization:
    """Fit the model-specific scaling constants used by the design matrices."""

    X_mean, X_variance, Z = _raw_gaussian_mixture_designs(dataset, setting)
    return GaussianMixtureStandardization(
        **fit_design_standardization(
            {"X_mean": X_mean, "X_variance": X_variance, "Z": Z},
            setting.standardize,
        ),
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


def initialize_gaussian_mixture_variational_params(
    inputs: GaussianMixtureInputs,
    setting: GaussianMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family."""

    mean_tree = initialize_gaussian_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)



def tree_to_gaussian_params(tree: dict[str, jnp.ndarray]) -> GaussianMixtureParams:
    return GaussianMixtureParams(
        mean_coef=tree["mean_coef"],
        log_variance_coef=tree["log_variance_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_gaussian_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> GaussianMixturePosterior:
    return GaussianMixturePosterior(
        mean=tree_to_gaussian_params(tree["mean"]),
        log_std=tree_to_gaussian_params(tree["log_std"]),
    )



def gaussian_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: GaussianMixtureInputs,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: GaussianMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO with optional ARD coefficient shrinkage."""

    params = tree_to_gaussian_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_variance = jnp.asarray(inputs.X_variance)
    Z = jnp.asarray(inputs.Z)

    log_likelihood = gaussian_log_prob(params, y, X_mean, X_variance, Z)
    log_prior = coefficient_log_prior_sum(
        param_tree,
        (
            (
                "mean_coef",
                inputs.X_mean,
                coefficient_priors.mean if coefficient_priors is not None else None,
            ),
            (
                "log_variance_coef",
                inputs.X_variance,
                coefficient_priors.log_variance if coefficient_priors is not None else None,
            ),
            (
                "gating_coef",
                inputs.Z,
                coefficient_priors.gating if coefficient_priors is not None else None,
            ),
        ),
        coefficient_prior_scale,
        use_ard,
        ard_shape,
        ard_rate,
    )
    return log_likelihood + log_prior


def gaussian_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: GaussianMixtureInputs,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: GaussianMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a diagonal Gaussian variational posterior."""

    def sample_log_joint(param_tree):
        return gaussian_mixture_elbo(
            param_tree,
            inputs,
            coefficient_prior_scale=coefficient_prior_scale,
            use_ard=use_ard,
            ard_shape=ard_shape,
            ard_rate=ard_rate,
            coefficient_priors=coefficient_priors,
        )

    return monte_carlo_variational_elbo(variational_tree, noise_tree, sample_log_joint)



def sample_gaussian_mixture_posterior(
    posterior: GaussianMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted mean-field Gaussian posterior."""

    return sample_posterior_tree(posterior, _gaussian_params_to_tree, seed, n_samples)


def _gaussian_params_to_tree(params: GaussianMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "log_variance_coef": params.log_variance_coef,
        "gating_coef": params.gating_coef,
    }



def fit_gaussian_mixture_vb(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the Gaussian mixture scaffold with JAX gradients and local Adam."""

    fit = fit or FitConfig()
    inputs = prepare_gaussian_mixture_inputs(dataset, setting)
    coefficient_priors = build_gaussian_mixture_priors(inputs, setting)
    return fit_mean_field_mixture_vb(
        fit=fit,
        inputs=inputs,
        initialize_variational_params=lambda: initialize_gaussian_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        variational_elbo=gaussian_mixture_variational_elbo,
        coefficient_priors=coefficient_priors,
        build_result=lambda variational_params, history, converged: _build_variational_result(
            variational_params,
            inputs,
            history,
            converged,
        ),
    )



def _raw_gaussian_mixture_designs(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs_mix],
    )



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


def _build_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: GaussianMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_gaussian_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_variance = jnp.asarray(inputs.X_variance)
    Z = jnp.asarray(inputs.Z)
    resp = gaussian_responsibilities(posterior.mean, y, X_mean, X_variance, Z)
    pred_mean, pred_var = predict_mean_variance(posterior.mean, X_mean, X_variance, Z)
    return VariationalResult(
        params=posterior.mean,
        posterior=posterior,
        elbo_history=history,
        converged=converged,
        responsibilities=np.asarray(resp),
        predictive_mean=np.asarray(pred_mean),
        predictive_variance=np.asarray(pred_var),
    )
