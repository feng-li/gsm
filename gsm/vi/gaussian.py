"""Variational inference for Gaussian mixtures."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gsm.config import FitConfig, GaussianMixtureSetting
from gsm.data import Dataset, standardize_covariates
from gsm.models.gaussian import (
    GaussianMixtureParams,
    log_prob as gaussian_log_prob,
    predict_mean_variance,
    responsibilities as gaussian_responsibilities,
)
from gsm.priors import (
    GaussianMixtureCoefficientPriors,
    build_gaussian_mixture_priors,
    coefficient_log_prior,
)
from gsm.vi.common import VariationalResult, apply_standardization
from gsm.vi.engine import (
    jitter_params,
    mean_field_gaussian_entropy,
    optax_maximize,
    sample_noise_like,
    sample_param_trees,
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

    if setting.standardize:
        if standardization is None:
            X_mean, _, _ = standardize_covariates(X_mean, setting.standardize)
            X_variance, _, _ = standardize_covariates(X_variance, setting.standardize)
            Z, _, _ = standardize_covariates(Z, setting.standardize)
        else:
            X_mean = apply_standardization(
                X_mean,
                setting.standardize,
                standardization.X_mean_c1,
                standardization.X_mean_c2,
            )
            X_variance = apply_standardization(
                X_variance,
                setting.standardize,
                standardization.X_variance_c1,
                standardization.X_variance_c2,
            )
            Z = apply_standardization(
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


def initialize_gaussian_mixture_variational_params(
    inputs: GaussianMixtureInputs,
    setting: GaussianMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family."""

    mean_tree = initialize_gaussian_mixture_params(inputs, setting)
    log_std_tree = jax.tree_util.tree_map(
        lambda value: jnp.full_like(value, init_log_std),
        mean_tree,
    )
    return {"mean": mean_tree, "log_std": log_std_tree}



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
    log_prior = (
        coefficient_log_prior(
            param_tree["mean_coef"],
            inputs.X_mean,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
            coefficient_priors.mean if coefficient_priors is not None else None,
        )
        + coefficient_log_prior(
            param_tree["log_variance_coef"],
            inputs.X_variance,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
            coefficient_priors.log_variance if coefficient_priors is not None else None,
        )
        + coefficient_log_prior(
            param_tree["gating_coef"],
            inputs.Z,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
            coefficient_priors.gating if coefficient_priors is not None else None,
        )
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

    sample_tree = sample_param_trees(
        variational_tree["mean"],
        variational_tree["log_std"],
        noise_tree,
    )

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

    log_joint = jax.vmap(sample_log_joint)(sample_tree)
    return jnp.mean(log_joint) + mean_field_gaussian_entropy(variational_tree["log_std"])



def sample_gaussian_mixture_posterior(
    posterior: GaussianMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted mean-field Gaussian posterior."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    mean_tree = _gaussian_params_to_tree(posterior.mean)
    log_std_tree = _gaussian_params_to_tree(posterior.log_std)
    noise_tree = sample_noise_like(mean_tree, jax.random.PRNGKey(seed), n_samples)
    return sample_param_trees(mean_tree, log_std_tree, noise_tree)


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
    if fit.n_elbo_samples < 1:
        raise ValueError("n_elbo_samples must be positive")

    inputs = prepare_gaussian_mixture_inputs(dataset, setting)
    coefficient_priors = build_gaussian_mixture_priors(inputs, setting)
    best_result: VariationalResult | None = None

    for restart in range(fit.n_restarts):
        variational_params = initialize_gaussian_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        )
        if restart:
            variational_params["mean"] = jitter_params(
                variational_params["mean"],
                fit.seed + restart,
            )

        noise_tree = sample_noise_like(
            variational_params["mean"],
            jax.random.PRNGKey(fit.seed + 1009 * (restart + 1)),
            fit.n_elbo_samples,
        )
        variational_params, history, converged = optax_maximize(
            variational_params,
            lambda q: gaussian_mixture_variational_elbo(
                q,
                inputs,
                noise_tree,
                coefficient_prior_scale=fit.coefficient_prior_scale,
                use_ard=fit.use_ard,
                ard_shape=fit.ard_shape,
                ard_rate=fit.ard_rate,
                coefficient_priors=coefficient_priors,
            ),
            max_iter=fit.max_iter,
            learning_rate=fit.learning_rate,
            tol=fit.tol,
        )
        result = _build_variational_result(variational_params, inputs, history, converged)
        if best_result is None or result.elbo_history[-1] > best_result.elbo_history[-1]:
            best_result = result

    if best_result is None:
        raise RuntimeError("no variational optimization runs were executed")
    return best_result



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
