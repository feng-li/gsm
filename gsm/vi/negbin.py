"""Variational inference for negative-binomial mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import FitConfig, NegBinMixtureSetting
from gsm.data import Dataset
from gsm.models.negbin import (
    NegBinMixtureParams,
    log_prob as negbin_log_prob,
    predict_mean_variance as negbin_predict_mean_variance,
    responsibilities as negbin_responsibilities,
)
from gsm.priors import (
    NegBinMixtureCoefficientPriors,
    build_negbin_mixture_priors,
    coefficient_log_prior_sum,
)
from gsm.vi.common import (
    VariationalResult,
    fit_design_standardization,
    standardize_designs,
    validate_count_response,
)
from gsm.vi.engine import (
    initialize_mean_field_variational_params,
    monte_carlo_variational_elbo,
    optimize_restarts,
    sample_posterior_tree,
)


@dataclass(frozen=True)
class NegBinMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_dispersion: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class NegBinMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    X_dispersion_c1: np.ndarray
    X_dispersion_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray


@dataclass(frozen=True)
class NegBinMixturePosterior:
    """Mean-field Gaussian posterior over negative-binomial coefficients."""

    mean: NegBinMixtureParams
    log_std: NegBinMixtureParams


def prepare_negbin_mixture_inputs(
    dataset: Dataset,
    setting: NegBinMixtureSetting,
    standardization: NegBinMixtureStandardization | None = None,
) -> NegBinMixtureInputs:
    """Build feature-specific design matrices for negative-binomial mixtures."""

    validate_count_response(dataset.y)
    X_mean, X_dispersion, Z = _raw_negbin_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {"X_mean": X_mean, "X_dispersion": X_dispersion, "Z": Z},
        setting.standardize,
        standardization,
    )
    return NegBinMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=designs["X_mean"],
        X_dispersion=designs["X_dispersion"],
        Z=designs["Z"],
    )


def fit_negbin_mixture_standardization(
    dataset: Dataset,
    setting: NegBinMixtureSetting,
) -> NegBinMixtureStandardization:
    """Fit the negative-binomial model-specific scaling constants."""

    X_mean, X_dispersion, Z = _raw_negbin_mixture_designs(dataset, setting)
    return NegBinMixtureStandardization(
        **fit_design_standardization(
            {"X_mean": X_mean, "X_dispersion": X_dispersion, "Z": Z},
            setting.standardize,
        ),
    )


def initialize_negbin_mixture_params(
    inputs: NegBinMixtureInputs,
    setting: NegBinMixtureSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the negative-binomial VB optimizer."""

    y = np.asarray(inputs.y)
    n_components = setting.n_components
    quantiles = np.linspace(0.15, 0.85, n_components)
    means = np.maximum(np.quantile(y, quantiles), 1e-3)
    variance = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    global_mean = max(float(np.mean(y)), 1e-3)
    overdispersion = max(variance - global_mean, 1e-3)
    dispersion = np.clip(global_mean**2 / overdispersion, 1e-3, 1e4)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = np.log(means)
    dispersion_coef = np.zeros((n_components, inputs.X_dispersion.shape[1]))
    dispersion_coef[:, 0] = np.log(dispersion)
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "dispersion_coef": jnp.asarray(dispersion_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def initialize_negbin_mixture_variational_params(
    inputs: NegBinMixtureInputs,
    setting: NegBinMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family."""

    mean_tree = initialize_negbin_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)


def tree_to_negbin_params(tree: dict[str, jnp.ndarray]) -> NegBinMixtureParams:
    return NegBinMixtureParams(
        mean_coef=tree["mean_coef"],
        dispersion_coef=tree["dispersion_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_negbin_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> NegBinMixturePosterior:
    return NegBinMixturePosterior(
        mean=tree_to_negbin_params(tree["mean"]),
        log_std=tree_to_negbin_params(tree["log_std"]),
    )


def negbin_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: NegBinMixtureInputs,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: NegBinMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO for negative-binomial mixtures."""

    params = tree_to_negbin_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_dispersion = jnp.asarray(inputs.X_dispersion)
    Z = jnp.asarray(inputs.Z)
    log_likelihood = negbin_log_prob(params, y, X_mean, X_dispersion, Z)
    log_prior = coefficient_log_prior_sum(
        param_tree,
        (
            (
                "mean_coef",
                inputs.X_mean,
                coefficient_priors.mean if coefficient_priors is not None else None,
            ),
            (
                "dispersion_coef",
                inputs.X_dispersion,
                coefficient_priors.dispersion if coefficient_priors is not None else None,
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


def negbin_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: NegBinMixtureInputs,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: NegBinMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a negative-binomial mean-field Gaussian posterior."""

    def sample_log_joint(param_tree):
        return negbin_mixture_elbo(
            param_tree,
            inputs,
            coefficient_prior_scale=coefficient_prior_scale,
            use_ard=use_ard,
            ard_shape=ard_shape,
            ard_rate=ard_rate,
            coefficient_priors=coefficient_priors,
        )

    return monte_carlo_variational_elbo(variational_tree, noise_tree, sample_log_joint)


def sample_negbin_mixture_posterior(
    posterior: NegBinMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted negative-binomial posterior."""

    return sample_posterior_tree(posterior, _negbin_params_to_tree, seed, n_samples)


def _negbin_params_to_tree(params: NegBinMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "dispersion_coef": params.dispersion_coef,
        "gating_coef": params.gating_coef,
    }


def fit_negbin_mixture_vb(
    dataset: Dataset,
    setting: NegBinMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the negative-binomial mixture scaffold with JAX gradients and Adam."""

    fit = fit or FitConfig()
    inputs = prepare_negbin_mixture_inputs(dataset, setting)
    coefficient_priors = build_negbin_mixture_priors(inputs, setting)
    return optimize_restarts(
        fit,
        lambda: initialize_negbin_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        lambda noise_tree: lambda q: negbin_mixture_variational_elbo(
            q,
            inputs,
            noise_tree,
            coefficient_prior_scale=fit.coefficient_prior_scale,
            use_ard=fit.use_ard,
            ard_shape=fit.ard_shape,
            ard_rate=fit.ard_rate,
            coefficient_priors=coefficient_priors,
        ),
        lambda variational_params, history, converged: _build_negbin_variational_result(
            variational_params, inputs, history, converged
        ),
    )


def _raw_negbin_mixture_designs(
    dataset: Dataset,
    setting: NegBinMixtureSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs_mix],
    )


def _build_negbin_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: NegBinMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_negbin_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_dispersion = jnp.asarray(inputs.X_dispersion)
    Z = jnp.asarray(inputs.Z)
    resp = negbin_responsibilities(posterior.mean, y, X_mean, X_dispersion, Z)
    pred_mean, pred_var = negbin_predict_mean_variance(
        posterior.mean, X_mean, X_dispersion, Z
    )
    return VariationalResult(
        params=posterior.mean,
        posterior=posterior,
        elbo_history=history,
        converged=converged,
        responsibilities=np.asarray(resp),
        predictive_mean=np.asarray(pred_mean),
        predictive_variance=np.asarray(pred_var),
    )
