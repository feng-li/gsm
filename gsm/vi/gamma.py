"""Variational inference for Gamma and GammaRep mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import FitConfig, GammaMixtureSetting, GammaRepMixtureSetting
from gsm.data import Dataset
from gsm.models.gamma import (
    GammaMixtureParams,
    log_prob as gamma_log_prob,
    predict_mean_variance as gamma_predict_mean_variance,
    responsibilities as gamma_responsibilities,
)
from gsm.priors import (
    GammaMixtureCoefficientPriors,
    build_gamma_mixture_priors,
    coefficient_log_prior_sum,
)
from gsm.vi.common import (
    VariationalResult,
    fit_design_standardization,
    standardize_designs,
    validate_positive_response,
)
from gsm.vi.engine import (
    fit_mean_field_mixture_vb,
    initialize_mean_field_variational_params,
    monte_carlo_variational_elbo,
    sample_posterior_tree,
)


GammaSetting = GammaMixtureSetting | GammaRepMixtureSetting


@dataclass(frozen=True)
class GammaMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_variance: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class GammaMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    X_variance_c1: np.ndarray
    X_variance_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray


@dataclass(frozen=True)
class GammaMixturePosterior:
    """Mean-field Gaussian posterior over gamma-mixture coefficients."""

    mean: GammaMixtureParams
    log_std: GammaMixtureParams


def prepare_gamma_mixture_inputs(
    dataset: Dataset,
    setting: GammaSetting,
    standardization: GammaMixtureStandardization | None = None,
) -> GammaMixtureInputs:
    """Build feature-specific design matrices for gamma mixtures."""

    validate_positive_response(dataset.y)
    X_mean, X_variance, Z = _raw_gamma_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {"X_mean": X_mean, "X_variance": X_variance, "Z": Z},
        setting.standardize,
        standardization,
    )

    return GammaMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=designs["X_mean"],
        X_variance=designs["X_variance"],
        Z=designs["Z"],
    )


def fit_gamma_mixture_standardization(
    dataset: Dataset,
    setting: GammaSetting,
) -> GammaMixtureStandardization:
    """Fit the gamma model-specific scaling constants."""

    X_mean, X_variance, Z = _raw_gamma_mixture_designs(dataset, setting)
    return GammaMixtureStandardization(
        **fit_design_standardization(
            {"X_mean": X_mean, "X_variance": X_variance, "Z": Z},
            setting.standardize,
        ),
    )


def initialize_gamma_mixture_params(
    inputs: GammaMixtureInputs,
    setting: GammaSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the gamma VB optimizer."""

    y = np.asarray(inputs.y)
    n_components = setting.n_components
    quantiles = np.linspace(0.15, 0.85, n_components)
    component_means = np.maximum(np.quantile(y, quantiles), 1e-6)
    variance = max(float(np.var(y, ddof=1)), 1e-6)

    if setting.parameterization == "shape_scale":
        feature1 = np.maximum(component_means**2 / variance, 1e-6)
        feature2 = np.maximum(variance / component_means, 1e-6)
    else:
        feature1 = component_means
        feature2 = np.full(n_components, variance)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = np.log(feature1)
    variance_coef = np.zeros((n_components, inputs.X_variance.shape[1]))
    variance_coef[:, 0] = np.log(feature2)
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "variance_coef": jnp.asarray(variance_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def initialize_gamma_mixture_variational_params(
    inputs: GammaMixtureInputs,
    setting: GammaSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family for gamma mixtures."""

    mean_tree = initialize_gamma_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)


def tree_to_gamma_params(tree: dict[str, jnp.ndarray]) -> GammaMixtureParams:
    return GammaMixtureParams(
        mean_coef=tree["mean_coef"],
        variance_coef=tree["variance_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_gamma_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> GammaMixturePosterior:
    return GammaMixturePosterior(
        mean=tree_to_gamma_params(tree["mean"]),
        log_std=tree_to_gamma_params(tree["log_std"]),
    )


def gamma_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: GammaMixtureInputs,
    setting: GammaSetting,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: GammaMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO for gamma mixtures."""

    params = tree_to_gamma_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_variance = jnp.asarray(inputs.X_variance)
    Z = jnp.asarray(inputs.Z)

    log_likelihood = gamma_log_prob(
        params,
        y,
        X_mean,
        X_variance,
        Z,
        parameterization=setting.parameterization,
    )
    log_prior = coefficient_log_prior_sum(
        param_tree,
        (
            (
                "mean_coef",
                inputs.X_mean,
                coefficient_priors.mean if coefficient_priors is not None else None,
            ),
            (
                "variance_coef",
                inputs.X_variance,
                coefficient_priors.variance if coefficient_priors is not None else None,
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


def gamma_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: GammaMixtureInputs,
    setting: GammaSetting,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: GammaMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a gamma mean-field Gaussian posterior."""

    def sample_log_joint(param_tree):
        return gamma_mixture_elbo(
            param_tree,
            inputs,
            setting,
            coefficient_prior_scale=coefficient_prior_scale,
            use_ard=use_ard,
            ard_shape=ard_shape,
            ard_rate=ard_rate,
            coefficient_priors=coefficient_priors,
        )

    return monte_carlo_variational_elbo(variational_tree, noise_tree, sample_log_joint)


def sample_gamma_mixture_posterior(
    posterior: GammaMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted gamma mean-field posterior."""

    return sample_posterior_tree(posterior, _gamma_params_to_tree, seed, n_samples)


def _gamma_params_to_tree(params: GammaMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "variance_coef": params.variance_coef,
        "gating_coef": params.gating_coef,
    }


def fit_gamma_mixture_vb(
    dataset: Dataset,
    setting: GammaSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the gamma mixture scaffold with JAX gradients and local Adam."""

    fit = fit or FitConfig()
    inputs = prepare_gamma_mixture_inputs(dataset, setting)
    coefficient_priors = build_gamma_mixture_priors(inputs, setting)
    return fit_mean_field_mixture_vb(
        fit=fit,
        inputs=inputs,
        initialize_variational_params=lambda: initialize_gamma_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        variational_elbo=gamma_mixture_variational_elbo,
        coefficient_priors=coefficient_priors,
        build_result=lambda variational_params, history, converged: _build_gamma_variational_result(
            variational_params, inputs, setting, history, converged
        ),
        extra_args=(setting,),
    )


def _raw_gamma_mixture_designs(
    dataset: Dataset,
    setting: GammaSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs_mix],
    )


def _build_gamma_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: GammaMixtureInputs,
    setting: GammaSetting,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_gamma_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_variance = jnp.asarray(inputs.X_variance)
    Z = jnp.asarray(inputs.Z)
    resp = gamma_responsibilities(
        posterior.mean,
        y,
        X_mean,
        X_variance,
        Z,
        parameterization=setting.parameterization,
    )
    pred_mean, pred_var = gamma_predict_mean_variance(
        posterior.mean,
        X_mean,
        X_variance,
        Z,
        parameterization=setting.parameterization,
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
