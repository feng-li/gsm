"""Variational inference for Poisson mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import FitConfig, PoissonMixtureSetting
from gsm.data import Dataset
from gsm.models.poisson import (
    PoissonMixtureParams,
    log_prob as poisson_log_prob,
    predict_mean_variance as poisson_predict_mean_variance,
    responsibilities as poisson_responsibilities,
)
from gsm.priors import (
    PoissonMixtureCoefficientPriors,
    build_poisson_mixture_priors,
    coefficient_log_prior_sum,
)
from gsm.vi.common import (
    VariationalResult,
    fit_design_standardization,
    standardize_designs,
    validate_count_response,
)
from gsm.vi.engine import (
    fit_mean_field_mixture_vb,
    initialize_mean_field_variational_params,
    monte_carlo_variational_elbo,
    sample_posterior_tree,
)


@dataclass(frozen=True)
class PoissonMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class PoissonMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray


@dataclass(frozen=True)
class PoissonMixturePosterior:
    """Mean-field Gaussian posterior over Poisson-mixture coefficients."""

    mean: PoissonMixtureParams
    log_std: PoissonMixtureParams


def prepare_poisson_mixture_inputs(
    dataset: Dataset,
    setting: PoissonMixtureSetting,
    standardization: PoissonMixtureStandardization | None = None,
) -> PoissonMixtureInputs:
    """Build feature-specific design matrices for Poisson mixtures."""

    validate_count_response(dataset.y)
    X_mean, Z = _raw_poisson_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {"X_mean": X_mean, "Z": Z},
        setting.standardize,
        standardization,
    )
    return PoissonMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=designs["X_mean"],
        Z=designs["Z"],
    )


def fit_poisson_mixture_standardization(
    dataset: Dataset,
    setting: PoissonMixtureSetting,
) -> PoissonMixtureStandardization:
    """Fit the Poisson model-specific scaling constants."""

    X_mean, Z = _raw_poisson_mixture_designs(dataset, setting)
    return PoissonMixtureStandardization(
        **fit_design_standardization(
            {"X_mean": X_mean, "Z": Z},
            setting.standardize,
        ),
    )


def initialize_poisson_mixture_params(
    inputs: PoissonMixtureInputs,
    setting: PoissonMixtureSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the Poisson VB optimizer."""

    y = np.asarray(inputs.y)
    n_components = setting.n_components
    if n_components == 1:
        means = np.asarray([np.mean(y)])
    else:
        quantiles = np.linspace(0.15, 0.85, n_components)
        means = np.quantile(y, quantiles)
    means = np.maximum(means, 1e-3)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = np.log(means)
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def initialize_poisson_mixture_variational_params(
    inputs: PoissonMixtureInputs,
    setting: PoissonMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family."""

    mean_tree = initialize_poisson_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)


def tree_to_poisson_params(tree: dict[str, jnp.ndarray]) -> PoissonMixtureParams:
    return PoissonMixtureParams(
        mean_coef=tree["mean_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_poisson_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> PoissonMixturePosterior:
    return PoissonMixturePosterior(
        mean=tree_to_poisson_params(tree["mean"]),
        log_std=tree_to_poisson_params(tree["log_std"]),
    )


def poisson_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: PoissonMixtureInputs,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: PoissonMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO for Poisson mixtures."""

    params = tree_to_poisson_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    Z = jnp.asarray(inputs.Z)
    log_likelihood = poisson_log_prob(params, y, X_mean, Z)
    log_prior = coefficient_log_prior_sum(
        param_tree,
        (
            (
                "mean_coef",
                inputs.X_mean,
                coefficient_priors.mean if coefficient_priors is not None else None,
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


def poisson_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: PoissonMixtureInputs,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: PoissonMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a Poisson mean-field Gaussian posterior."""

    def sample_log_joint(param_tree):
        return poisson_mixture_elbo(
            param_tree,
            inputs,
            coefficient_prior_scale=coefficient_prior_scale,
            use_ard=use_ard,
            ard_shape=ard_shape,
            ard_rate=ard_rate,
            coefficient_priors=coefficient_priors,
        )

    return monte_carlo_variational_elbo(variational_tree, noise_tree, sample_log_joint)


def sample_poisson_mixture_posterior(
    posterior: PoissonMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted Poisson mean-field posterior."""

    return sample_posterior_tree(posterior, _poisson_params_to_tree, seed, n_samples)


def _poisson_params_to_tree(params: PoissonMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "gating_coef": params.gating_coef,
    }


def fit_poisson_mixture_vb(
    dataset: Dataset,
    setting: PoissonMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the Poisson mixture scaffold with JAX gradients and Adam."""

    fit = fit or FitConfig()
    inputs = prepare_poisson_mixture_inputs(dataset, setting)
    coefficient_priors = build_poisson_mixture_priors(inputs, setting)
    return fit_mean_field_mixture_vb(
        fit=fit,
        inputs=inputs,
        initialize_variational_params=lambda: initialize_poisson_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        variational_elbo=poisson_mixture_variational_elbo,
        coefficient_priors=coefficient_priors,
        build_result=lambda variational_params, history, converged: _build_poisson_variational_result(
            variational_params, inputs, history, converged
        ),
    )


def _raw_poisson_mixture_designs(
    dataset: Dataset,
    setting: PoissonMixtureSetting,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs_mix],
    )


def _build_poisson_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: PoissonMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_poisson_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    Z = jnp.asarray(inputs.Z)
    resp = poisson_responsibilities(posterior.mean, y, X_mean, Z)
    pred_mean, pred_var = poisson_predict_mean_variance(posterior.mean, X_mean, Z)
    return VariationalResult(
        params=posterior.mean,
        posterior=posterior,
        elbo_history=history,
        converged=converged,
        responsibilities=np.asarray(resp),
        predictive_mean=np.asarray(pred_mean),
        predictive_variance=np.asarray(pred_var),
    )
