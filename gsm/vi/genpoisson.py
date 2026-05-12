"""Variational inference for GenPois and GenPoisAlt mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import FitConfig, GenPoissonAltMixtureSetting, GenPoissonMixtureSetting
from gsm.data import Dataset
from gsm.models.genpoisson import (
    GenPoissonMixtureParams,
    log_prob as genpoisson_log_prob,
    predict_mean_variance as genpoisson_predict_mean_variance,
    responsibilities as genpoisson_responsibilities,
)
from gsm.priors import (
    GenPoissonMixtureCoefficientPriors,
    build_genpoisson_mixture_priors,
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


GenPoissonSetting = GenPoissonMixtureSetting | GenPoissonAltMixtureSetting


@dataclass(frozen=True)
class GenPoissonMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_dispersion: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class GenPoissonMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    X_dispersion_c1: np.ndarray
    X_dispersion_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray


@dataclass(frozen=True)
class GenPoissonMixturePosterior:
    """Mean-field Gaussian posterior over generalized Poisson coefficients."""

    mean: GenPoissonMixtureParams
    log_std: GenPoissonMixtureParams


def prepare_genpoisson_mixture_inputs(
    dataset: Dataset,
    setting: GenPoissonSetting,
    standardization: GenPoissonMixtureStandardization | None = None,
) -> GenPoissonMixtureInputs:
    """Build feature-specific design matrices for generalized Poisson mixtures."""

    validate_count_response(dataset.y)
    X_mean, X_dispersion, Z = _raw_genpoisson_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {"X_mean": X_mean, "X_dispersion": X_dispersion, "Z": Z},
        setting.standardize,
        standardization,
    )
    return GenPoissonMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=designs["X_mean"],
        X_dispersion=designs["X_dispersion"],
        Z=designs["Z"],
    )


def fit_genpoisson_mixture_standardization(
    dataset: Dataset,
    setting: GenPoissonSetting,
) -> GenPoissonMixtureStandardization:
    """Fit the GenPois model-specific scaling constants."""

    X_mean, X_dispersion, Z = _raw_genpoisson_mixture_designs(dataset, setting)
    return GenPoissonMixtureStandardization(
        **fit_design_standardization(
            {"X_mean": X_mean, "X_dispersion": X_dispersion, "Z": Z},
            setting.standardize,
        ),
    )


def initialize_genpoisson_mixture_params(
    inputs: GenPoissonMixtureInputs,
    setting: GenPoissonSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the GenPois VB optimizer."""

    y = np.asarray(inputs.y)
    n_components = setting.n_components
    if n_components == 1:
        means = np.asarray([np.mean(y)])
    else:
        quantiles = np.linspace(0.15, 0.85, n_components)
        means = np.quantile(y, quantiles)
    means = np.maximum(means, 1e-3)
    dispersion = _initial_dispersion(y, setting)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = np.log(means)
    dispersion_coef = np.zeros((n_components, inputs.X_dispersion.shape[1]))
    if setting.parameterization == "alternative":
        dispersion_coef[:, 0] = np.log(max(dispersion - 1.0, 1e-3))
    else:
        dispersion_coef[:, 0] = np.log(max(dispersion, 1e-3))
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "dispersion_coef": jnp.asarray(dispersion_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def initialize_genpoisson_mixture_variational_params(
    inputs: GenPoissonMixtureInputs,
    setting: GenPoissonSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family."""

    mean_tree = initialize_genpoisson_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)


def tree_to_genpoisson_params(tree: dict[str, jnp.ndarray]) -> GenPoissonMixtureParams:
    return GenPoissonMixtureParams(
        mean_coef=tree["mean_coef"],
        dispersion_coef=tree["dispersion_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_genpoisson_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> GenPoissonMixturePosterior:
    return GenPoissonMixturePosterior(
        mean=tree_to_genpoisson_params(tree["mean"]),
        log_std=tree_to_genpoisson_params(tree["log_std"]),
    )


def genpoisson_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: GenPoissonMixtureInputs,
    setting: GenPoissonSetting,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: GenPoissonMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO for generalized Poisson mixtures."""

    params = tree_to_genpoisson_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_dispersion = jnp.asarray(inputs.X_dispersion)
    Z = jnp.asarray(inputs.Z)
    log_likelihood = genpoisson_log_prob(
        params,
        y,
        X_mean,
        X_dispersion,
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


def genpoisson_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: GenPoissonMixtureInputs,
    setting: GenPoissonSetting,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: GenPoissonMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a generalized Poisson mean-field posterior."""

    def sample_log_joint(param_tree):
        return genpoisson_mixture_elbo(
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


def sample_genpoisson_mixture_posterior(
    posterior: GenPoissonMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted generalized Poisson posterior."""

    return sample_posterior_tree(posterior, _genpoisson_params_to_tree, seed, n_samples)


def _genpoisson_params_to_tree(
    params: GenPoissonMixtureParams,
) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "dispersion_coef": params.dispersion_coef,
        "gating_coef": params.gating_coef,
    }


def fit_genpoisson_mixture_vb(
    dataset: Dataset,
    setting: GenPoissonSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the generalized Poisson mixture scaffold with JAX gradients and Adam."""

    fit = fit or FitConfig()
    inputs = prepare_genpoisson_mixture_inputs(dataset, setting)
    coefficient_priors = build_genpoisson_mixture_priors(inputs, setting)
    return optimize_restarts(
        fit,
        lambda: initialize_genpoisson_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        lambda noise_tree: lambda q: genpoisson_mixture_variational_elbo(
            q,
            inputs,
            setting,
            noise_tree,
            coefficient_prior_scale=fit.coefficient_prior_scale,
            use_ard=fit.use_ard,
            ard_shape=fit.ard_shape,
            ard_rate=fit.ard_rate,
            coefficient_priors=coefficient_priors,
        ),
        lambda variational_params, history, converged: _build_genpoisson_variational_result(
            variational_params, inputs, setting, history, converged
        ),
    )


def _raw_genpoisson_mixture_designs(
    dataset: Dataset,
    setting: GenPoissonSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs_mix],
    )


def _build_genpoisson_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: GenPoissonMixtureInputs,
    setting: GenPoissonSetting,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_genpoisson_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_dispersion = jnp.asarray(inputs.X_dispersion)
    Z = jnp.asarray(inputs.Z)
    resp = genpoisson_responsibilities(
        posterior.mean,
        y,
        X_mean,
        X_dispersion,
        Z,
        parameterization=setting.parameterization,
    )
    pred_mean, pred_var = genpoisson_predict_mean_variance(
        posterior.mean,
        X_mean,
        X_dispersion,
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


def _initial_dispersion(y: np.ndarray, setting: GenPoissonSetting) -> float:
    y = np.asarray(y, dtype=float)
    mean = max(float(np.mean(y)), 1e-3)
    variance = float(np.var(y, ddof=1)) if y.size > 1 else mean
    if setting.parameterization == "alternative":
        return max(np.sqrt(max(variance, 1e-3) / mean), 1.001)
    if variance <= mean:
        return 1e-3
    return max((np.sqrt(variance / mean) - 1.0) / mean, 1e-3)
