"""Variational inference for BetaReg mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import BetaRegMixtureSetting, FitConfig
from gsm.data import Dataset
from gsm.models.betareg import (
    BetaRegMixtureParams,
    log_prob as betareg_log_prob,
    predict_mean_variance as betareg_predict_mean_variance,
    responsibilities as betareg_responsibilities,
)
from gsm.priors import (
    BetaRegMixtureCoefficientPriors,
    build_betareg_mixture_priors,
    coefficient_log_prior_sum,
)
from gsm.vi.common import (
    VariationalResult,
    fit_design_standardization,
    standardize_designs,
    validate_unit_interval_response,
)
from gsm.vi.engine import (
    initialize_mean_field_variational_params,
    monte_carlo_variational_elbo,
    optimize_restarts,
    sample_posterior_tree,
)


@dataclass(frozen=True)
class BetaRegMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_dispersion: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class BetaRegMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    X_dispersion_c1: np.ndarray
    X_dispersion_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray


@dataclass(frozen=True)
class BetaRegMixturePosterior:
    """Mean-field Gaussian posterior over beta-regression mixture coefficients."""

    mean: BetaRegMixtureParams
    log_std: BetaRegMixtureParams


def prepare_betareg_mixture_inputs(
    dataset: Dataset,
    setting: BetaRegMixtureSetting,
    standardization: BetaRegMixtureStandardization | None = None,
) -> BetaRegMixtureInputs:
    """Build feature-specific design matrices for beta-regression mixtures."""

    validate_unit_interval_response(dataset.y)
    X_mean, X_dispersion, Z = _raw_betareg_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {"X_mean": X_mean, "X_dispersion": X_dispersion, "Z": Z},
        setting.standardize,
        standardization,
    )
    return BetaRegMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=designs["X_mean"],
        X_dispersion=designs["X_dispersion"],
        Z=designs["Z"],
    )


def fit_betareg_mixture_standardization(
    dataset: Dataset,
    setting: BetaRegMixtureSetting,
) -> BetaRegMixtureStandardization:
    """Fit the beta-regression model-specific scaling constants."""

    X_mean, X_dispersion, Z = _raw_betareg_mixture_designs(dataset, setting)
    return BetaRegMixtureStandardization(
        **fit_design_standardization(
            {"X_mean": X_mean, "X_dispersion": X_dispersion, "Z": Z},
            setting.standardize,
        ),
    )


def initialize_betareg_mixture_params(
    inputs: BetaRegMixtureInputs,
    setting: BetaRegMixtureSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the beta-regression VB optimizer."""

    y = np.asarray(inputs.y)
    n_components = setting.n_components
    quantiles = np.linspace(0.15, 0.85, n_components)
    means = np.clip(np.quantile(y, quantiles), 1e-4, 1.0 - 1e-4)
    variance = max(float(np.var(y, ddof=1)), 1e-6)
    global_mean = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    dispersion = max(global_mean * (1.0 - global_mean) / variance - 1.0, 1e-3)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = np.log(means / (1.0 - means))
    dispersion_coef = np.zeros((n_components, inputs.X_dispersion.shape[1]))
    dispersion_coef[:, 0] = np.log(dispersion)
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "dispersion_coef": jnp.asarray(dispersion_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def initialize_betareg_mixture_variational_params(
    inputs: BetaRegMixtureInputs,
    setting: BetaRegMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family."""

    mean_tree = initialize_betareg_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)


def tree_to_betareg_params(tree: dict[str, jnp.ndarray]) -> BetaRegMixtureParams:
    return BetaRegMixtureParams(
        mean_coef=tree["mean_coef"],
        dispersion_coef=tree["dispersion_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_betareg_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> BetaRegMixturePosterior:
    return BetaRegMixturePosterior(
        mean=tree_to_betareg_params(tree["mean"]),
        log_std=tree_to_betareg_params(tree["log_std"]),
    )


def betareg_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: BetaRegMixtureInputs,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: BetaRegMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO for beta-regression mixtures."""

    params = tree_to_betareg_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_dispersion = jnp.asarray(inputs.X_dispersion)
    Z = jnp.asarray(inputs.Z)
    log_likelihood = betareg_log_prob(params, y, X_mean, X_dispersion, Z)
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


def betareg_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: BetaRegMixtureInputs,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: BetaRegMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a beta-regression mean-field Gaussian posterior."""

    def sample_log_joint(param_tree):
        return betareg_mixture_elbo(
            param_tree,
            inputs,
            coefficient_prior_scale=coefficient_prior_scale,
            use_ard=use_ard,
            ard_shape=ard_shape,
            ard_rate=ard_rate,
            coefficient_priors=coefficient_priors,
        )

    return monte_carlo_variational_elbo(variational_tree, noise_tree, sample_log_joint)


def sample_betareg_mixture_posterior(
    posterior: BetaRegMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted beta-regression mean-field posterior."""

    return sample_posterior_tree(posterior, _betareg_params_to_tree, seed, n_samples)


def _betareg_params_to_tree(params: BetaRegMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "dispersion_coef": params.dispersion_coef,
        "gating_coef": params.gating_coef,
    }


def fit_betareg_mixture_vb(
    dataset: Dataset,
    setting: BetaRegMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the beta-regression mixture scaffold with JAX gradients and Adam."""

    fit = fit or FitConfig()
    inputs = prepare_betareg_mixture_inputs(dataset, setting)
    coefficient_priors = build_betareg_mixture_priors(inputs, setting)
    return optimize_restarts(
        fit,
        lambda: initialize_betareg_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        lambda noise_tree: lambda q: betareg_mixture_variational_elbo(
            q,
            inputs,
            noise_tree,
            coefficient_prior_scale=fit.coefficient_prior_scale,
            use_ard=fit.use_ard,
            ard_shape=fit.ard_shape,
            ard_rate=fit.ard_rate,
            coefficient_priors=coefficient_priors,
        ),
        lambda variational_params, history, converged: _build_betareg_variational_result(
            variational_params, inputs, history, converged
        ),
    )


def _raw_betareg_mixture_designs(
    dataset: Dataset,
    setting: BetaRegMixtureSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs_mix],
    )


def _build_betareg_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: BetaRegMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_betareg_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_dispersion = jnp.asarray(inputs.X_dispersion)
    Z = jnp.asarray(inputs.Z)
    resp = betareg_responsibilities(posterior.mean, y, X_mean, X_dispersion, Z)
    pred_mean, pred_var = betareg_predict_mean_variance(
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
