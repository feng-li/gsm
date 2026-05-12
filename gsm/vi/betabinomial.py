"""Variational inference for beta-binomial mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import BetaBinMixtureSetting, FitConfig
from gsm.data import Dataset
from gsm.models.betabinomial import (
    BetaBinMixtureParams,
    log_prob as betabin_log_prob,
    predict_mean_variance as betabin_predict_mean_variance,
    responsibilities as betabin_responsibilities,
)
from gsm.priors import (
    BetaBinMixtureCoefficientPriors,
    build_betabin_mixture_priors,
    coefficient_log_prior_sum,
)
from gsm.vi.common import (
    VariationalResult,
    fit_design_standardization,
    standardize_designs,
    validate_binomial_response,
)
from gsm.vi.engine import (
    initialize_mean_field_variational_params,
    monte_carlo_variational_elbo,
    optimize_restarts,
    sample_posterior_tree,
)


@dataclass(frozen=True)
class BetaBinMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_dispersion: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class BetaBinMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    X_dispersion_c1: np.ndarray
    X_dispersion_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray


@dataclass(frozen=True)
class BetaBinMixturePosterior:
    """Mean-field Gaussian posterior over beta-binomial coefficients."""

    mean: BetaBinMixtureParams
    log_std: BetaBinMixtureParams


def prepare_betabin_mixture_inputs(
    dataset: Dataset,
    setting: BetaBinMixtureSetting,
    standardization: BetaBinMixtureStandardization | None = None,
) -> BetaBinMixtureInputs:
    """Build feature-specific design matrices for beta-binomial mixtures."""

    validate_binomial_response(dataset.y)
    X_mean, X_dispersion, Z = _raw_betabin_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {"X_mean": X_mean, "X_dispersion": X_dispersion, "Z": Z},
        setting.standardize,
        standardization,
    )
    return BetaBinMixtureInputs(
        y=np.asarray(dataset.y, dtype=float),
        X_mean=designs["X_mean"],
        X_dispersion=designs["X_dispersion"],
        Z=designs["Z"],
    )


def fit_betabin_mixture_standardization(
    dataset: Dataset,
    setting: BetaBinMixtureSetting,
) -> BetaBinMixtureStandardization:
    """Fit the beta-binomial model-specific scaling constants."""

    X_mean, X_dispersion, Z = _raw_betabin_mixture_designs(dataset, setting)
    return BetaBinMixtureStandardization(
        **fit_design_standardization(
            {"X_mean": X_mean, "X_dispersion": X_dispersion, "Z": Z},
            setting.standardize,
        ),
    )


def initialize_betabin_mixture_params(
    inputs: BetaBinMixtureInputs,
    setting: BetaBinMixtureSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the beta-binomial VB optimizer."""

    rates = _success_rates(inputs.y)
    n_components = setting.n_components
    if n_components == 1:
        means = np.asarray([_pooled_rate(inputs.y)])
    else:
        quantiles = np.linspace(0.15, 0.85, n_components)
        means = np.quantile(rates, quantiles)
    means = np.clip(means, 1e-4, 1.0 - 1e-4)
    dispersion = max(float(setting.prior_mean_feat[1]), 1e-3)

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


def initialize_betabin_mixture_variational_params(
    inputs: BetaBinMixtureInputs,
    setting: BetaBinMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family."""

    mean_tree = initialize_betabin_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)


def tree_to_betabin_params(tree: dict[str, jnp.ndarray]) -> BetaBinMixtureParams:
    return BetaBinMixtureParams(
        mean_coef=tree["mean_coef"],
        dispersion_coef=tree["dispersion_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_betabin_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> BetaBinMixturePosterior:
    return BetaBinMixturePosterior(
        mean=tree_to_betabin_params(tree["mean"]),
        log_std=tree_to_betabin_params(tree["log_std"]),
    )


def betabin_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: BetaBinMixtureInputs,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: BetaBinMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO for beta-binomial mixtures."""

    params = tree_to_betabin_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_dispersion = jnp.asarray(inputs.X_dispersion)
    Z = jnp.asarray(inputs.Z)
    log_likelihood = betabin_log_prob(params, y, X_mean, X_dispersion, Z)
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


def betabin_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: BetaBinMixtureInputs,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: BetaBinMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a beta-binomial mean-field Gaussian posterior."""

    def sample_log_joint(param_tree):
        return betabin_mixture_elbo(
            param_tree,
            inputs,
            coefficient_prior_scale=coefficient_prior_scale,
            use_ard=use_ard,
            ard_shape=ard_shape,
            ard_rate=ard_rate,
            coefficient_priors=coefficient_priors,
        )

    return monte_carlo_variational_elbo(variational_tree, noise_tree, sample_log_joint)


def sample_betabin_mixture_posterior(
    posterior: BetaBinMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted beta-binomial posterior."""

    return sample_posterior_tree(posterior, _betabin_params_to_tree, seed, n_samples)


def _betabin_params_to_tree(params: BetaBinMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "dispersion_coef": params.dispersion_coef,
        "gating_coef": params.gating_coef,
    }


def fit_betabin_mixture_vb(
    dataset: Dataset,
    setting: BetaBinMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the beta-binomial mixture scaffold with JAX gradients and Adam."""

    fit = fit or FitConfig()
    inputs = prepare_betabin_mixture_inputs(dataset, setting)
    coefficient_priors = build_betabin_mixture_priors(inputs, setting)
    return optimize_restarts(
        fit,
        lambda: initialize_betabin_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        lambda noise_tree: lambda q: betabin_mixture_variational_elbo(
            q,
            inputs,
            noise_tree,
            coefficient_prior_scale=fit.coefficient_prior_scale,
            use_ard=fit.use_ard,
            ard_shape=fit.ard_shape,
            ard_rate=fit.ard_rate,
            coefficient_priors=coefficient_priors,
        ),
        lambda variational_params, history, converged: _build_betabin_variational_result(
            variational_params, inputs, history, converged
        ),
    )


def _raw_betabin_mixture_designs(
    dataset: Dataset,
    setting: BetaBinMixtureSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs_mix],
    )


def _build_betabin_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: BetaBinMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_betabin_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_dispersion = jnp.asarray(inputs.X_dispersion)
    Z = jnp.asarray(inputs.Z)
    resp = betabin_responsibilities(posterior.mean, y, X_mean, X_dispersion, Z)
    pred_mean, pred_var = betabin_predict_mean_variance(
        posterior.mean,
        X_mean,
        X_dispersion,
        Z,
        trials=y,
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


def _success_rates(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    successes = y[:, 0]
    trials = y[:, 1]
    mask = trials > 0.0
    if not np.any(mask):
        return np.asarray([0.5])
    return np.clip(successes[mask] / trials[mask], 1e-4, 1.0 - 1e-4)


def _pooled_rate(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    total_trials = float(np.sum(y[:, 1]))
    if total_trials <= 0.0:
        return 0.5
    return float(np.clip(np.sum(y[:, 0]) / total_trials, 1e-4, 1.0 - 1e-4))
