"""Variational inference for binomial mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import BinomialMixtureSetting, FitConfig
from gsm.data import Dataset
from gsm.models.binomial import (
    BinomialMixtureParams,
    log_prob as binomial_log_prob,
    predict_mean_variance as binomial_predict_mean_variance,
    responsibilities as binomial_responsibilities,
)
from gsm.priors import (
    BinomialMixtureCoefficientPriors,
    build_binomial_mixture_priors,
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
class BinomialMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class BinomialMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray


@dataclass(frozen=True)
class BinomialMixturePosterior:
    """Mean-field Gaussian posterior over binomial-mixture coefficients."""

    mean: BinomialMixtureParams
    log_std: BinomialMixtureParams


def prepare_binomial_mixture_inputs(
    dataset: Dataset,
    setting: BinomialMixtureSetting,
    standardization: BinomialMixtureStandardization | None = None,
) -> BinomialMixtureInputs:
    """Build feature-specific design matrices for binomial mixtures."""

    validate_binomial_response(dataset.y)
    X_mean, Z = _raw_binomial_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {"X_mean": X_mean, "Z": Z},
        setting.standardize,
        standardization,
    )
    return BinomialMixtureInputs(
        y=np.asarray(dataset.y, dtype=float),
        X_mean=designs["X_mean"],
        Z=designs["Z"],
    )


def fit_binomial_mixture_standardization(
    dataset: Dataset,
    setting: BinomialMixtureSetting,
) -> BinomialMixtureStandardization:
    """Fit the binomial model-specific scaling constants."""

    X_mean, Z = _raw_binomial_mixture_designs(dataset, setting)
    return BinomialMixtureStandardization(
        **fit_design_standardization(
            {"X_mean": X_mean, "Z": Z},
            setting.standardize,
        ),
    )


def initialize_binomial_mixture_params(
    inputs: BinomialMixtureInputs,
    setting: BinomialMixtureSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the binomial VB optimizer."""

    rates = _success_rates(inputs.y)
    n_components = setting.n_components
    if n_components == 1:
        means = np.asarray([_pooled_rate(inputs.y)])
    else:
        quantiles = np.linspace(0.15, 0.85, n_components)
        means = np.quantile(rates, quantiles)
    means = np.clip(means, 1e-4, 1.0 - 1e-4)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = np.log(means / (1.0 - means))
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def initialize_binomial_mixture_variational_params(
    inputs: BinomialMixtureInputs,
    setting: BinomialMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family."""

    mean_tree = initialize_binomial_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)


def tree_to_binomial_params(tree: dict[str, jnp.ndarray]) -> BinomialMixtureParams:
    return BinomialMixtureParams(
        mean_coef=tree["mean_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_binomial_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> BinomialMixturePosterior:
    return BinomialMixturePosterior(
        mean=tree_to_binomial_params(tree["mean"]),
        log_std=tree_to_binomial_params(tree["log_std"]),
    )


def binomial_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: BinomialMixtureInputs,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: BinomialMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO for binomial mixtures."""

    params = tree_to_binomial_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    Z = jnp.asarray(inputs.Z)
    log_likelihood = binomial_log_prob(params, y, X_mean, Z)
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


def binomial_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: BinomialMixtureInputs,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: BinomialMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a binomial mean-field Gaussian posterior."""

    def sample_log_joint(param_tree):
        return binomial_mixture_elbo(
            param_tree,
            inputs,
            coefficient_prior_scale=coefficient_prior_scale,
            use_ard=use_ard,
            ard_shape=ard_shape,
            ard_rate=ard_rate,
            coefficient_priors=coefficient_priors,
        )

    return monte_carlo_variational_elbo(variational_tree, noise_tree, sample_log_joint)


def sample_binomial_mixture_posterior(
    posterior: BinomialMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted binomial mean-field posterior."""

    return sample_posterior_tree(posterior, _binomial_params_to_tree, seed, n_samples)


def _binomial_params_to_tree(params: BinomialMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "gating_coef": params.gating_coef,
    }


def fit_binomial_mixture_vb(
    dataset: Dataset,
    setting: BinomialMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the binomial mixture scaffold with JAX gradients and Adam."""

    fit = fit or FitConfig()
    inputs = prepare_binomial_mixture_inputs(dataset, setting)
    coefficient_priors = build_binomial_mixture_priors(inputs, setting)
    return optimize_restarts(
        fit,
        lambda: initialize_binomial_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        lambda noise_tree: lambda q: binomial_mixture_variational_elbo(
            q,
            inputs,
            noise_tree,
            coefficient_prior_scale=fit.coefficient_prior_scale,
            use_ard=fit.use_ard,
            ard_shape=fit.ard_shape,
            ard_rate=fit.ard_rate,
            coefficient_priors=coefficient_priors,
        ),
        lambda variational_params, history, converged: _build_binomial_variational_result(
            variational_params, inputs, history, converged
        ),
    )


def _raw_binomial_mixture_designs(
    dataset: Dataset,
    setting: BinomialMixtureSetting,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs_mix],
    )


def _build_binomial_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: BinomialMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_binomial_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    Z = jnp.asarray(inputs.Z)
    resp = binomial_responsibilities(posterior.mean, y, X_mean, Z)
    pred_mean, pred_var = binomial_predict_mean_variance(
        posterior.mean, X_mean, Z, trials=y
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
