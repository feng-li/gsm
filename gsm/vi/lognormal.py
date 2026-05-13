"""Variational inference for LogNorm and LogNormRep mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import FitConfig, LogNormalMixtureSetting, LogNormalRepMixtureSetting
from gsm.data import Dataset
from gsm.models.lognormal import (
    LogNormalMixtureParams,
    log_prob as lognormal_log_prob,
    predict_mean_variance as lognormal_predict_mean_variance,
    responsibilities as lognormal_responsibilities,
)
from gsm.priors import (
    LogNormalMixtureCoefficientPriors,
    build_lognormal_mixture_priors,
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


@dataclass(frozen=True)
class LogNormalMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_scale: np.ndarray
    Z: np.ndarray




@dataclass(frozen=True)
class LogNormalMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    X_scale_c1: np.ndarray
    X_scale_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray




@dataclass(frozen=True)
class LogNormalMixturePosterior:
    """Mean-field Gaussian posterior over lognormal-mixture coefficients."""

    mean: LogNormalMixtureParams
    log_std: LogNormalMixtureParams




def prepare_lognormal_mixture_inputs(
    dataset: Dataset,
    setting: LogNormalMixtureSetting | LogNormalRepMixtureSetting,
    standardization: LogNormalMixtureStandardization | None = None,
) -> LogNormalMixtureInputs:
    """Build feature-specific design matrices for lognormal mixtures."""

    validate_positive_response(dataset.y)
    X_mean, X_scale, Z = _raw_lognormal_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {"X_mean": X_mean, "X_scale": X_scale, "Z": Z},
        setting.standardize,
        standardization,
    )

    return LogNormalMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=designs["X_mean"],
        X_scale=designs["X_scale"],
        Z=designs["Z"],
    )



def fit_lognormal_mixture_standardization(
    dataset: Dataset,
    setting: LogNormalMixtureSetting | LogNormalRepMixtureSetting,
) -> LogNormalMixtureStandardization:
    """Fit the lognormal model-specific scaling constants."""

    X_mean, X_scale, Z = _raw_lognormal_mixture_designs(dataset, setting)
    return LogNormalMixtureStandardization(
        **fit_design_standardization(
            {"X_mean": X_mean, "X_scale": X_scale, "Z": Z},
            setting.standardize,
        ),
    )



def initialize_lognormal_mixture_params(
    inputs: LogNormalMixtureInputs,
    setting: LogNormalMixtureSetting | LogNormalRepMixtureSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the lognormal VB optimizer."""

    y = np.asarray(inputs.y)
    n_components = setting.n_components
    quantiles = np.linspace(0.15, 0.85, n_components)
    if setting.parameterization == "response":
        means = np.maximum(np.quantile(y, quantiles), 1e-6)
        scale = max(float(np.std(y, ddof=1)), 1e-3)
    else:
        log_y = np.log(y)
        means = np.maximum(np.quantile(log_y, quantiles), 1e-6)
        scale = max(float(np.std(log_y, ddof=1)), 1e-3)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = np.log(means)
    scale_coef = np.zeros((n_components, inputs.X_scale.shape[1]))
    scale_coef[:, 0] = np.log(scale)
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "scale_coef": jnp.asarray(scale_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def initialize_lognormal_mixture_variational_params(
    inputs: LogNormalMixtureInputs,
    setting: LogNormalMixtureSetting | LogNormalRepMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family for lognormal mixtures."""

    mean_tree = initialize_lognormal_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)



def tree_to_lognormal_params(tree: dict[str, jnp.ndarray]) -> LogNormalMixtureParams:
    return LogNormalMixtureParams(
        mean_coef=tree["mean_coef"],
        scale_coef=tree["scale_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_lognormal_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> LogNormalMixturePosterior:
    return LogNormalMixturePosterior(
        mean=tree_to_lognormal_params(tree["mean"]),
        log_std=tree_to_lognormal_params(tree["log_std"]),
    )



def lognormal_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: LogNormalMixtureInputs,
    setting: LogNormalMixtureSetting | LogNormalRepMixtureSetting,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: LogNormalMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO for lognormal mixtures."""

    params = tree_to_lognormal_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_scale = jnp.asarray(inputs.X_scale)
    Z = jnp.asarray(inputs.Z)

    log_likelihood = lognormal_log_prob(
        params,
        y,
        X_mean,
        X_scale,
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
                "scale_coef",
                inputs.X_scale,
                coefficient_priors.scale if coefficient_priors is not None else None,
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


def lognormal_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: LogNormalMixtureInputs,
    setting: LogNormalMixtureSetting | LogNormalRepMixtureSetting,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: LogNormalMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a lognormal mean-field Gaussian posterior."""

    def sample_log_joint(param_tree):
        return lognormal_mixture_elbo(
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



def sample_lognormal_mixture_posterior(
    posterior: LogNormalMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted lognormal mean-field posterior."""

    return sample_posterior_tree(posterior, _lognormal_params_to_tree, seed, n_samples)


def _lognormal_params_to_tree(params: LogNormalMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "scale_coef": params.scale_coef,
        "gating_coef": params.gating_coef,
    }



def fit_lognormal_mixture_vb(
    dataset: Dataset,
    setting: LogNormalMixtureSetting | LogNormalRepMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the lognormal mixture scaffold with JAX gradients and local Adam."""

    fit = fit or FitConfig()
    inputs = prepare_lognormal_mixture_inputs(dataset, setting)
    coefficient_priors = build_lognormal_mixture_priors(inputs, setting)
    return fit_mean_field_mixture_vb(
        fit=fit,
        inputs=inputs,
        initialize_variational_params=lambda: initialize_lognormal_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        variational_elbo=lognormal_mixture_variational_elbo,
        coefficient_priors=coefficient_priors,
        build_result=lambda variational_params, history, converged: _build_lognormal_variational_result(
            variational_params, inputs, setting, history, converged
        ),
        extra_args=(setting,),
    )



def _raw_lognormal_mixture_designs(
    dataset: Dataset,
    setting: LogNormalMixtureSetting | LogNormalRepMixtureSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs_mix],
    )



def _build_lognormal_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: LogNormalMixtureInputs,
    setting: LogNormalMixtureSetting | LogNormalRepMixtureSetting,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_lognormal_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_scale = jnp.asarray(inputs.X_scale)
    Z = jnp.asarray(inputs.Z)
    resp = lognormal_responsibilities(
        posterior.mean,
        y,
        X_mean,
        X_scale,
        Z,
        parameterization=setting.parameterization,
    )
    pred_mean, pred_var = lognormal_predict_mean_variance(
        posterior.mean,
        X_mean,
        X_scale,
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
