"""Variational inference for split-normal mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import FitConfig, SplitNormalMixtureSetting
from gsm.data import Dataset
from gsm.models.splitnormal import (
    SplitNormalMixtureParams,
    log_prob as splitnormal_log_prob,
    predict_mean_variance as splitnormal_predict_mean_variance,
    responsibilities as splitnormal_responsibilities,
)
from gsm.priors import (
    SplitNormalMixtureCoefficientPriors,
    build_splitnormal_mixture_priors,
    coefficient_log_prior,
)
from gsm.vi.common import (
    VariationalResult,
    fit_design_standardization,
    standardize_designs,
)
from gsm.vi.engine import (
    initialize_mean_field_variational_params,
    monte_carlo_variational_elbo,
    optimize_restarts,
    sample_posterior_tree,
)


@dataclass(frozen=True)
class SplitNormalMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_scale: np.ndarray
    X_skewness: np.ndarray
    Z: np.ndarray




@dataclass(frozen=True)
class SplitNormalMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    X_scale_c1: np.ndarray
    X_scale_c2: np.ndarray
    X_skewness_c1: np.ndarray
    X_skewness_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray



@dataclass(frozen=True)
class SplitNormalMixturePosterior:
    """Mean-field Gaussian posterior over split-normal mixture coefficients."""

    mean: SplitNormalMixtureParams
    log_std: SplitNormalMixtureParams




def prepare_splitnormal_mixture_inputs(
    dataset: Dataset,
    setting: SplitNormalMixtureSetting,
    standardization: SplitNormalMixtureStandardization | None = None,
) -> SplitNormalMixtureInputs:
    """Build feature-specific design matrices for the split-normal mixture."""

    X_mean, X_scale, X_skewness, Z = _raw_splitnormal_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {"X_mean": X_mean, "X_scale": X_scale, "X_skewness": X_skewness, "Z": Z},
        setting.standardize,
        standardization,
    )

    return SplitNormalMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=designs["X_mean"],
        X_scale=designs["X_scale"],
        X_skewness=designs["X_skewness"],
        Z=designs["Z"],
    )



def fit_splitnormal_mixture_standardization(
    dataset: Dataset,
    setting: SplitNormalMixtureSetting,
) -> SplitNormalMixtureStandardization:
    """Fit the split-normal model-specific scaling constants."""

    X_mean, X_scale, X_skewness, Z = _raw_splitnormal_mixture_designs(dataset, setting)
    return SplitNormalMixtureStandardization(
        **fit_design_standardization(
            {"X_mean": X_mean, "X_scale": X_scale, "X_skewness": X_skewness, "Z": Z},
            setting.standardize,
        ),
    )



def initialize_splitnormal_mixture_params(
    inputs: SplitNormalMixtureInputs,
    setting: SplitNormalMixtureSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the split-normal VB optimizer."""

    y = np.asarray(inputs.y)
    n_components = setting.n_components
    quantiles = np.linspace(0.15, 0.85, n_components)
    means = np.quantile(y, quantiles)
    scale = max(float(np.std(y, ddof=1)), 1e-3)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = means
    scale_coef = np.zeros((n_components, inputs.X_scale.shape[1]))
    scale_coef[:, 0] = np.log(scale)
    skewness_coef = np.zeros((n_components, inputs.X_skewness.shape[1]))
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "scale_coef": jnp.asarray(scale_coef),
        "skewness_coef": jnp.asarray(skewness_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def initialize_splitnormal_mixture_variational_params(
    inputs: SplitNormalMixtureInputs,
    setting: SplitNormalMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family for split-normal mixtures."""

    mean_tree = initialize_splitnormal_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)



def tree_to_splitnormal_params(tree: dict[str, jnp.ndarray]) -> SplitNormalMixtureParams:
    return SplitNormalMixtureParams(
        mean_coef=tree["mean_coef"],
        scale_coef=tree["scale_coef"],
        skewness_coef=tree["skewness_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_splitnormal_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> SplitNormalMixturePosterior:
    return SplitNormalMixturePosterior(
        mean=tree_to_splitnormal_params(tree["mean"]),
        log_std=tree_to_splitnormal_params(tree["log_std"]),
    )



def splitnormal_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: SplitNormalMixtureInputs,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: SplitNormalMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO for the split-normal mixture."""

    params = tree_to_splitnormal_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_scale = jnp.asarray(inputs.X_scale)
    X_skewness = jnp.asarray(inputs.X_skewness)
    Z = jnp.asarray(inputs.Z)

    log_likelihood = splitnormal_log_prob(params, y, X_mean, X_scale, X_skewness, Z)
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
            param_tree["scale_coef"],
            inputs.X_scale,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
            coefficient_priors.scale if coefficient_priors is not None else None,
        )
        + coefficient_log_prior(
            param_tree["skewness_coef"],
            inputs.X_skewness,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
            coefficient_priors.skewness if coefficient_priors is not None else None,
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


def splitnormal_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: SplitNormalMixtureInputs,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: SplitNormalMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a split-normal mean-field Gaussian posterior."""

    def sample_log_joint(param_tree):
        return splitnormal_mixture_elbo(
            param_tree,
            inputs,
            coefficient_prior_scale=coefficient_prior_scale,
            use_ard=use_ard,
            ard_shape=ard_shape,
            ard_rate=ard_rate,
            coefficient_priors=coefficient_priors,
        )

    return monte_carlo_variational_elbo(variational_tree, noise_tree, sample_log_joint)



def sample_splitnormal_mixture_posterior(
    posterior: SplitNormalMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted split-normal mean-field posterior."""

    return sample_posterior_tree(posterior, _splitnormal_params_to_tree, seed, n_samples)


def _splitnormal_params_to_tree(params: SplitNormalMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "scale_coef": params.scale_coef,
        "skewness_coef": params.skewness_coef,
        "gating_coef": params.gating_coef,
    }



def fit_splitnormal_mixture_vb(
    dataset: Dataset,
    setting: SplitNormalMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the split-normal mixture scaffold with JAX gradients and local Adam."""

    fit = fit or FitConfig()
    inputs = prepare_splitnormal_mixture_inputs(dataset, setting)
    coefficient_priors = build_splitnormal_mixture_priors(inputs, setting)
    return optimize_restarts(
        fit,
        lambda: initialize_splitnormal_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        lambda noise_tree: lambda q: splitnormal_mixture_variational_elbo(
                q,
                inputs,
                noise_tree,
                coefficient_prior_scale=fit.coefficient_prior_scale,
                use_ard=fit.use_ard,
                ard_shape=fit.ard_shape,
                ard_rate=fit.ard_rate,
                coefficient_priors=coefficient_priors,
            ),
        lambda variational_params, history, converged: _build_splitnormal_variational_result(
            variational_params, inputs, history, converged
        )
    )



def _raw_splitnormal_mixture_designs(
    dataset: Dataset,
    setting: SplitNormalMixtureSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs[2]],
        dataset.X[:, setting.covs_mix],
    )



def _build_splitnormal_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: SplitNormalMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_splitnormal_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_scale = jnp.asarray(inputs.X_scale)
    X_skewness = jnp.asarray(inputs.X_skewness)
    Z = jnp.asarray(inputs.Z)
    resp = splitnormal_responsibilities(posterior.mean, y, X_mean, X_scale, X_skewness, Z)
    pred_mean, pred_var = splitnormal_predict_mean_variance(
        posterior.mean,
        X_mean,
        X_scale,
        X_skewness,
        Z,
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
