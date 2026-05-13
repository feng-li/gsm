"""Variational inference for split-t mixtures."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gsm.config import FitConfig, SplitTMixtureSetting
from gsm.data import Dataset
from gsm.models.splitt import (
    SplitTMixtureParams,
    log_prob as splitt_log_prob,
    predict_mean_variance as splitt_predict_mean_variance,
    responsibilities as splitt_responsibilities,
)
from gsm.priors import (
    SplitTMixtureCoefficientPriors,
    build_splitt_mixture_priors,
    coefficient_log_prior_sum,
)
from gsm.vi.common import (
    VariationalResult,
    fit_design_standardization,
    standardize_designs,
)
from gsm.vi.engine import (
    fit_mean_field_mixture_vb,
    initialize_mean_field_variational_params,
    monte_carlo_variational_elbo,
    sample_posterior_tree,
)


@dataclass(frozen=True)
class SplitTMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_df: np.ndarray
    X_scale: np.ndarray
    X_skewness: np.ndarray
    Z: np.ndarray




@dataclass(frozen=True)
class SplitTMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    X_df_c1: np.ndarray
    X_df_c2: np.ndarray
    X_scale_c1: np.ndarray
    X_scale_c2: np.ndarray
    X_skewness_c1: np.ndarray
    X_skewness_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray




@dataclass(frozen=True)
class SplitTMixturePosterior:
    """Mean-field Gaussian posterior over split-t mixture coefficients."""

    mean: SplitTMixtureParams
    log_std: SplitTMixtureParams




def prepare_splitt_mixture_inputs(
    dataset: Dataset,
    setting: SplitTMixtureSetting,
    standardization: SplitTMixtureStandardization | None = None,
) -> SplitTMixtureInputs:
    """Build feature-specific design matrices for the split-t mixture."""

    X_mean, X_df, X_scale, X_skewness, Z = _raw_splitt_mixture_designs(dataset, setting)
    designs = standardize_designs(
        {
            "X_mean": X_mean,
            "X_df": X_df,
            "X_scale": X_scale,
            "X_skewness": X_skewness,
            "Z": Z,
        },
        setting.standardize,
        standardization,
    )

    return SplitTMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=designs["X_mean"],
        X_df=designs["X_df"],
        X_scale=designs["X_scale"],
        X_skewness=designs["X_skewness"],
        Z=designs["Z"],
    )



def fit_splitt_mixture_standardization(
    dataset: Dataset,
    setting: SplitTMixtureSetting,
) -> SplitTMixtureStandardization:
    """Fit the split-t model-specific scaling constants."""

    X_mean, X_df, X_scale, X_skewness, Z = _raw_splitt_mixture_designs(dataset, setting)
    return SplitTMixtureStandardization(
        **fit_design_standardization(
            {
                "X_mean": X_mean,
                "X_df": X_df,
                "X_scale": X_scale,
                "X_skewness": X_skewness,
                "Z": Z,
            },
            setting.standardize,
        ),
    )



def initialize_splitt_mixture_params(
    inputs: SplitTMixtureInputs,
    setting: SplitTMixtureSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the split-t VB optimizer."""

    y = np.asarray(inputs.y)
    n_components = setting.n_components
    quantiles = np.linspace(0.15, 0.85, n_components)
    means = np.quantile(y, quantiles)
    scale = max(float(np.std(y, ddof=1)), 1e-3)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = means
    df_coef = np.zeros((n_components, inputs.X_df.shape[1]))
    df_coef[:, 0] = np.log(10.0)
    scale_coef = np.zeros((n_components, inputs.X_scale.shape[1]))
    scale_coef[:, 0] = np.log(scale)
    skewness_coef = np.zeros((n_components, inputs.X_skewness.shape[1]))
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "df_coef": jnp.asarray(df_coef),
        "scale_coef": jnp.asarray(scale_coef),
        "skewness_coef": jnp.asarray(skewness_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def initialize_splitt_mixture_variational_params(
    inputs: SplitTMixtureInputs,
    setting: SplitTMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family for split-t mixtures."""

    mean_tree = initialize_splitt_mixture_params(inputs, setting)
    return initialize_mean_field_variational_params(mean_tree, init_log_std)



def tree_to_splitt_params(tree: dict[str, jnp.ndarray]) -> SplitTMixtureParams:
    return SplitTMixtureParams(
        mean_coef=tree["mean_coef"],
        df_coef=tree["df_coef"],
        scale_coef=tree["scale_coef"],
        skewness_coef=tree["skewness_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_splitt_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> SplitTMixturePosterior:
    return SplitTMixturePosterior(
        mean=tree_to_splitt_params(tree["mean"]),
        log_std=tree_to_splitt_params(tree["log_std"]),
    )



def splitt_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: SplitTMixtureInputs,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: SplitTMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO for the split-t mixture."""

    params = tree_to_splitt_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_df = jnp.asarray(inputs.X_df)
    X_scale = jnp.asarray(inputs.X_scale)
    X_skewness = jnp.asarray(inputs.X_skewness)
    Z = jnp.asarray(inputs.Z)

    log_likelihood = splitt_log_prob(params, y, X_mean, X_df, X_scale, X_skewness, Z)
    log_prior = coefficient_log_prior_sum(
        param_tree,
        (
            (
                "mean_coef",
                inputs.X_mean,
                coefficient_priors.mean if coefficient_priors is not None else None,
            ),
            (
                "df_coef",
                inputs.X_df,
                coefficient_priors.df if coefficient_priors is not None else None,
            ),
            (
                "scale_coef",
                inputs.X_scale,
                coefficient_priors.scale if coefficient_priors is not None else None,
            ),
            (
                "skewness_coef",
                inputs.X_skewness,
                coefficient_priors.skewness if coefficient_priors is not None else None,
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


def splitt_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: SplitTMixtureInputs,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: SplitTMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a split-t mean-field Gaussian posterior."""

    def sample_log_joint(param_tree):
        return splitt_mixture_elbo(
            param_tree,
            inputs,
            coefficient_prior_scale=coefficient_prior_scale,
            use_ard=use_ard,
            ard_shape=ard_shape,
            ard_rate=ard_rate,
            coefficient_priors=coefficient_priors,
        )

    return monte_carlo_variational_elbo(variational_tree, noise_tree, sample_log_joint)



def sample_splitt_mixture_posterior(
    posterior: SplitTMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted split-t mean-field posterior."""

    return sample_posterior_tree(posterior, _splitt_params_to_tree, seed, n_samples)


def _splitt_params_to_tree(params: SplitTMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "df_coef": params.df_coef,
        "scale_coef": params.scale_coef,
        "skewness_coef": params.skewness_coef,
        "gating_coef": params.gating_coef,
    }



def fit_splitt_mixture_vb(
    dataset: Dataset,
    setting: SplitTMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the split-t mixture scaffold with JAX gradients and local Adam."""

    fit = fit or FitConfig()
    inputs = prepare_splitt_mixture_inputs(dataset, setting)
    coefficient_priors = build_splitt_mixture_priors(inputs, setting)
    return fit_mean_field_mixture_vb(
        fit=fit,
        inputs=inputs,
        initialize_variational_params=lambda: initialize_splitt_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        ),
        variational_elbo=splitt_mixture_variational_elbo,
        coefficient_priors=coefficient_priors,
        build_result=lambda variational_params, history, converged: _build_splitt_variational_result(
            variational_params, inputs, history, converged
        ),
    )



def _raw_splitt_mixture_designs(
    dataset: Dataset,
    setting: SplitTMixtureSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs[2]],
        dataset.X[:, setting.covs[3]],
        dataset.X[:, setting.covs_mix],
    )



def _build_splitt_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: SplitTMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_splitt_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_df = jnp.asarray(inputs.X_df)
    X_scale = jnp.asarray(inputs.X_scale)
    X_skewness = jnp.asarray(inputs.X_skewness)
    Z = jnp.asarray(inputs.Z)
    resp = splitt_responsibilities(posterior.mean, y, X_mean, X_df, X_scale, X_skewness, Z)
    pred_mean, pred_var = splitt_predict_mean_variance(
        posterior.mean,
        X_mean,
        X_df,
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
