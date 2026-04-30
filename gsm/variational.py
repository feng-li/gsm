"""Minimal variational inference framework for the Gaussian mixture sketch.

The first inference path uses a collapsed-responsibility objective: allocations
are integrated out through ``logsumexp`` and the coefficient variational family
is a diagonal Gaussian. This is intentionally modest, but it gives us the right
API surface for replacing the old MATLAB Metropolis-Hastings/Newton path with an
ELBO-style JAX optimization loop.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import FitConfig, GaussianMixtureSetting, ModelConfig, SplitTMixtureSetting
from .data import Dataset, standardize_covariates
from .models.gaussian_mixture import (
    GaussianMixtureParams,
    log_prob as gaussian_log_prob,
    predict_mean_variance,
    responsibilities as gaussian_responsibilities,
)
from .models.splitt import (
    SplitTMixtureParams,
    log_prob as splitt_log_prob,
    predict_mean_variance as splitt_predict_mean_variance,
    responsibilities as splitt_responsibilities,
)
from .priors import (
    GaussianMixtureCoefficientPriors,
    SplitTMixtureCoefficientPriors,
    build_gaussian_mixture_priors,
    build_splitt_mixture_priors,
    coefficient_log_prior,
)


@dataclass(frozen=True)
class GaussianMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_variance: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class SplitTMixtureInputs:
    y: np.ndarray
    X_mean: np.ndarray
    X_df: np.ndarray
    X_scale: np.ndarray
    X_skewness: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class GaussianMixtureStandardization:
    X_mean_c1: np.ndarray
    X_mean_c2: np.ndarray
    X_variance_c1: np.ndarray
    X_variance_c2: np.ndarray
    Z_c1: np.ndarray
    Z_c2: np.ndarray


@dataclass(frozen=True)
class GaussianMixturePosterior:
    """Mean-field Gaussian posterior over Gaussian-mixture coefficients."""

    mean: GaussianMixtureParams
    log_std: GaussianMixtureParams


@dataclass(frozen=True)
class SplitTMixturePosterior:
    """Mean-field Gaussian posterior over split-t mixture coefficients."""

    mean: SplitTMixtureParams
    log_std: SplitTMixtureParams


@dataclass(frozen=True)
class VariationalResult:
    params: Any
    elbo_history: np.ndarray
    converged: bool
    posterior: Any | None = None
    responsibilities: np.ndarray | None = None
    predictive_mean: np.ndarray | None = None
    predictive_variance: np.ndarray | None = None


def prepare_gaussian_mixture_inputs(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
    standardization: GaussianMixtureStandardization | None = None,
) -> GaussianMixtureInputs:
    """Build feature-specific design matrices from a loaded dataset."""

    X_mean, X_variance, Z = _raw_gaussian_mixture_designs(dataset, setting)

    if setting.standardize:
        if standardization is None:
            X_mean, _, _ = standardize_covariates(X_mean, setting.standardize)
            X_variance, _, _ = standardize_covariates(X_variance, setting.standardize)
            Z, _, _ = standardize_covariates(Z, setting.standardize)
        else:
            X_mean = _apply_standardization(
                X_mean,
                setting.standardize,
                standardization.X_mean_c1,
                standardization.X_mean_c2,
            )
            X_variance = _apply_standardization(
                X_variance,
                setting.standardize,
                standardization.X_variance_c1,
                standardization.X_variance_c2,
            )
            Z = _apply_standardization(
                Z,
                setting.standardize,
                standardization.Z_c1,
                standardization.Z_c2,
            )

    return GaussianMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=X_mean,
        X_variance=X_variance,
        Z=Z,
    )


def prepare_splitt_mixture_inputs(
    dataset: Dataset,
    setting: SplitTMixtureSetting,
) -> SplitTMixtureInputs:
    """Build feature-specific design matrices for the split-t mixture."""

    X_mean, X_df, X_scale, X_skewness, Z = _raw_splitt_mixture_designs(dataset, setting)
    if setting.standardize:
        X_mean, _, _ = standardize_covariates(X_mean, setting.standardize)
        X_df, _, _ = standardize_covariates(X_df, setting.standardize)
        X_scale, _, _ = standardize_covariates(X_scale, setting.standardize)
        X_skewness, _, _ = standardize_covariates(X_skewness, setting.standardize)
        Z, _, _ = standardize_covariates(Z, setting.standardize)

    return SplitTMixtureInputs(
        y=dataset.y.reshape(-1),
        X_mean=X_mean,
        X_df=X_df,
        X_scale=X_scale,
        X_skewness=X_skewness,
        Z=Z,
    )


def fit_gaussian_mixture_standardization(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
) -> GaussianMixtureStandardization:
    """Fit the model-specific scaling constants used by the design matrices."""

    X_mean, X_variance, Z = _raw_gaussian_mixture_designs(dataset, setting)
    _, X_mean_c1, X_mean_c2 = standardize_covariates(X_mean, setting.standardize)
    _, X_variance_c1, X_variance_c2 = standardize_covariates(X_variance, setting.standardize)
    _, Z_c1, Z_c2 = standardize_covariates(Z, setting.standardize)
    return GaussianMixtureStandardization(
        X_mean_c1=X_mean_c1,
        X_mean_c2=X_mean_c2,
        X_variance_c1=X_variance_c1,
        X_variance_c2=X_variance_c2,
        Z_c1=Z_c1,
        Z_c2=Z_c2,
    )


def initialize_gaussian_mixture_params(
    inputs: GaussianMixtureInputs,
    setting: GaussianMixtureSetting,
) -> dict[str, jnp.ndarray]:
    """Deterministic initialization for the first VB optimizer."""

    y = np.asarray(inputs.y)
    n_components = setting.n_components
    quantiles = np.linspace(0.15, 0.85, n_components)
    means = np.quantile(y, quantiles)
    variance = max(float(np.var(y, ddof=1)), 1e-6)

    mean_coef = np.zeros((n_components, inputs.X_mean.shape[1]))
    mean_coef[:, 0] = means
    log_variance_coef = np.zeros((n_components, inputs.X_variance.shape[1]))
    log_variance_coef[:, 0] = np.log(variance)
    gating_coef = np.zeros((max(n_components - 1, 0), inputs.Z.shape[1]))

    return {
        "mean_coef": jnp.asarray(mean_coef),
        "log_variance_coef": jnp.asarray(log_variance_coef),
        "gating_coef": jnp.asarray(gating_coef),
    }


def initialize_gaussian_mixture_variational_params(
    inputs: GaussianMixtureInputs,
    setting: GaussianMixtureSetting,
    init_log_std: float = -5.0,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Initialize a diagonal Gaussian variational family."""

    mean_tree = initialize_gaussian_mixture_params(inputs, setting)
    log_std_tree = jax.tree_util.tree_map(
        lambda value: jnp.full_like(value, init_log_std),
        mean_tree,
    )
    return {"mean": mean_tree, "log_std": log_std_tree}


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
    log_std_tree = jax.tree_util.tree_map(
        lambda value: jnp.full_like(value, init_log_std),
        mean_tree,
    )
    return {"mean": mean_tree, "log_std": log_std_tree}


def tree_to_gaussian_params(tree: dict[str, jnp.ndarray]) -> GaussianMixtureParams:
    return GaussianMixtureParams(
        mean_coef=tree["mean_coef"],
        log_variance_coef=tree["log_variance_coef"],
        gating_coef=tree["gating_coef"],
    )


def tree_to_gaussian_posterior(
    tree: dict[str, dict[str, jnp.ndarray]],
) -> GaussianMixturePosterior:
    return GaussianMixturePosterior(
        mean=tree_to_gaussian_params(tree["mean"]),
        log_std=tree_to_gaussian_params(tree["log_std"]),
    )


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


def gaussian_mixture_elbo(
    param_tree: dict[str, jnp.ndarray],
    inputs: GaussianMixtureInputs,
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: GaussianMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Collapsed-allocation ELBO with optional ARD coefficient shrinkage."""

    params = tree_to_gaussian_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_variance = jnp.asarray(inputs.X_variance)
    Z = jnp.asarray(inputs.Z)

    log_likelihood = gaussian_log_prob(params, y, X_mean, X_variance, Z)
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
            param_tree["log_variance_coef"],
            inputs.X_variance,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
            coefficient_priors.log_variance if coefficient_priors is not None else None,
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


def gaussian_mixture_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: GaussianMixtureInputs,
    noise_tree: dict[str, jnp.ndarray],
    coefficient_prior_scale: float = 10.0,
    use_ard: bool = False,
    ard_shape: float = 1e-2,
    ard_rate: float = 1e-2,
    coefficient_priors: GaussianMixtureCoefficientPriors | None = None,
) -> jnp.ndarray:
    """Monte Carlo ELBO for a diagonal Gaussian variational posterior."""

    sample_tree = _sample_param_trees(
        variational_tree["mean"],
        variational_tree["log_std"],
        noise_tree,
    )

    def sample_log_joint(param_tree):
        return gaussian_mixture_elbo(
            param_tree,
            inputs,
            coefficient_prior_scale=coefficient_prior_scale,
            use_ard=use_ard,
            ard_shape=ard_shape,
            ard_rate=ard_rate,
            coefficient_priors=coefficient_priors,
        )

    log_joint = jax.vmap(sample_log_joint)(sample_tree)
    return jnp.mean(log_joint) + _mean_field_gaussian_entropy(variational_tree["log_std"])


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
            param_tree["df_coef"],
            inputs.X_df,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
            coefficient_priors.df if coefficient_priors is not None else None,
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

    sample_tree = _sample_param_trees(
        variational_tree["mean"],
        variational_tree["log_std"],
        noise_tree,
    )

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

    log_joint = jax.vmap(sample_log_joint)(sample_tree)
    return jnp.mean(log_joint) + _mean_field_gaussian_entropy(variational_tree["log_std"])


def sample_gaussian_mixture_posterior(
    posterior: GaussianMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted mean-field Gaussian posterior."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    mean_tree = _gaussian_params_to_tree(posterior.mean)
    log_std_tree = _gaussian_params_to_tree(posterior.log_std)
    noise_tree = _sample_noise_like(mean_tree, jax.random.PRNGKey(seed), n_samples)
    return _sample_param_trees(mean_tree, log_std_tree, noise_tree)


def sample_splitt_mixture_posterior(
    posterior: SplitTMixturePosterior,
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    """Draw coefficient trees from a fitted split-t mean-field posterior."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    mean_tree = _splitt_params_to_tree(posterior.mean)
    log_std_tree = _splitt_params_to_tree(posterior.log_std)
    noise_tree = _sample_noise_like(mean_tree, jax.random.PRNGKey(seed), n_samples)
    return _sample_param_trees(mean_tree, log_std_tree, noise_tree)


def fit_gaussian_mixture_vb(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the Gaussian mixture scaffold with JAX gradients and local Adam."""

    fit = fit or FitConfig()
    if fit.n_elbo_samples < 1:
        raise ValueError("n_elbo_samples must be positive")

    inputs = prepare_gaussian_mixture_inputs(dataset, setting)
    coefficient_priors = build_gaussian_mixture_priors(inputs, setting)
    best_result: VariationalResult | None = None

    for restart in range(fit.n_restarts):
        variational_params = initialize_gaussian_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        )
        if restart:
            variational_params["mean"] = _jitter_params(
                variational_params["mean"],
                fit.seed + restart,
            )

        noise_tree = _sample_noise_like(
            variational_params["mean"],
            jax.random.PRNGKey(fit.seed + 1009 * (restart + 1)),
            fit.n_elbo_samples,
        )
        variational_params, history, converged = _optax_maximize(
            variational_params,
            lambda q: gaussian_mixture_variational_elbo(
                q,
                inputs,
                noise_tree,
                coefficient_prior_scale=fit.coefficient_prior_scale,
                use_ard=fit.use_ard,
                ard_shape=fit.ard_shape,
                ard_rate=fit.ard_rate,
                coefficient_priors=coefficient_priors,
            ),
            max_iter=fit.max_iter,
            learning_rate=fit.learning_rate,
            tol=fit.tol,
        )
        result = _build_variational_result(variational_params, inputs, history, converged)
        if best_result is None or result.elbo_history[-1] > best_result.elbo_history[-1]:
            best_result = result

    if best_result is None:
        raise RuntimeError("no variational optimization runs were executed")
    return best_result


def fit_splitt_mixture_vb(
    dataset: Dataset,
    setting: SplitTMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Fit the split-t mixture scaffold with JAX gradients and local Adam."""

    fit = fit or FitConfig()
    if fit.n_elbo_samples < 1:
        raise ValueError("n_elbo_samples must be positive")

    inputs = prepare_splitt_mixture_inputs(dataset, setting)
    coefficient_priors = build_splitt_mixture_priors(inputs, setting)
    best_result: VariationalResult | None = None

    for restart in range(fit.n_restarts):
        variational_params = initialize_splitt_mixture_variational_params(
            inputs,
            setting,
            init_log_std=fit.posterior_init_log_std,
        )
        if restart:
            variational_params["mean"] = _jitter_params(
                variational_params["mean"],
                fit.seed + restart,
            )

        noise_tree = _sample_noise_like(
            variational_params["mean"],
            jax.random.PRNGKey(fit.seed + 1009 * (restart + 1)),
            fit.n_elbo_samples,
        )
        variational_params, history, converged = _optax_maximize(
            variational_params,
            lambda q: splitt_mixture_variational_elbo(
                q,
                inputs,
                noise_tree,
                coefficient_prior_scale=fit.coefficient_prior_scale,
                use_ard=fit.use_ard,
                ard_shape=fit.ard_shape,
                ard_rate=fit.ard_rate,
                coefficient_priors=coefficient_priors,
            ),
            max_iter=fit.max_iter,
            learning_rate=fit.learning_rate,
            tol=fit.tol,
        )
        result = _build_splitt_variational_result(variational_params, inputs, history, converged)
        if best_result is None or result.elbo_history[-1] > best_result.elbo_history[-1]:
            best_result = result

    if best_result is None:
        raise RuntimeError("no variational optimization runs were executed")
    return best_result


def fit_variational(
    dataset: Dataset,
    model: ModelConfig | GaussianMixtureSetting | SplitTMixtureSetting,
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Dispatch to the available variational inference implementation."""

    if isinstance(model, GaussianMixtureSetting):
        return fit_gaussian_mixture_vb(dataset, model, fit)
    if isinstance(model, SplitTMixtureSetting):
        return fit_splitt_mixture_vb(dataset, model, fit)
    raise NotImplementedError(f"no variational implementation for model type {type(model)!r}")


def _optax_maximize(
    params,
    objective,
    max_iter: int,
    learning_rate: float,
    tol: float,
):
    loss_and_grad = jax.value_and_grad(lambda p: -objective(p))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    history: list[float] = []
    converged = False

    for _ in range(1, max_iter + 1):
        loss, grad = loss_and_grad(params)
        history.append(float(-loss))
        if len(history) > 5 and abs(history[-1] - history[-2]) < tol:
            converged = True
            break

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

    return params, np.asarray(history), converged


def _gaussian_params_to_tree(params: GaussianMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "log_variance_coef": params.log_variance_coef,
        "gating_coef": params.gating_coef,
    }


def _splitt_params_to_tree(params: SplitTMixtureParams) -> dict[str, jnp.ndarray]:
    return {
        "mean_coef": params.mean_coef,
        "df_coef": params.df_coef,
        "scale_coef": params.scale_coef,
        "skewness_coef": params.skewness_coef,
        "gating_coef": params.gating_coef,
    }


def _sample_noise_like(
    param_tree: dict[str, jnp.ndarray],
    key,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    leaves, tree_def = jax.tree_util.tree_flatten(param_tree)
    keys = jax.random.split(key, len(leaves))
    noise_leaves = [
        jax.random.normal(sample_key, (n_samples, *leaf.shape), dtype=leaf.dtype)
        for sample_key, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(tree_def, noise_leaves)


def _sample_param_trees(
    mean_tree: dict[str, jnp.ndarray],
    log_std_tree: dict[str, jnp.ndarray],
    noise_tree: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    return jax.tree_util.tree_map(
        lambda mean, log_std, noise: mean + jnp.exp(log_std) * noise,
        mean_tree,
        log_std_tree,
        noise_tree,
    )


def _mean_field_gaussian_entropy(log_std_tree: dict[str, jnp.ndarray]) -> jnp.ndarray:
    log_two_pi_e = jnp.log(2.0 * jnp.pi * jnp.e)
    return sum(
        jnp.sum(log_std + 0.5 * log_two_pi_e)
        for log_std in jax.tree_util.tree_leaves(log_std_tree)
    )


def _raw_gaussian_mixture_designs(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        dataset.X[:, setting.covs[0]],
        dataset.X[:, setting.covs[1]],
        dataset.X[:, setting.covs_mix],
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


def _apply_standardization(
    X: np.ndarray,
    method: int,
    c1: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    if X.shape[1] != c1.shape[0] or X.shape[1] != c2.shape[0]:
        raise ValueError("standardization constants do not match design matrix")
    if method == 0:
        return X

    use_column = ~(np.isnan(c1) | np.isnan(c2))
    if not np.any(use_column):
        return X

    if method == 1:
        X[:, use_column] = (X[:, use_column] - c1[use_column]) / c2[use_column]
    elif method == 2:
        scale = 2.0 / (c2[use_column] - c1[use_column])
        offset = 1.0 - scale * c2[use_column]
        X[:, use_column] = X[:, use_column] * scale + offset
    elif method == 3:
        X[:, use_column] = X[:, use_column] - c1[use_column]
    else:
        raise ValueError(f"unknown standardization method: {method}")
    return X


def _build_result(
    param_tree: dict[str, jnp.ndarray],
    inputs: GaussianMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    params = tree_to_gaussian_params(param_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_variance = jnp.asarray(inputs.X_variance)
    Z = jnp.asarray(inputs.Z)
    resp = gaussian_responsibilities(params, y, X_mean, X_variance, Z)
    pred_mean, pred_var = predict_mean_variance(params, X_mean, X_variance, Z)
    return VariationalResult(
        params=params,
        elbo_history=history,
        converged=converged,
        responsibilities=np.asarray(resp),
        predictive_mean=np.asarray(pred_mean),
        predictive_variance=np.asarray(pred_var),
    )


def _build_variational_result(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    inputs: GaussianMixtureInputs,
    history: np.ndarray,
    converged: bool,
) -> VariationalResult:
    posterior = tree_to_gaussian_posterior(variational_tree)
    y = jnp.asarray(inputs.y)
    X_mean = jnp.asarray(inputs.X_mean)
    X_variance = jnp.asarray(inputs.X_variance)
    Z = jnp.asarray(inputs.Z)
    resp = gaussian_responsibilities(posterior.mean, y, X_mean, X_variance, Z)
    pred_mean, pred_var = predict_mean_variance(posterior.mean, X_mean, X_variance, Z)
    return VariationalResult(
        params=posterior.mean,
        posterior=posterior,
        elbo_history=history,
        converged=converged,
        responsibilities=np.asarray(resp),
        predictive_mean=np.asarray(pred_mean),
        predictive_variance=np.asarray(pred_var),
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


def _jitter_params(params: dict[str, jnp.ndarray], seed: int) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: value + jnp.asarray(0.01 * rng.standard_normal(value.shape))
        for name, value in params.items()
    }
