"""Predictive scoring utilities for VB fits."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import vmap
from jax.nn import logsumexp
import numpy as np

from .config import (
    BetaRegMixtureSetting,
    FitConfig,
    GammaMixtureSetting,
    GammaRepMixtureSetting,
    GaussianMixtureSetting,
    LogNormalMixtureSetting,
    LogNormalRepMixtureSetting,
    SplitNormalMixtureSetting,
    SplitTMixtureSetting,
)
from .data import Dataset, subset_dataset
from .models.betareg import log_prob_observations as betareg_log_prob_observations
from .models.gamma import log_prob_observations as gamma_log_prob_observations
from .models.gaussian import log_prob_observations as gaussian_log_prob_observations
from .models.lognormal import log_prob_observations as lognormal_log_prob_observations
from .models.splitnormal import log_prob_observations as splitnormal_log_prob_observations
from .models.splitt import log_prob_observations as splitt_log_prob_observations
from .variational import (
    BetaRegMixtureStandardization,
    GammaMixtureStandardization,
    GaussianMixtureStandardization,
    LogNormalMixtureStandardization,
    SplitNormalMixtureStandardization,
    SplitTMixtureStandardization,
    VariationalResult,
    fit_betareg_mixture_standardization,
    fit_gamma_mixture_standardization,
    fit_gaussian_mixture_standardization,
    fit_lognormal_mixture_standardization,
    fit_splitnormal_mixture_standardization,
    fit_splitt_mixture_standardization,
    fit_variational,
    prepare_betareg_mixture_inputs,
    prepare_gamma_mixture_inputs,
    prepare_gaussian_mixture_inputs,
    prepare_lognormal_mixture_inputs,
    prepare_splitnormal_mixture_inputs,
    prepare_splitt_mixture_inputs,
    sample_betareg_mixture_posterior,
    sample_gamma_mixture_posterior,
    sample_gaussian_mixture_posterior,
    sample_lognormal_mixture_posterior,
    sample_splitnormal_mixture_posterior,
    sample_splitt_mixture_posterior,
    tree_to_betareg_params,
    tree_to_gamma_params,
    tree_to_gaussian_params,
    tree_to_lognormal_params,
    tree_to_splitnormal_params,
    tree_to_splitt_params,
)


ModelSetting = (
    BetaRegMixtureSetting
    | GammaMixtureSetting
    | GammaRepMixtureSetting
    | GaussianMixtureSetting
    | LogNormalMixtureSetting
    | LogNormalRepMixtureSetting
    | SplitNormalMixtureSetting
    | SplitTMixtureSetting
)
ModelStandardization = (
    BetaRegMixtureStandardization
    | GammaMixtureStandardization
    | GaussianMixtureStandardization
    | LogNormalMixtureStandardization
    | SplitNormalMixtureStandardization
    | SplitTMixtureStandardization
)


@dataclass(frozen=True)
class PredictiveLogScore:
    """Pointwise and aggregate predictive log-density score."""

    pointwise: np.ndarray
    elpd: float
    mean_elpd: float
    n_posterior_samples: int = 0


@dataclass(frozen=True)
class HeldoutLPDSResult:
    """Chronological held-out predictive score for one fitted VB model."""

    train_result: VariationalResult
    train_score: PredictiveLogScore
    test_score: PredictiveLogScore
    train_indices: np.ndarray
    test_indices: np.ndarray
    standardization: ModelStandardization


def predictive_log_score(
    result: VariationalResult,
    dataset: Dataset,
    setting: ModelSetting,
    standardization: ModelStandardization | None = None,
    n_samples: int = 0,
    seed: int = 0,
) -> PredictiveLogScore:
    """Compute held-out ELPD from posterior-sampled predictive densities."""

    inputs = _prepare_model_inputs(dataset, setting, standardization)

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if result.posterior is not None and n_samples > 0:
        param_samples = _sample_model_posterior(result.posterior, setting, seed, n_samples)

        def sample_pointwise_log_prob(sample_tree):
            return _pointwise_log_prob(_tree_to_model_params(sample_tree, setting), inputs, setting)

        sample_pointwise = vmap(sample_pointwise_log_prob)(param_samples)
        pointwise = np.asarray(logsumexp(sample_pointwise, axis=0) - np.log(n_samples))
        n_posterior_samples = n_samples
    else:
        pointwise = np.asarray(_pointwise_log_prob(result.params, inputs, setting))
        n_posterior_samples = 0

    return PredictiveLogScore(
        pointwise=pointwise,
        elpd=float(np.sum(pointwise)),
        mean_elpd=float(np.mean(pointwise)),
        n_posterior_samples=n_posterior_samples,
    )


def fit_heldout_gaussian_mixture_lpds(
    dataset: Dataset,
    setting: GaussianMixtureSetting,
    fit: FitConfig | None = None,
    test_size: float | int = 0.2,
    n_predictive_samples: int | None = None,
    seed: int | None = None,
) -> HeldoutLPDSResult:
    """Fit a Gaussian mixture and score the final held-out block."""

    return fit_heldout_model_lpds(
        dataset,
        setting,
        fit,
        test_size=test_size,
        n_predictive_samples=n_predictive_samples,
        seed=seed,
    )


def fit_heldout_model_lpds(
    dataset: Dataset,
    setting: ModelSetting,
    fit: FitConfig | None = None,
    test_size: float | int = 0.2,
    n_predictive_samples: int | None = None,
    seed: int | None = None,
) -> HeldoutLPDSResult:
    """Fit on the first block of observations and score the final held-out block."""

    fit = fit or FitConfig()
    if n_predictive_samples is None:
        n_predictive_samples = fit.n_predictive_samples
    if n_predictive_samples < 0:
        raise ValueError("n_predictive_samples must be non-negative")
    if seed is None:
        seed = fit.seed + 10000

    train_idx, test_idx = chronological_holdout_indices(dataset.y.shape[0], test_size)
    train_dataset = subset_dataset(dataset, train_idx)
    test_dataset = subset_dataset(dataset, test_idx)
    standardization = _fit_model_standardization(train_dataset, setting)

    result = fit_variational(train_dataset, setting, fit)
    train_score = predictive_log_score(
        result,
        train_dataset,
        setting,
        standardization,
        n_samples=n_predictive_samples,
        seed=seed,
    )
    test_score = predictive_log_score(
        result,
        test_dataset,
        setting,
        standardization,
        n_samples=n_predictive_samples,
        seed=seed,
    )
    return HeldoutLPDSResult(
        train_result=result,
        train_score=train_score,
        test_score=test_score,
        train_indices=train_idx,
        test_indices=test_idx,
        standardization=standardization,
    )


def _fit_model_standardization(
    dataset: Dataset,
    setting: ModelSetting,
) -> ModelStandardization:
    if isinstance(setting, BetaRegMixtureSetting):
        return fit_betareg_mixture_standardization(dataset, setting)
    if isinstance(setting, (GammaMixtureSetting, GammaRepMixtureSetting)):
        return fit_gamma_mixture_standardization(dataset, setting)
    if isinstance(setting, GaussianMixtureSetting):
        return fit_gaussian_mixture_standardization(dataset, setting)
    if isinstance(setting, (LogNormalMixtureSetting, LogNormalRepMixtureSetting)):
        return fit_lognormal_mixture_standardization(dataset, setting)
    if isinstance(setting, SplitNormalMixtureSetting):
        return fit_splitnormal_mixture_standardization(dataset, setting)
    if isinstance(setting, SplitTMixtureSetting):
        return fit_splitt_mixture_standardization(dataset, setting)
    raise NotImplementedError(f"no standardization for model type {type(setting)!r}")


def _prepare_model_inputs(
    dataset: Dataset,
    setting: ModelSetting,
    standardization: ModelStandardization | None,
) -> Any:
    if isinstance(setting, BetaRegMixtureSetting):
        return prepare_betareg_mixture_inputs(dataset, setting, standardization)
    if isinstance(setting, (GammaMixtureSetting, GammaRepMixtureSetting)):
        return prepare_gamma_mixture_inputs(dataset, setting, standardization)
    if isinstance(setting, GaussianMixtureSetting):
        return prepare_gaussian_mixture_inputs(dataset, setting, standardization)
    if isinstance(setting, (LogNormalMixtureSetting, LogNormalRepMixtureSetting)):
        return prepare_lognormal_mixture_inputs(dataset, setting, standardization)
    if isinstance(setting, SplitNormalMixtureSetting):
        return prepare_splitnormal_mixture_inputs(dataset, setting, standardization)
    if isinstance(setting, SplitTMixtureSetting):
        return prepare_splitt_mixture_inputs(dataset, setting, standardization)
    raise NotImplementedError(f"no input builder for model type {type(setting)!r}")


def _sample_model_posterior(posterior, setting: ModelSetting, seed: int, n_samples: int):
    if isinstance(setting, BetaRegMixtureSetting):
        return sample_betareg_mixture_posterior(posterior, seed, n_samples)
    if isinstance(setting, (GammaMixtureSetting, GammaRepMixtureSetting)):
        return sample_gamma_mixture_posterior(posterior, seed, n_samples)
    if isinstance(setting, GaussianMixtureSetting):
        return sample_gaussian_mixture_posterior(posterior, seed, n_samples)
    if isinstance(setting, (LogNormalMixtureSetting, LogNormalRepMixtureSetting)):
        return sample_lognormal_mixture_posterior(posterior, seed, n_samples)
    if isinstance(setting, SplitNormalMixtureSetting):
        return sample_splitnormal_mixture_posterior(posterior, seed, n_samples)
    if isinstance(setting, SplitTMixtureSetting):
        return sample_splitt_mixture_posterior(posterior, seed, n_samples)
    raise NotImplementedError(f"no posterior sampler for model type {type(setting)!r}")


def _tree_to_model_params(sample_tree, setting: ModelSetting):
    if isinstance(setting, BetaRegMixtureSetting):
        return tree_to_betareg_params(sample_tree)
    if isinstance(setting, (GammaMixtureSetting, GammaRepMixtureSetting)):
        return tree_to_gamma_params(sample_tree)
    if isinstance(setting, GaussianMixtureSetting):
        return tree_to_gaussian_params(sample_tree)
    if isinstance(setting, (LogNormalMixtureSetting, LogNormalRepMixtureSetting)):
        return tree_to_lognormal_params(sample_tree)
    if isinstance(setting, SplitNormalMixtureSetting):
        return tree_to_splitnormal_params(sample_tree)
    if isinstance(setting, SplitTMixtureSetting):
        return tree_to_splitt_params(sample_tree)
    raise NotImplementedError(f"no tree conversion for model type {type(setting)!r}")


def _pointwise_log_prob(params, inputs, setting: ModelSetting):
    if isinstance(setting, BetaRegMixtureSetting):
        return betareg_log_prob_observations(
            params,
            jnp.asarray(inputs.y),
            jnp.asarray(inputs.X_mean),
            jnp.asarray(inputs.X_dispersion),
            jnp.asarray(inputs.Z),
        )
    if isinstance(setting, (GammaMixtureSetting, GammaRepMixtureSetting)):
        return gamma_log_prob_observations(
            params,
            jnp.asarray(inputs.y),
            jnp.asarray(inputs.X_mean),
            jnp.asarray(inputs.X_variance),
            jnp.asarray(inputs.Z),
            parameterization=setting.parameterization,
        )
    if isinstance(setting, GaussianMixtureSetting):
        return gaussian_log_prob_observations(
            params,
            jnp.asarray(inputs.y),
            jnp.asarray(inputs.X_mean),
            jnp.asarray(inputs.X_variance),
            jnp.asarray(inputs.Z),
        )
    if isinstance(setting, (LogNormalMixtureSetting, LogNormalRepMixtureSetting)):
        return lognormal_log_prob_observations(
            params,
            jnp.asarray(inputs.y),
            jnp.asarray(inputs.X_mean),
            jnp.asarray(inputs.X_scale),
            jnp.asarray(inputs.Z),
            parameterization=setting.parameterization,
        )
    if isinstance(setting, SplitNormalMixtureSetting):
        return splitnormal_log_prob_observations(
            params,
            jnp.asarray(inputs.y),
            jnp.asarray(inputs.X_mean),
            jnp.asarray(inputs.X_scale),
            jnp.asarray(inputs.X_skewness),
            jnp.asarray(inputs.Z),
        )
    if isinstance(setting, SplitTMixtureSetting):
        return splitt_log_prob_observations(
            params,
            jnp.asarray(inputs.y),
            jnp.asarray(inputs.X_mean),
            jnp.asarray(inputs.X_df),
            jnp.asarray(inputs.X_scale),
            jnp.asarray(inputs.X_skewness),
            jnp.asarray(inputs.Z),
        )
    raise NotImplementedError(f"no pointwise log probability for model type {type(setting)!r}")


def chronological_holdout_indices(
    n_obs: int,
    test_size: float | int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test indices with the last observations held out."""

    if n_obs < 2:
        raise ValueError("at least two observations are required for a holdout split")
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("float test_size must be between 0 and 1")
        n_test = int(np.ceil(n_obs * test_size))
    else:
        n_test = int(test_size)

    if n_test < 1 or n_test >= n_obs:
        raise ValueError("test_size must leave at least one train and one test observation")

    split = n_obs - n_test
    indices = np.arange(n_obs)
    return indices[:split], indices[split:]
