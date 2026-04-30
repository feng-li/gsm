"""Predictive scoring utilities for VB fits."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import vmap
from jax.nn import logsumexp
import numpy as np

from .config import FitConfig, GaussianMixtureSetting
from .data import Dataset, subset_dataset
from .models.gaussian_mixture import log_prob_observations
from .variational import (
    GaussianMixtureStandardization,
    VariationalResult,
    fit_gaussian_mixture_standardization,
    fit_variational,
    prepare_gaussian_mixture_inputs,
    sample_gaussian_mixture_posterior,
    tree_to_gaussian_params,
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
    standardization: GaussianMixtureStandardization


def predictive_log_score(
    result: VariationalResult,
    dataset: Dataset,
    setting: GaussianMixtureSetting,
    standardization: GaussianMixtureStandardization | None = None,
    n_samples: int = 0,
    seed: int = 0,
) -> PredictiveLogScore:
    """Compute held-out ELPD from posterior-sampled predictive densities."""

    inputs = prepare_gaussian_mixture_inputs(dataset, setting, standardization)

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if result.posterior is not None and n_samples > 0:
        param_samples = sample_gaussian_mixture_posterior(
            result.posterior,
            seed=seed,
            n_samples=n_samples,
        )

        def sample_pointwise_log_prob(sample_tree):
            return log_prob_observations(
                tree_to_gaussian_params(sample_tree),
                jnp.asarray(inputs.y),
                jnp.asarray(inputs.X_mean),
                jnp.asarray(inputs.X_variance),
                jnp.asarray(inputs.Z),
            )

        sample_pointwise = vmap(sample_pointwise_log_prob)(param_samples)
        pointwise = np.asarray(logsumexp(sample_pointwise, axis=0) - np.log(n_samples))
        n_posterior_samples = n_samples
    else:
        pointwise = np.asarray(
            log_prob_observations(
                result.params,
                jnp.asarray(inputs.y),
                jnp.asarray(inputs.X_mean),
                jnp.asarray(inputs.X_variance),
                jnp.asarray(inputs.Z),
            )
        )
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
    standardization = fit_gaussian_mixture_standardization(train_dataset, setting)

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
