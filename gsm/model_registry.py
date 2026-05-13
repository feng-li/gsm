"""Central registry for model-specific GSM operations."""

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from gsm.config import (
    BetaBinMixtureSetting,
    BetaRegMixtureSetting,
    BinomialMixtureSetting,
    GammaMixtureSetting,
    GammaRepMixtureSetting,
    GenPoissonAltMixtureSetting,
    GenPoissonMixtureSetting,
    GaussianMixtureSetting,
    LogNormalMixtureSetting,
    LogNormalRepMixtureSetting,
    NegBinMixtureSetting,
    PoissonMixtureSetting,
    SplitNormalMixtureSetting,
    SplitTMixtureSetting,
    StudentTMixtureSetting,
)
from gsm.models.betabinomial import log_prob_observations as betabin_log_prob_observations
from gsm.models.betareg import log_prob_observations as betareg_log_prob_observations
from gsm.models.binomial import log_prob_observations as binomial_log_prob_observations
from gsm.models.gamma import log_prob_observations as gamma_log_prob_observations
from gsm.models.gaussian import log_prob_observations as gaussian_log_prob_observations
from gsm.models.genpoisson import log_prob_observations as genpoisson_log_prob_observations
from gsm.models.lognormal import log_prob_observations as lognormal_log_prob_observations
from gsm.models.negbin import log_prob_observations as negbin_log_prob_observations
from gsm.models.poisson import log_prob_observations as poisson_log_prob_observations
from gsm.models.splitnormal import log_prob_observations as splitnormal_log_prob_observations
from gsm.models.splitt import log_prob_observations as splitt_log_prob_observations
from gsm.models.studentt import log_prob_observations as studentt_log_prob_observations
from gsm.vi.betabinomial import (
    BetaBinMixtureStandardization,
    fit_betabin_mixture_standardization,
    fit_betabin_mixture_vb,
    prepare_betabin_mixture_inputs,
    sample_betabin_mixture_posterior,
    tree_to_betabin_params,
)
from gsm.vi.betareg import (
    BetaRegMixtureStandardization,
    fit_betareg_mixture_standardization,
    fit_betareg_mixture_vb,
    prepare_betareg_mixture_inputs,
    sample_betareg_mixture_posterior,
    tree_to_betareg_params,
)
from gsm.vi.binomial import (
    BinomialMixtureStandardization,
    fit_binomial_mixture_standardization,
    fit_binomial_mixture_vb,
    prepare_binomial_mixture_inputs,
    sample_binomial_mixture_posterior,
    tree_to_binomial_params,
)
from gsm.vi.gamma import (
    GammaMixtureStandardization,
    fit_gamma_mixture_standardization,
    fit_gamma_mixture_vb,
    prepare_gamma_mixture_inputs,
    sample_gamma_mixture_posterior,
    tree_to_gamma_params,
)
from gsm.vi.gaussian import (
    GaussianMixtureStandardization,
    fit_gaussian_mixture_standardization,
    fit_gaussian_mixture_vb,
    prepare_gaussian_mixture_inputs,
    sample_gaussian_mixture_posterior,
    tree_to_gaussian_params,
)
from gsm.vi.genpoisson import (
    GenPoissonMixtureStandardization,
    fit_genpoisson_mixture_standardization,
    fit_genpoisson_mixture_vb,
    prepare_genpoisson_mixture_inputs,
    sample_genpoisson_mixture_posterior,
    tree_to_genpoisson_params,
)
from gsm.vi.lognormal import (
    LogNormalMixtureStandardization,
    fit_lognormal_mixture_standardization,
    fit_lognormal_mixture_vb,
    prepare_lognormal_mixture_inputs,
    sample_lognormal_mixture_posterior,
    tree_to_lognormal_params,
)
from gsm.vi.negbin import (
    NegBinMixtureStandardization,
    fit_negbin_mixture_standardization,
    fit_negbin_mixture_vb,
    prepare_negbin_mixture_inputs,
    sample_negbin_mixture_posterior,
    tree_to_negbin_params,
)
from gsm.vi.poisson import (
    PoissonMixtureStandardization,
    fit_poisson_mixture_standardization,
    fit_poisson_mixture_vb,
    prepare_poisson_mixture_inputs,
    sample_poisson_mixture_posterior,
    tree_to_poisson_params,
)
from gsm.vi.splitnormal import (
    SplitNormalMixtureStandardization,
    fit_splitnormal_mixture_standardization,
    fit_splitnormal_mixture_vb,
    prepare_splitnormal_mixture_inputs,
    sample_splitnormal_mixture_posterior,
    tree_to_splitnormal_params,
)
from gsm.vi.splitt import (
    SplitTMixtureStandardization,
    fit_splitt_mixture_standardization,
    fit_splitt_mixture_vb,
    prepare_splitt_mixture_inputs,
    sample_splitt_mixture_posterior,
    tree_to_splitt_params,
)
from gsm.vi.studentt import (
    StudentTMixtureStandardization,
    fit_studentt_mixture_standardization,
    fit_studentt_mixture_vb,
    prepare_studentt_mixture_inputs,
    sample_studentt_mixture_posterior,
    tree_to_studentt_params,
)


ModelSetting = (
    BetaBinMixtureSetting
    | BetaRegMixtureSetting
    | BinomialMixtureSetting
    | GammaMixtureSetting
    | GammaRepMixtureSetting
    | GenPoissonAltMixtureSetting
    | GenPoissonMixtureSetting
    | GaussianMixtureSetting
    | LogNormalMixtureSetting
    | LogNormalRepMixtureSetting
    | NegBinMixtureSetting
    | PoissonMixtureSetting
    | SplitNormalMixtureSetting
    | SplitTMixtureSetting
    | StudentTMixtureSetting
)
ModelStandardization = (
    BetaBinMixtureStandardization
    | BetaRegMixtureStandardization
    | BinomialMixtureStandardization
    | GammaMixtureStandardization
    | GenPoissonMixtureStandardization
    | GaussianMixtureStandardization
    | LogNormalMixtureStandardization
    | NegBinMixtureStandardization
    | PoissonMixtureStandardization
    | SplitNormalMixtureStandardization
    | SplitTMixtureStandardization
    | StudentTMixtureStandardization
)


@dataclass(frozen=True)
class ModelAdapter:
    setting_types: tuple[type, ...]
    fit_vb: Callable
    fit_standardization: Callable
    prepare_inputs: Callable
    sample_posterior: Callable
    tree_to_params: Callable
    pointwise_log_prob: Callable


def get_model_adapter(setting: ModelSetting) -> ModelAdapter:
    """Return the registered adapter for a model setting."""

    for adapter in MODEL_ADAPTERS:
        if isinstance(setting, adapter.setting_types):
            return adapter
    raise NotImplementedError(f"no model adapter for model type {type(setting)!r}")


def _betabin_pointwise(params, inputs, setting):
    return betabin_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.X_dispersion),
        jnp.asarray(inputs.Z),
    )


def _betareg_pointwise(params, inputs, setting):
    return betareg_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.X_dispersion),
        jnp.asarray(inputs.Z),
    )


def _binomial_pointwise(params, inputs, setting):
    return binomial_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.Z),
    )


def _gamma_pointwise(params, inputs, setting):
    return gamma_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.X_variance),
        jnp.asarray(inputs.Z),
        parameterization=setting.parameterization,
    )


def _genpoisson_pointwise(params, inputs, setting):
    return genpoisson_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.X_dispersion),
        jnp.asarray(inputs.Z),
        parameterization=setting.parameterization,
    )


def _gaussian_pointwise(params, inputs, setting):
    return gaussian_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.X_variance),
        jnp.asarray(inputs.Z),
    )


def _lognormal_pointwise(params, inputs, setting):
    return lognormal_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.X_scale),
        jnp.asarray(inputs.Z),
        parameterization=setting.parameterization,
    )


def _negbin_pointwise(params, inputs, setting):
    return negbin_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.X_dispersion),
        jnp.asarray(inputs.Z),
    )


def _poisson_pointwise(params, inputs, setting):
    return poisson_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.Z),
    )


def _splitnormal_pointwise(params, inputs, setting):
    return splitnormal_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.X_scale),
        jnp.asarray(inputs.X_skewness),
        jnp.asarray(inputs.Z),
    )


def _splitt_pointwise(params, inputs, setting):
    return splitt_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.X_df),
        jnp.asarray(inputs.X_scale),
        jnp.asarray(inputs.X_skewness),
        jnp.asarray(inputs.Z),
    )


def _studentt_pointwise(params, inputs, setting):
    return studentt_log_prob_observations(
        params,
        jnp.asarray(inputs.y),
        jnp.asarray(inputs.X_mean),
        jnp.asarray(inputs.X_df),
        jnp.asarray(inputs.X_scale),
        jnp.asarray(inputs.Z),
    )


MODEL_ADAPTERS: tuple[ModelAdapter, ...] = (
    ModelAdapter(
        setting_types=(BetaBinMixtureSetting,),
        fit_vb=fit_betabin_mixture_vb,
        fit_standardization=fit_betabin_mixture_standardization,
        prepare_inputs=prepare_betabin_mixture_inputs,
        sample_posterior=sample_betabin_mixture_posterior,
        tree_to_params=tree_to_betabin_params,
        pointwise_log_prob=_betabin_pointwise,
    ),
    ModelAdapter(
        setting_types=(BetaRegMixtureSetting,),
        fit_vb=fit_betareg_mixture_vb,
        fit_standardization=fit_betareg_mixture_standardization,
        prepare_inputs=prepare_betareg_mixture_inputs,
        sample_posterior=sample_betareg_mixture_posterior,
        tree_to_params=tree_to_betareg_params,
        pointwise_log_prob=_betareg_pointwise,
    ),
    ModelAdapter(
        setting_types=(BinomialMixtureSetting,),
        fit_vb=fit_binomial_mixture_vb,
        fit_standardization=fit_binomial_mixture_standardization,
        prepare_inputs=prepare_binomial_mixture_inputs,
        sample_posterior=sample_binomial_mixture_posterior,
        tree_to_params=tree_to_binomial_params,
        pointwise_log_prob=_binomial_pointwise,
    ),
    ModelAdapter(
        setting_types=(GammaMixtureSetting, GammaRepMixtureSetting),
        fit_vb=fit_gamma_mixture_vb,
        fit_standardization=fit_gamma_mixture_standardization,
        prepare_inputs=prepare_gamma_mixture_inputs,
        sample_posterior=sample_gamma_mixture_posterior,
        tree_to_params=tree_to_gamma_params,
        pointwise_log_prob=_gamma_pointwise,
    ),
    ModelAdapter(
        setting_types=(GenPoissonMixtureSetting, GenPoissonAltMixtureSetting),
        fit_vb=fit_genpoisson_mixture_vb,
        fit_standardization=fit_genpoisson_mixture_standardization,
        prepare_inputs=prepare_genpoisson_mixture_inputs,
        sample_posterior=sample_genpoisson_mixture_posterior,
        tree_to_params=tree_to_genpoisson_params,
        pointwise_log_prob=_genpoisson_pointwise,
    ),
    ModelAdapter(
        setting_types=(GaussianMixtureSetting,),
        fit_vb=fit_gaussian_mixture_vb,
        fit_standardization=fit_gaussian_mixture_standardization,
        prepare_inputs=prepare_gaussian_mixture_inputs,
        sample_posterior=sample_gaussian_mixture_posterior,
        tree_to_params=tree_to_gaussian_params,
        pointwise_log_prob=_gaussian_pointwise,
    ),
    ModelAdapter(
        setting_types=(LogNormalMixtureSetting, LogNormalRepMixtureSetting),
        fit_vb=fit_lognormal_mixture_vb,
        fit_standardization=fit_lognormal_mixture_standardization,
        prepare_inputs=prepare_lognormal_mixture_inputs,
        sample_posterior=sample_lognormal_mixture_posterior,
        tree_to_params=tree_to_lognormal_params,
        pointwise_log_prob=_lognormal_pointwise,
    ),
    ModelAdapter(
        setting_types=(NegBinMixtureSetting,),
        fit_vb=fit_negbin_mixture_vb,
        fit_standardization=fit_negbin_mixture_standardization,
        prepare_inputs=prepare_negbin_mixture_inputs,
        sample_posterior=sample_negbin_mixture_posterior,
        tree_to_params=tree_to_negbin_params,
        pointwise_log_prob=_negbin_pointwise,
    ),
    ModelAdapter(
        setting_types=(PoissonMixtureSetting,),
        fit_vb=fit_poisson_mixture_vb,
        fit_standardization=fit_poisson_mixture_standardization,
        prepare_inputs=prepare_poisson_mixture_inputs,
        sample_posterior=sample_poisson_mixture_posterior,
        tree_to_params=tree_to_poisson_params,
        pointwise_log_prob=_poisson_pointwise,
    ),
    ModelAdapter(
        setting_types=(SplitNormalMixtureSetting,),
        fit_vb=fit_splitnormal_mixture_vb,
        fit_standardization=fit_splitnormal_mixture_standardization,
        prepare_inputs=prepare_splitnormal_mixture_inputs,
        sample_posterior=sample_splitnormal_mixture_posterior,
        tree_to_params=tree_to_splitnormal_params,
        pointwise_log_prob=_splitnormal_pointwise,
    ),
    ModelAdapter(
        setting_types=(SplitTMixtureSetting,),
        fit_vb=fit_splitt_mixture_vb,
        fit_standardization=fit_splitt_mixture_standardization,
        prepare_inputs=prepare_splitt_mixture_inputs,
        sample_posterior=sample_splitt_mixture_posterior,
        tree_to_params=tree_to_splitt_params,
        pointwise_log_prob=_splitt_pointwise,
    ),
    ModelAdapter(
        setting_types=(StudentTMixtureSetting,),
        fit_vb=fit_studentt_mixture_vb,
        fit_standardization=fit_studentt_mixture_standardization,
        prepare_inputs=prepare_studentt_mixture_inputs,
        sample_posterior=sample_studentt_mixture_posterior,
        tree_to_params=tree_to_studentt_params,
        pointwise_log_prob=_studentt_pointwise,
    ),
)
