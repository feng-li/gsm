"""Public variational inference facade.

Model-specific implementations live under :mod:`gsm.vi`. This module keeps the
existing import path stable while the migration grows.
"""

from .config import (
    BetaRegMixtureSetting,
    FitConfig,
    GammaMixtureSetting,
    GammaRepMixtureSetting,
    GaussianMixtureSetting,
    LogNormalMixtureSetting,
    LogNormalRepMixtureSetting,
    ModelConfig,
    SplitNormalMixtureSetting,
    SplitTMixtureSetting,
)
from .data import Dataset
from .vi.common import VariationalResult
from .vi.betareg import (
    BetaRegMixtureInputs,
    BetaRegMixturePosterior,
    BetaRegMixtureStandardization,
    betareg_mixture_elbo,
    betareg_mixture_variational_elbo,
    fit_betareg_mixture_standardization,
    fit_betareg_mixture_vb,
    initialize_betareg_mixture_params,
    initialize_betareg_mixture_variational_params,
    prepare_betareg_mixture_inputs,
    sample_betareg_mixture_posterior,
    tree_to_betareg_params,
    tree_to_betareg_posterior,
)
from .vi.gaussian import (
    GaussianMixtureInputs,
    GaussianMixturePosterior,
    GaussianMixtureStandardization,
    fit_gaussian_mixture_standardization,
    fit_gaussian_mixture_vb,
    gaussian_mixture_elbo,
    gaussian_mixture_variational_elbo,
    initialize_gaussian_mixture_params,
    initialize_gaussian_mixture_variational_params,
    prepare_gaussian_mixture_inputs,
    sample_gaussian_mixture_posterior,
    tree_to_gaussian_params,
    tree_to_gaussian_posterior,
)
from .vi.gamma import (
    GammaMixtureInputs,
    GammaMixturePosterior,
    GammaMixtureStandardization,
    fit_gamma_mixture_standardization,
    fit_gamma_mixture_vb,
    gamma_mixture_elbo,
    gamma_mixture_variational_elbo,
    initialize_gamma_mixture_params,
    initialize_gamma_mixture_variational_params,
    prepare_gamma_mixture_inputs,
    sample_gamma_mixture_posterior,
    tree_to_gamma_params,
    tree_to_gamma_posterior,
)
from .vi.lognormal import (
    LogNormalMixtureInputs,
    LogNormalMixturePosterior,
    LogNormalMixtureStandardization,
    fit_lognormal_mixture_standardization,
    fit_lognormal_mixture_vb,
    initialize_lognormal_mixture_params,
    initialize_lognormal_mixture_variational_params,
    lognormal_mixture_elbo,
    lognormal_mixture_variational_elbo,
    prepare_lognormal_mixture_inputs,
    sample_lognormal_mixture_posterior,
    tree_to_lognormal_params,
    tree_to_lognormal_posterior,
)
from .vi.splitnormal import (
    SplitNormalMixtureInputs,
    SplitNormalMixturePosterior,
    SplitNormalMixtureStandardization,
    fit_splitnormal_mixture_standardization,
    fit_splitnormal_mixture_vb,
    initialize_splitnormal_mixture_params,
    initialize_splitnormal_mixture_variational_params,
    prepare_splitnormal_mixture_inputs,
    sample_splitnormal_mixture_posterior,
    splitnormal_mixture_elbo,
    splitnormal_mixture_variational_elbo,
    tree_to_splitnormal_params,
    tree_to_splitnormal_posterior,
)
from .vi.splitt import (
    SplitTMixtureInputs,
    SplitTMixturePosterior,
    SplitTMixtureStandardization,
    fit_splitt_mixture_standardization,
    fit_splitt_mixture_vb,
    initialize_splitt_mixture_params,
    initialize_splitt_mixture_variational_params,
    prepare_splitt_mixture_inputs,
    sample_splitt_mixture_posterior,
    splitt_mixture_elbo,
    splitt_mixture_variational_elbo,
    tree_to_splitt_params,
    tree_to_splitt_posterior,
)


def fit_variational(
    dataset: Dataset,
    model: (
        ModelConfig
        | BetaRegMixtureSetting
        | GammaMixtureSetting
        | GammaRepMixtureSetting
        | GaussianMixtureSetting
        | LogNormalMixtureSetting
        | LogNormalRepMixtureSetting
        | SplitNormalMixtureSetting
        | SplitTMixtureSetting
    ),
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Dispatch to the available variational inference implementation."""

    if isinstance(model, BetaRegMixtureSetting):
        return fit_betareg_mixture_vb(dataset, model, fit)
    if isinstance(model, (GammaMixtureSetting, GammaRepMixtureSetting)):
        return fit_gamma_mixture_vb(dataset, model, fit)
    if isinstance(model, GaussianMixtureSetting):
        return fit_gaussian_mixture_vb(dataset, model, fit)
    if isinstance(model, (LogNormalMixtureSetting, LogNormalRepMixtureSetting)):
        return fit_lognormal_mixture_vb(dataset, model, fit)
    if isinstance(model, SplitNormalMixtureSetting):
        return fit_splitnormal_mixture_vb(dataset, model, fit)
    if isinstance(model, SplitTMixtureSetting):
        return fit_splitt_mixture_vb(dataset, model, fit)
    raise NotImplementedError(f"no variational implementation for model type {type(model)!r}")
