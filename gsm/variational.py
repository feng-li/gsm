"""Public variational inference facade.

Model-specific implementations live under :mod:`gsm.vi`. This module keeps the
existing import path stable while the migration grows.
"""

from .config import (
    BetaBinMixtureSetting,
    BetaRegMixtureSetting,
    BinomialMixtureSetting,
    FitConfig,
    GammaMixtureSetting,
    GammaRepMixtureSetting,
    GenPoissonAltMixtureSetting,
    GenPoissonMixtureSetting,
    GaussianMixtureSetting,
    LogNormalMixtureSetting,
    LogNormalRepMixtureSetting,
    ModelConfig,
    NegBinMixtureSetting,
    PoissonMixtureSetting,
    SplitNormalMixtureSetting,
    SplitTMixtureSetting,
    StudentTMixtureSetting,
)
from .data import Dataset
from .model_registry import get_model_adapter
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
from .vi.betabinomial import (
    BetaBinMixtureInputs,
    BetaBinMixturePosterior,
    BetaBinMixtureStandardization,
    betabin_mixture_elbo,
    betabin_mixture_variational_elbo,
    fit_betabin_mixture_standardization,
    fit_betabin_mixture_vb,
    initialize_betabin_mixture_params,
    initialize_betabin_mixture_variational_params,
    prepare_betabin_mixture_inputs,
    sample_betabin_mixture_posterior,
    tree_to_betabin_params,
    tree_to_betabin_posterior,
)
from .vi.binomial import (
    BinomialMixtureInputs,
    BinomialMixturePosterior,
    BinomialMixtureStandardization,
    binomial_mixture_elbo,
    binomial_mixture_variational_elbo,
    fit_binomial_mixture_standardization,
    fit_binomial_mixture_vb,
    initialize_binomial_mixture_params,
    initialize_binomial_mixture_variational_params,
    prepare_binomial_mixture_inputs,
    sample_binomial_mixture_posterior,
    tree_to_binomial_params,
    tree_to_binomial_posterior,
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
from .vi.genpoisson import (
    GenPoissonMixtureInputs,
    GenPoissonMixturePosterior,
    GenPoissonMixtureStandardization,
    fit_genpoisson_mixture_standardization,
    fit_genpoisson_mixture_vb,
    genpoisson_mixture_elbo,
    genpoisson_mixture_variational_elbo,
    initialize_genpoisson_mixture_params,
    initialize_genpoisson_mixture_variational_params,
    prepare_genpoisson_mixture_inputs,
    sample_genpoisson_mixture_posterior,
    tree_to_genpoisson_params,
    tree_to_genpoisson_posterior,
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
from .vi.negbin import (
    NegBinMixtureInputs,
    NegBinMixturePosterior,
    NegBinMixtureStandardization,
    fit_negbin_mixture_standardization,
    fit_negbin_mixture_vb,
    initialize_negbin_mixture_params,
    initialize_negbin_mixture_variational_params,
    negbin_mixture_elbo,
    negbin_mixture_variational_elbo,
    prepare_negbin_mixture_inputs,
    sample_negbin_mixture_posterior,
    tree_to_negbin_params,
    tree_to_negbin_posterior,
)
from .vi.poisson import (
    PoissonMixtureInputs,
    PoissonMixturePosterior,
    PoissonMixtureStandardization,
    fit_poisson_mixture_standardization,
    fit_poisson_mixture_vb,
    initialize_poisson_mixture_params,
    initialize_poisson_mixture_variational_params,
    poisson_mixture_elbo,
    poisson_mixture_variational_elbo,
    prepare_poisson_mixture_inputs,
    sample_poisson_mixture_posterior,
    tree_to_poisson_params,
    tree_to_poisson_posterior,
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
from .vi.studentt import (
    StudentTMixtureInputs,
    StudentTMixturePosterior,
    StudentTMixtureStandardization,
    fit_studentt_mixture_standardization,
    fit_studentt_mixture_vb,
    initialize_studentt_mixture_params,
    initialize_studentt_mixture_variational_params,
    prepare_studentt_mixture_inputs,
    sample_studentt_mixture_posterior,
    studentt_mixture_elbo,
    studentt_mixture_variational_elbo,
    tree_to_studentt_params,
    tree_to_studentt_posterior,
)


def fit_variational(
    dataset: Dataset,
    model: (
        ModelConfig
        | BetaBinMixtureSetting
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
    ),
    fit: FitConfig | None = None,
) -> VariationalResult:
    """Dispatch to the available variational inference implementation."""

    return get_model_adapter(model).fit_vb(dataset, model, fit)
