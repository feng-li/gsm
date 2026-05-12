"""Small configuration objects for the migration skeleton."""

from dataclasses import dataclass, field
import math
from typing import Literal


@dataclass
class ModelConfig:
    """Model choices that used to live in MATLAB model-setting files."""

    model_name: str
    feature_names: tuple[str, ...]
    link_types: tuple[str | float, ...]
    n_components: int = 1
    parametrization: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.link_types):
            raise ValueError("feature_names and link_types must have the same length")
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if not self.parametrization:
            self.parametrization = tuple(0 for _ in self.feature_names)
        if len(self.parametrization) != len(self.feature_names):
            raise ValueError("parametrization must have one entry per feature")


@dataclass
class FitConfig:
    """Controls for the first variational Bayes implementation."""

    seed: int = 10000
    max_iter: int = 1000
    learning_rate: float = 1e-2
    n_restarts: int = 3
    tol: float = 1e-6
    coefficient_prior_scale: float = 10.0
    use_ard: bool = False
    ard_shape: float = 1e-2
    ard_rate: float = 1e-2
    n_elbo_samples: int = 8
    n_predictive_samples: int = 100
    posterior_init_log_std: float = -5.0


@dataclass(frozen=True)
class GaussianMixtureSetting:
    """Minimal Python version of MATLAB's HeteroGauss model setting.

    Indices are converted from MATLAB's 1-based convention to Python's 0-based
    convention. The default mirrors the simple intercept-only Gaussian mixture
    setting in ``MatlabCode/models/heteroGauss/HeteroGauss.m``.
    """

    model_name: str = "HeteroGauss"
    feature_names: tuple[str, str] = ("Mean", "Variance")
    link_types: tuple[str, str] = ("identity", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...]] = ((0,), (0,))
    covs_mix: tuple[int, ...] = (0,)
    on_trial: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    on_trial_mix: tuple[int, ...] = ()
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, float] = (0.0, 1.0)
    prior_std_feat: tuple[float, float] = (10.0, 10.0)
    prior_shrink: tuple[float | str, float | str] = (100.0, 100.0)
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, float] = (0.5, 0.5)
    prior_inclusion_mix: float = 0.5


@dataclass(frozen=True)
class BetaRegMixtureSetting:
    """Minimal Python version of the Rajan ``BetaReg`` GSM setting."""

    model_name: str = "BetaReg"
    feature_names: tuple[str, str] = ("Mean", "Disp")
    link_types: tuple[str, str] = ("logit", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(range(5)),
        tuple(range(5)),
    )
    covs_mix: tuple[int, ...] = tuple(range(5))
    on_trial: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(range(1, 5)),
        tuple(range(1, 5)),
    )
    on_trial_mix: tuple[int, ...] = tuple(range(1, 5))
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 2
    prior_mean_feat: tuple[float, float] = (0.367, 2.2649)
    prior_std_feat: tuple[float, float] = (0.1, 10.0)
    prior_shrink: tuple[float | str, float | str] = ("unitinfo", "unitinfo")
    prior_shrink_mix: float | str = 100.0
    prior_inclusion: tuple[float, float] = (0.5, 0.5)
    prior_inclusion_mix: float = 0.5


@dataclass(frozen=True)
class BinomialMixtureSetting:
    """Minimal Python version of MATLAB's ``Bin`` GSM setting.

    The response convention is a two-column matrix: ``successes, trials``.
    """

    model_name: str = "Bin"
    feature_names: tuple[str] = ("Mean",)
    link_types: tuple[str] = ("logit",)
    covs: tuple[tuple[int, ...]] = (tuple(range(7)),)
    covs_mix: tuple[int, ...] = tuple(range(7))
    on_trial: tuple[tuple[int, ...]] = (tuple(range(1, 7)),)
    on_trial_mix: tuple[int, ...] = tuple(range(1, 7))
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float] = (0.5,)
    prior_std_feat: tuple[float] = (10.0,)
    prior_shrink: tuple[float | str] = ("unitinfo",)
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float] = (0.5,)
    prior_inclusion_mix: float = 0.5


@dataclass(frozen=True)
class BetaBinMixtureSetting:
    """Minimal Python version of MATLAB's ``BetaBin`` GSM setting.

    The response convention is a two-column matrix: ``successes, trials``.
    """

    model_name: str = "BetaBin"
    feature_names: tuple[str, str] = ("Mean", "Disp")
    link_types: tuple[str, str] = ("logit", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(range(8)),
        tuple(range(8)),
    )
    covs_mix: tuple[int, ...] = tuple(range(7))
    on_trial: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(range(1, 7)),
        tuple(range(1, 7)),
    )
    on_trial_mix: tuple[int, ...] = tuple(range(1, 7))
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, float] = (0.5, 1.0)
    prior_std_feat: tuple[float, float] = (10.0, 10.0)
    prior_shrink: tuple[float | str, float | str] = ("unitinfo", "unitinfo")
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, float] = (0.5, 0.5)
    prior_inclusion_mix: float = 0.5


@dataclass(frozen=True)
class PoissonMixtureSetting:
    """Minimal Python version of the mdvisits ``Pois`` GSM setting."""

    model_name: str = "Pois"
    feature_names: tuple[str] = ("Mean",)
    link_types: tuple[str] = ("log",)
    covs: tuple[tuple[int, ...]] = (tuple(range(8)),)
    covs_mix: tuple[int, ...] = tuple(range(8))
    on_trial: tuple[tuple[int, ...]] = (tuple(range(1, 8)),)
    on_trial_mix: tuple[int, ...] = tuple(range(1, 8))
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float] = (2.5891,)
    prior_std_feat: tuple[float] = (10.0,)
    prior_shrink: tuple[float | str] = ("UnitInfo",)
    prior_shrink_mix: float | str = 100.0
    prior_inclusion: tuple[float] = (0.5,)
    prior_inclusion_mix: float = 0.5


@dataclass(frozen=True)
class NegBinMixtureSetting:
    """Minimal Python version of the mdvisits ``NegBin`` GSM setting."""

    model_name: str = "NegBin"
    feature_names: tuple[str, str] = ("Mean", "Disp")
    link_types: tuple[str, str] = ("log", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(range(8)),
        tuple(range(8)),
    )
    covs_mix: tuple[int, ...] = tuple(range(8))
    on_trial: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(range(1, 8)),
        tuple(range(1, 8)),
    )
    on_trial_mix: tuple[int, ...] = tuple(range(1, 8))
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, float] = (2.5891, 0.4951)
    prior_std_feat: tuple[float, float] = (10.0, 10.0)
    prior_shrink: tuple[float | str, float | str] = ("unitinfo", "unitinfo")
    prior_shrink_mix: float | str = 100.0
    prior_inclusion: tuple[float, float] = (0.5, 0.5)
    prior_inclusion_mix: float = 0.5


@dataclass(frozen=True)
class GenPoissonMixtureSetting:
    """Minimal Python version of the mdvisits ``GenPois`` GSM setting."""

    model_name: str = "GenPois"
    feature_names: tuple[str, str] = ("Mean", "Disp")
    link_types: tuple[str, str] = ("log", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(range(8)),
        tuple(range(8)),
    )
    covs_mix: tuple[int, ...] = tuple(range(8))
    on_trial: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(range(1, 8)),
        tuple(range(1, 8)),
    )
    on_trial_mix: tuple[int, ...] = tuple(range(1, 8))
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, float] = (2.5891, 0.5778)
    prior_std_feat: tuple[float, float] = (10.0, 1.0)
    prior_shrink: tuple[float | str, float | str] = ("unitinfo", "unitinfo")
    prior_shrink_mix: float | str = 100.0
    prior_inclusion: tuple[float, float] = (0.5, 0.5)
    prior_inclusion_mix: float = 0.5
    parameterization: Literal["standard", "alternative"] = "standard"


@dataclass(frozen=True)
class GenPoissonAltMixtureSetting:
    """Minimal Python version of MATLAB's ``GenPoisAlt`` GSM setting."""

    model_name: str = "GenPoisAlt"
    feature_names: tuple[str, str] = ("Mean", "Disp")
    link_types: tuple[str, str] = ("log", "log1")
    covs: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(range(7)),
        tuple(range(7)),
    )
    covs_mix: tuple[int, ...] = tuple(range(7))
    on_trial: tuple[tuple[int, ...], tuple[int, ...]] = (
        tuple(range(1, 7)),
        tuple(range(1, 7)),
    )
    on_trial_mix: tuple[int, ...] = tuple(range(1, 7))
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, float] = (5.0, 1.0)
    prior_std_feat: tuple[float, float] = (10.0, 1.0)
    prior_shrink: tuple[float | str, float | str] = (10.0, 10.0)
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, float] = (0.5, 0.5)
    prior_inclusion_mix: float = 0.5
    parameterization: Literal["standard", "alternative"] = "alternative"


@dataclass(frozen=True)
class LogNormalMixtureSetting:
    """Minimal Python version of MATLAB's ``LogNorm`` GSM setting."""

    model_name: str = "LogNorm"
    feature_names: tuple[str, str] = ("Mean", "Scale")
    link_types: tuple[str, str] = ("log", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...]] = ((0,), (0,))
    covs_mix: tuple[int, ...] = (0,)
    on_trial: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    on_trial_mix: tuple[int, ...] = ()
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, float] = (math.log(361.0), math.log(176.377))
    prior_std_feat: tuple[float, float] = (math.log(100.0), math.log(100.0))
    prior_shrink: tuple[float | str, float | str] = ("UnitInfo", "UnitInfo")
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, float] = (0.5, 0.5)
    prior_inclusion_mix: float = 0.5
    parameterization: Literal["standard", "response"] = "standard"


@dataclass(frozen=True)
class LogNormalRepMixtureSetting:
    """Minimal Python version of MATLAB's ``LogNormRep`` GSM setting."""

    model_name: str = "LogNormRep"
    feature_names: tuple[str, str] = ("Mean", "Scale")
    link_types: tuple[str, str] = ("log", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...]] = ((0,), (0,))
    covs_mix: tuple[int, ...] = (0,)
    on_trial: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    on_trial_mix: tuple[int, ...] = ()
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, float] = (361.0, 176.377)
    prior_std_feat: tuple[float, float] = (100.0, 100.0)
    prior_shrink: tuple[float | str, float | str] = ("UnitInfo", "UnitInfo")
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, float] = (0.5, 0.5)
    prior_inclusion_mix: float = 0.5
    parameterization: Literal["standard", "response"] = "response"


@dataclass(frozen=True)
class GammaMixtureSetting:
    """Minimal Python version of MATLAB's ``Gamma`` GSM setting."""

    model_name: str = "Gamma"
    feature_names: tuple[str, str] = ("Mean", "Variance")
    link_types: tuple[str, str] = ("log", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...]] = ((0,), (0,))
    covs_mix: tuple[int, ...] = (0,)
    on_trial: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    on_trial_mix: tuple[int, ...] = ()
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, float] = (361.0, 176.377**2)
    prior_std_feat: tuple[float, float] = (100.0, 100.0**2)
    prior_shrink: tuple[float | str, float | str] = ("UnitInfo", "UnitInfo")
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, float] = (0.5, 0.5)
    prior_inclusion_mix: float = 0.5
    parameterization: Literal["mean_variance", "shape_scale"] = "mean_variance"


@dataclass(frozen=True)
class GammaRepMixtureSetting:
    """Response-equivalent Gamma setting using direct shape/scale features."""

    model_name: str = "GammaRep"
    feature_names: tuple[str, str] = ("Shape", "Scale")
    link_types: tuple[str, str] = ("log", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...]] = ((0,), (0,))
    covs_mix: tuple[int, ...] = (0,)
    on_trial: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    on_trial_mix: tuple[int, ...] = ()
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, float] = (361.0**2 / 176.377**2, 176.377**2 / 361.0)
    prior_std_feat: tuple[float, float] = (10.0, 100.0)
    prior_shrink: tuple[float | str, float | str] = ("UnitInfo", "UnitInfo")
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, float] = (0.5, 0.5)
    prior_inclusion_mix: float = 0.5
    parameterization: Literal["mean_variance", "shape_scale"] = "shape_scale"


@dataclass(frozen=True)
class SplitTMixtureSetting:
    """Minimal Python version of MATLAB's asymmetric Student-t GSM setting.

    The MATLAB model is named ``AsymStudT``. The Python migration uses
    ``SplitT`` because the density is a two-piece, or split, Student-t kernel.
    """

    model_name: str = "SplitT"
    feature_names: tuple[str, str, str, str] = ("Mean", "DF", "Scale", "Skewness")
    link_types: tuple[str, str, str, str] = ("identity", "log", "log", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (0,),
        tuple(range(10)),
        tuple(range(10)),
        tuple(range(10)),
    )
    covs_mix: tuple[int, ...] = tuple(range(10))
    on_trial: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (),
        tuple(range(1, 10)),
        tuple(range(1, 10)),
        tuple(range(1, 10)),
    )
    on_trial_mix: tuple[int, ...] = tuple(range(1, 10))
    add_constant: bool = True
    n_components: int = 3
    standardize: int = 2
    prior_mean_feat: tuple[float, float, float, float] = (0.0, 10.0, math.sqrt(0.8), 1.0)
    prior_std_feat: tuple[float, float, float, float] = (10.0, 7.0, 1.0, 1.0)
    prior_shrink: tuple[float | str, float | str, float | str, float | str] = (
        100.0,
        100.0,
        100.0,
        100.0,
    )
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
    prior_inclusion_mix: float = 0.5


@dataclass(frozen=True)
class SplitNormalMixtureSetting:
    """Minimal Python version of MATLAB's asymmetric normal GSM setting."""

    model_name: str = "SplitNormal"
    feature_names: tuple[str, str, str] = ("Mean", "Sigma", "Skewness")
    link_types: tuple[str, str, str] = ("identity", "log", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (0,),
        tuple(range(10)),
        tuple(range(10)),
    )
    covs_mix: tuple[int, ...] = tuple(range(10))
    on_trial: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (),
        tuple(range(1, 10)),
        tuple(range(1, 10)),
    )
    on_trial_mix: tuple[int, ...] = tuple(range(1, 10))
    add_constant: bool = True
    n_components: int = 1
    standardize: int = 2
    prior_mean_feat: tuple[float, float, float] = (0.0, 1.0, math.sqrt(0.8))
    prior_std_feat: tuple[float, float, float] = (10.0, 1.0, 1.0)
    prior_shrink: tuple[float | str, float | str, float | str] = (100.0, 100.0, 100.0)
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, float, float] = (0.5, 0.5, 0.5)
    prior_inclusion_mix: float = 0.5


@dataclass(frozen=True)
class StudentTMixtureSetting:
    """Minimal Python version of MATLAB's symmetric ``studT`` model."""

    model_name: str = "StudT"
    feature_names: tuple[str, str, str] = ("Mean", "DF", "Scale")
    link_types: tuple[str, str, str] = ("identity", "log", "log")
    covs: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (0,),
        (0,),
        (0,),
    )
    covs_mix: tuple[int, ...] = (0,)
    on_trial: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = ((), (), ())
    on_trial_mix: tuple[int, ...] = ()
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, float, float] = (0.0, 10.0, 1.0)
    prior_std_feat: tuple[float, float, float] = (10.0, 7.0, 1.0)
    prior_shrink: tuple[float | str, float | str, float | str] = (
        100.0,
        100.0,
        100.0,
    )
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, float, float] = (0.5, 0.5, 0.5)
    prior_inclusion_mix: float = 0.5


def hetero_gaussian_setting(n_components: int = 2) -> GaussianMixtureSetting:
    """Return the default HeteroGauss setting with an adjustable component count."""

    return GaussianMixtureSetting(n_components=n_components)


def rajan_betareg_mixture_setting(n_components: int = 2) -> BetaRegMixtureSetting:
    """Return MATLAB ``Rajan`` BetaReg defaults with an adjustable component count."""

    return BetaRegMixtureSetting(n_components=n_components)


def binomial_mixture_setting(n_components: int = 2) -> BinomialMixtureSetting:
    """Return MATLAB ``Bin`` defaults with an adjustable component count."""

    return BinomialMixtureSetting(n_components=n_components)


def betabin_mixture_setting(n_components: int = 2) -> BetaBinMixtureSetting:
    """Return MATLAB ``BetaBin`` defaults with an adjustable component count."""

    return BetaBinMixtureSetting(n_components=n_components)


def mdvisits_poisson_mixture_setting(n_components: int = 2) -> PoissonMixtureSetting:
    """Return MATLAB ``mdvisitsPois`` defaults with an adjustable component count."""

    return PoissonMixtureSetting(n_components=n_components)


def mdvisits_negbin_mixture_setting(n_components: int = 2) -> NegBinMixtureSetting:
    """Return MATLAB ``mdvisitsNegBin`` defaults with an adjustable component count."""

    return NegBinMixtureSetting(n_components=n_components)


def mdvisits_genpoisson_mixture_setting(n_components: int = 2) -> GenPoissonMixtureSetting:
    """Return MATLAB ``mdvisitsGenPois`` defaults with an adjustable component count."""

    return GenPoissonMixtureSetting(n_components=n_components)


def genpoisson_alt_mixture_setting(
    n_components: int = 2,
) -> GenPoissonAltMixtureSetting:
    """Return MATLAB ``GenPoisAlt`` defaults with an adjustable component count."""

    return GenPoissonAltMixtureSetting(n_components=n_components)


def lognormal_mixture_setting(n_components: int = 2) -> LogNormalMixtureSetting:
    """Return MATLAB ``LogNorm`` defaults with an adjustable component count."""

    return LogNormalMixtureSetting(n_components=n_components)


def lognormal_rep_mixture_setting(n_components: int = 2) -> LogNormalRepMixtureSetting:
    """Return MATLAB ``LogNormRep`` defaults with an adjustable component count."""

    return LogNormalRepMixtureSetting(n_components=n_components)


def gamma_mixture_setting(n_components: int = 2) -> GammaMixtureSetting:
    """Return MATLAB ``Gamma`` defaults with an adjustable component count."""

    return GammaMixtureSetting(n_components=n_components)


def gamma_rep_mixture_setting(n_components: int = 2) -> GammaRepMixtureSetting:
    """Return direct shape/scale GammaRep defaults with an adjustable component count."""

    return GammaRepMixtureSetting(n_components=n_components)


def studentt_mixture_setting(n_components: int = 2) -> StudentTMixtureSetting:
    """Return MATLAB ``studT`` defaults with an adjustable component count."""

    return StudentTMixtureSetting(n_components=n_components)


def sp500_gaussian_mixture_setting(n_components: int = 3) -> GaussianMixtureSetting:
    """Gaussian mixture setting for the S&P 500 covariate layout.

    This follows the S&P 500 asymmetric Student-t scripts where the mean uses
    only the intercept while scale-like features and gating use the first ten
    covariates: the inserted intercept plus lagged-return/volatility predictors,
    excluding ``Time``.
    """

    return GaussianMixtureSetting(
        covs=((0,), tuple(range(10))),
        covs_mix=tuple(range(10)),
        add_constant=True,
        n_components=n_components,
        standardize=2,
    )


def sp500_splitt_mixture_setting(n_components: int = 3) -> SplitTMixtureSetting:
    """Split-t mixture setting for the S&P 500 covariate layout."""

    return SplitTMixtureSetting(n_components=n_components)


def sp500_splitnormal_mixture_setting(n_components: int = 1) -> SplitNormalMixtureSetting:
    """Split-normal mixture setting for the S&P 500 covariate layout."""

    return SplitNormalMixtureSetting(n_components=n_components)
