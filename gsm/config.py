"""Small configuration objects for the migration skeleton."""

from dataclasses import dataclass, field
import math


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
    data_file_name: str = "simpleUnivDens"
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
class SplitTMixtureSetting:
    """Minimal Python version of MATLAB's asymmetric Student-t GSM setting.

    The MATLAB model is named ``AsymStudT``. The Python migration uses
    ``SplitT`` because the density is a two-piece, or split, Student-t kernel.
    """

    model_name: str = "SplitT"
    data_file_name: str = "sp500_1990-2009_calendar.csv"
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
    data_file_name: str = "sp500_1990-2009_calendar.csv"
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


def hetero_gaussian_setting(n_components: int = 2) -> GaussianMixtureSetting:
    """Return the default HeteroGauss setting with an adjustable component count."""

    return GaussianMixtureSetting(n_components=n_components)


def sp500_gaussian_mixture_setting(n_components: int = 3) -> GaussianMixtureSetting:
    """Gaussian mixture setting for ``sp500_1990-2009_calendar.csv``.

    This follows the S&P 500 asymmetric Student-t scripts where the mean uses
    only the intercept while scale-like features and gating use the first ten
    covariates: the inserted intercept plus lagged-return/volatility predictors,
    excluding ``Time``.
    """

    return GaussianMixtureSetting(
        data_file_name="sp500_1990-2009_calendar.csv",
        covs=((0,), tuple(range(10))),
        covs_mix=tuple(range(10)),
        add_constant=True,
        n_components=n_components,
        standardize=2,
    )


def sp500_splitt_mixture_setting(n_components: int = 3) -> SplitTMixtureSetting:
    """Split-t mixture setting for ``sp500_1990-2009_calendar.csv``."""

    return SplitTMixtureSetting(n_components=n_components)


def sp500_splitnormal_mixture_setting(n_components: int = 1) -> SplitNormalMixtureSetting:
    """Split-normal mixture setting for ``sp500_1990-2009_calendar.csv``."""

    return SplitNormalMixtureSetting(n_components=n_components)
