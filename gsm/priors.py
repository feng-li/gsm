"""Prior construction mirroring MATLAB ``SetUpPrior.m`` and ``convertPrior.m``."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln


Shrinkage = float | str


@dataclass(frozen=True)
class GaussianCoefficientPrior:
    """Gaussian prior for one coefficient vector, reused across components."""

    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    log_det_covariance: float
    constant_mask: np.ndarray


@dataclass(frozen=True)
class GaussianMixtureCoefficientPriors:
    """Coefficient priors for the current Gaussian-mixture scaffold."""

    mean: GaussianCoefficientPrior
    log_variance: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


@dataclass(frozen=True)
class LogNormalMixtureCoefficientPriors:
    """Coefficient priors for standard and response-scale lognormal mixtures."""

    mean: GaussianCoefficientPrior
    scale: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


@dataclass(frozen=True)
class GammaMixtureCoefficientPriors:
    """Coefficient priors for mean/variance and shape/scale gamma mixtures."""

    mean: GaussianCoefficientPrior
    variance: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


@dataclass(frozen=True)
class BetaRegMixtureCoefficientPriors:
    """Coefficient priors for beta-regression mixtures."""

    mean: GaussianCoefficientPrior
    dispersion: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


@dataclass(frozen=True)
class BinomialMixtureCoefficientPriors:
    """Coefficient priors for binomial mixtures."""

    mean: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


@dataclass(frozen=True)
class BetaBinMixtureCoefficientPriors:
    """Coefficient priors for beta-binomial mixtures."""

    mean: GaussianCoefficientPrior
    dispersion: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


@dataclass(frozen=True)
class PoissonMixtureCoefficientPriors:
    """Coefficient priors for Poisson mixtures."""

    mean: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


@dataclass(frozen=True)
class NegBinMixtureCoefficientPriors:
    """Coefficient priors for negative-binomial mixtures."""

    mean: GaussianCoefficientPrior
    dispersion: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


@dataclass(frozen=True)
class SplitTMixtureCoefficientPriors:
    """Coefficient priors for the split-t mixture scaffold."""

    mean: GaussianCoefficientPrior
    df: GaussianCoefficientPrior
    scale: GaussianCoefficientPrior
    skewness: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


@dataclass(frozen=True)
class StudentTMixtureCoefficientPriors:
    """Coefficient priors for the symmetric Student-t mixture scaffold."""

    mean: GaussianCoefficientPrior
    df: GaussianCoefficientPrior
    scale: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


@dataclass(frozen=True)
class SplitNormalMixtureCoefficientPriors:
    """Coefficient priors for the split-normal mixture scaffold."""

    mean: GaussianCoefficientPrior
    scale: GaussianCoefficientPrior
    skewness: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None


def convert_prior_to_link_scale(
    prior_mean_feat: float,
    prior_std_feat: float,
    link_type: str | float,
) -> tuple[float, float]:
    """Convert an elicited feature prior to the linear predictor scale.

    This follows MATLAB ``convertPrior.m``: identity and log links use the
    closed-form transformations; logit links first convert the requested
    feature mean/standard deviation to a beta prior and then fit the closest
    logistic-normal intercept prior on MATLAB's grid; other links use the
    local delta-method approximation.
    """

    mean = float(prior_mean_feat)
    std = float(prior_std_feat)
    if std <= 0:
        raise ValueError("prior_std_feat must be positive")

    if isinstance(link_type, str):
        name = link_type.lower()
    else:
        name = ""

    if name == "identity":
        return mean, std
    if name == "log":
        if mean <= 0:
            raise ValueError("log-link feature prior mean must be positive")
        link_std = math.sqrt(math.log((std / mean) ** 2 + 1.0))
        link_mean = math.log(mean) - 0.5 * link_std**2
        return link_mean, link_std
    if name == "logit":
        try:
            return _logit_prior_from_beta_kl(mean, std)
        except ValueError:
            if not 0.0 < mean < 1.0:
                raise
            return math.log(mean / (1.0 - mean)), std

    link_mean = _link_eval_scalar(mean, link_type)
    deriv = _link_derivative_scalar(mean, link_type)
    return link_mean, std * abs(deriv)


def build_gaussian_coefficient_prior(
    design_matrix: np.ndarray,
    prior_mean_feat: float,
    prior_std_feat: float,
    link_type: str | float,
    shrinkage: Shrinkage = 100.0,
    unit_info_diag: np.ndarray | None = None,
) -> GaussianCoefficientPrior:
    """Build the MATLAB-style prior for one feature coefficient vector."""

    X = _as_2d_design(design_matrix)
    n_obs, n_covariates = X.shape
    if n_covariates < 1:
        raise ValueError("design_matrix must contain at least one column")

    prior_mean, prior_std = convert_prior_to_link_scale(
        prior_mean_feat,
        prior_std_feat,
        link_type,
    )
    mean = np.zeros(n_covariates, dtype=float)
    covariance = np.eye(n_covariates, dtype=float)
    mean[0] = prior_mean
    covariance[0, 0] = prior_std**2

    non_intercept = np.arange(1, n_covariates)
    if non_intercept.size:
        if _is_unit_info(shrinkage):
            gram = X[:, non_intercept].T @ X[:, non_intercept]
            if unit_info_diag is not None:
                diag = np.asarray(unit_info_diag, dtype=float).reshape(-1)
                if diag.shape[0] != n_obs:
                    raise ValueError("unit_info_diag length must match design rows")
                gram = X[:, non_intercept].T @ (diag[:, None] * X[:, non_intercept])
            covariance[np.ix_(non_intercept, non_intercept)] = (
                -n_obs * _regularized_inverse(gram)
                if unit_info_diag is not None
                else n_obs * _regularized_inverse(gram)
            )
        else:
            covariance[np.ix_(non_intercept, non_intercept)] = (
                float(shrinkage) * np.eye(non_intercept.size)
            )

    return _complete_prior(mean, covariance, X)


def build_gating_prior(
    design_matrix: np.ndarray,
    n_components: int,
    shrinkage: Shrinkage = "UnitInfo",
) -> GaussianCoefficientPrior | None:
    """Build the MATLAB mixing-function prior.

    For ``UnitInfo`` this is Zellner's g-prior,
    ``n * inv(Z.T @ Z)``, reused for each non-reference component.
    """

    if n_components < 1:
        raise ValueError("n_components must be positive")
    if n_components == 1:
        return None

    Z = _as_2d_design(design_matrix)
    n_obs, n_covariates = Z.shape
    if _is_unit_info(shrinkage):
        covariance = n_obs * _regularized_inverse(Z.T @ Z)
    else:
        covariance = float(shrinkage) * np.eye(n_covariates)

    mean = np.zeros(n_covariates, dtype=float)
    return _complete_prior(mean, covariance, Z)


def build_gaussian_mixture_priors(
    inputs: Any,
    setting: Any,
) -> GaussianMixtureCoefficientPriors:
    """Build all coefficient priors needed by ``fit_gaussian_mixture_vb``."""

    unit_info_diag = _gaussian_feature_unit_info_diagonals(inputs, setting)
    return GaussianMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
            unit_info_diag[0],
        ),
        log_variance=build_gaussian_coefficient_prior(
            inputs.X_variance,
            setting.prior_mean_feat[1],
            setting.prior_std_feat[1],
            setting.link_types[1],
            setting.prior_shrink[1],
            unit_info_diag[1],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def build_lognormal_mixture_priors(
    inputs: Any,
    setting: Any,
) -> LogNormalMixtureCoefficientPriors:
    """Build all coefficient priors needed by lognormal mixture VB fits."""

    return LogNormalMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
        ),
        scale=build_gaussian_coefficient_prior(
            inputs.X_scale,
            setting.prior_mean_feat[1],
            setting.prior_std_feat[1],
            setting.link_types[1],
            setting.prior_shrink[1],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def build_gamma_mixture_priors(
    inputs: Any,
    setting: Any,
) -> GammaMixtureCoefficientPriors:
    """Build all coefficient priors needed by gamma mixture VB fits."""

    return GammaMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
        ),
        variance=build_gaussian_coefficient_prior(
            inputs.X_variance,
            setting.prior_mean_feat[1],
            setting.prior_std_feat[1],
            setting.link_types[1],
            setting.prior_shrink[1],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def build_betareg_mixture_priors(
    inputs: Any,
    setting: Any,
) -> BetaRegMixtureCoefficientPriors:
    """Build all coefficient priors needed by beta-regression mixture VB fits."""

    return BetaRegMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
        ),
        dispersion=build_gaussian_coefficient_prior(
            inputs.X_dispersion,
            setting.prior_mean_feat[1],
            setting.prior_std_feat[1],
            setting.link_types[1],
            setting.prior_shrink[1],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def build_binomial_mixture_priors(
    inputs: Any,
    setting: Any,
) -> BinomialMixtureCoefficientPriors:
    """Build all coefficient priors needed by binomial mixture VB fits."""

    return BinomialMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def build_betabin_mixture_priors(
    inputs: Any,
    setting: Any,
) -> BetaBinMixtureCoefficientPriors:
    """Build all coefficient priors needed by beta-binomial mixture VB fits."""

    return BetaBinMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
        ),
        dispersion=build_gaussian_coefficient_prior(
            inputs.X_dispersion,
            setting.prior_mean_feat[1],
            setting.prior_std_feat[1],
            setting.link_types[1],
            setting.prior_shrink[1],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def build_poisson_mixture_priors(
    inputs: Any,
    setting: Any,
) -> PoissonMixtureCoefficientPriors:
    """Build all coefficient priors needed by Poisson mixture VB fits."""

    return PoissonMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def build_negbin_mixture_priors(
    inputs: Any,
    setting: Any,
) -> NegBinMixtureCoefficientPriors:
    """Build all coefficient priors needed by negative-binomial mixture VB fits."""

    return NegBinMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
        ),
        dispersion=build_gaussian_coefficient_prior(
            inputs.X_dispersion,
            setting.prior_mean_feat[1],
            setting.prior_std_feat[1],
            setting.link_types[1],
            setting.prior_shrink[1],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def build_splitt_mixture_priors(
    inputs: Any,
    setting: Any,
) -> SplitTMixtureCoefficientPriors:
    """Build all coefficient priors needed by ``fit_splitt_mixture_vb``."""

    return SplitTMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
        ),
        df=build_gaussian_coefficient_prior(
            inputs.X_df,
            setting.prior_mean_feat[1],
            setting.prior_std_feat[1],
            setting.link_types[1],
            setting.prior_shrink[1],
        ),
        scale=build_gaussian_coefficient_prior(
            inputs.X_scale,
            setting.prior_mean_feat[2],
            setting.prior_std_feat[2],
            setting.link_types[2],
            setting.prior_shrink[2],
        ),
        skewness=build_gaussian_coefficient_prior(
            inputs.X_skewness,
            setting.prior_mean_feat[3],
            setting.prior_std_feat[3],
            setting.link_types[3],
            setting.prior_shrink[3],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def build_studentt_mixture_priors(
    inputs: Any,
    setting: Any,
) -> StudentTMixtureCoefficientPriors:
    """Build all coefficient priors needed by ``fit_studentt_mixture_vb``."""

    return StudentTMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
        ),
        df=build_gaussian_coefficient_prior(
            inputs.X_df,
            setting.prior_mean_feat[1],
            setting.prior_std_feat[1],
            setting.link_types[1],
            setting.prior_shrink[1],
        ),
        scale=build_gaussian_coefficient_prior(
            inputs.X_scale,
            setting.prior_mean_feat[2],
            setting.prior_std_feat[2],
            setting.link_types[2],
            setting.prior_shrink[2],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def build_splitnormal_mixture_priors(
    inputs: Any,
    setting: Any,
) -> SplitNormalMixtureCoefficientPriors:
    """Build all coefficient priors needed by ``fit_splitnormal_mixture_vb``."""

    return SplitNormalMixtureCoefficientPriors(
        mean=build_gaussian_coefficient_prior(
            inputs.X_mean,
            setting.prior_mean_feat[0],
            setting.prior_std_feat[0],
            setting.link_types[0],
            setting.prior_shrink[0],
        ),
        scale=build_gaussian_coefficient_prior(
            inputs.X_scale,
            setting.prior_mean_feat[1],
            setting.prior_std_feat[1],
            setting.link_types[1],
            setting.prior_shrink[1],
        ),
        skewness=build_gaussian_coefficient_prior(
            inputs.X_skewness,
            setting.prior_mean_feat[2],
            setting.prior_std_feat[2],
            setting.link_types[2],
            setting.prior_shrink[2],
        ),
        gating=build_gating_prior(
            inputs.Z,
            setting.n_components,
            setting.prior_shrink_mix,
        ),
    )


def _gaussian_feature_unit_info_diagonals(
    inputs: Any,
    setting: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if setting.model_name.lower() != "heterogauss":
        return None, None
    if not any(_is_unit_info(shrinkage) for shrinkage in setting.prior_shrink):
        return None, None

    mean_eta, _ = convert_prior_to_link_scale(
        setting.prior_mean_feat[0],
        setting.prior_std_feat[0],
        setting.link_types[0],
    )
    variance_eta, _ = convert_prior_to_link_scale(
        setting.prior_mean_feat[1],
        setting.prior_std_feat[1],
        setting.link_types[1],
    )
    mean_feat = _inverse_link_eval_scalar(mean_eta, setting.link_types[0])
    variance_feat = _inverse_link_eval_scalar(variance_eta, setting.link_types[1])
    n_obs = np.asarray(inputs.y).reshape(-1).shape[0]

    mean_diag = None
    if _is_unit_info(setting.prior_shrink[0]):
        mean_kprime = _link_derivative_scalar(mean_feat, setting.link_types[0])
        mean_diag = np.full(n_obs, (-1.0 / variance_feat) / mean_kprime**2)

    variance_diag = None
    if _is_unit_info(setting.prior_shrink[1]):
        variance_kprime = _link_derivative_scalar(variance_feat, setting.link_types[1])
        variance_diag = np.full(
            n_obs,
            (-1.0 / (2.0 * variance_feat**2)) / variance_kprime**2,
        )

    return mean_diag, variance_diag


def gaussian_matrix_log_prob(
    value: jnp.ndarray,
    prior: GaussianCoefficientPrior | None,
) -> jnp.ndarray:
    """Multivariate Gaussian log density for each row of a coefficient matrix."""

    if prior is None or value.size == 0:
        return jnp.asarray(0.0, dtype=value.dtype)

    if value.shape[-1] != prior.mean.shape[0]:
        raise ValueError("prior and coefficient matrix column counts differ")

    mean = jnp.asarray(prior.mean, dtype=value.dtype)
    precision = jnp.asarray(prior.precision, dtype=value.dtype)
    delta = value - mean
    quad = jnp.einsum("...i,ij,...j->...", delta, precision, delta)
    log_norm = -0.5 * (
        prior.mean.shape[0] * jnp.log(2.0 * jnp.pi) + prior.log_det_covariance
    )
    return jnp.sum(log_norm - 0.5 * quad)


def coefficient_log_prior(
    value: jnp.ndarray,
    design_matrix: np.ndarray,
    coefficient_prior_scale: float,
    use_ard: bool,
    ard_shape: float,
    ard_rate: float,
    prior: GaussianCoefficientPrior | None = None,
) -> jnp.ndarray:
    """Coefficient prior used by the JAX ELBO.

    Without ARD and with an explicit prior object this is the MATLAB Gaussian
    prior. With ARD, constant/intercept columns retain their Gaussian prior,
    while non-constant columns use the existing integrated Gamma ARD penalty.
    """

    if coefficient_prior_scale <= 0:
        raise ValueError("coefficient_prior_scale must be positive")
    if not use_ard:
        if prior is not None:
            return gaussian_matrix_log_prob(value, prior)
        return -0.5 * jnp.sum((value / coefficient_prior_scale) ** 2)
    if ard_shape <= 0 or ard_rate <= 0:
        raise ValueError("ard_shape and ard_rate must be positive")
    if value.size == 0:
        return jnp.asarray(0.0, dtype=value.dtype)

    constant_mask = _constant_column_mask(design_matrix)
    if constant_mask.shape[0] != value.shape[-1]:
        raise ValueError("design matrix and coefficient matrix column counts differ")

    if prior is not None:
        constant_prior = _constant_column_gaussian_log_prob(value, prior, constant_mask)
    else:
        constant_weights = jnp.asarray(constant_mask, dtype=value.dtype)
        constant_prior = -0.5 * jnp.sum(
            ((value / coefficient_prior_scale) ** 2) * constant_weights
        )

    column_squares = jnp.sum(value**2, axis=0)
    group_size = value.shape[0]
    nonconstant_weights = 1.0 - jnp.asarray(constant_mask, dtype=value.dtype)
    ard_prior = -jnp.sum(
        nonconstant_weights
        * (ard_shape + 0.5 * group_size)
        * jnp.log(ard_rate + 0.5 * column_squares)
    )
    return constant_prior + ard_prior


def coefficient_log_prior_sum(
    param_tree: dict[str, jnp.ndarray],
    specs,
    coefficient_prior_scale: float,
    use_ard: bool,
    ard_shape: float,
    ard_rate: float,
) -> jnp.ndarray:
    """Sum coefficient priors for a model ELBO.

    Each spec is ``(coefficient_name, design_matrix, prior_or_none)``.
    """

    total = jnp.asarray(0.0)
    for coefficient_name, design_matrix, prior in specs:
        total = total + coefficient_log_prior(
            param_tree[coefficient_name],
            design_matrix,
            coefficient_prior_scale,
            use_ard,
            ard_shape,
            ard_rate,
            prior,
        )
    return total


def _constant_column_gaussian_log_prob(
    value: jnp.ndarray,
    prior: GaussianCoefficientPrior,
    constant_mask: np.ndarray,
) -> jnp.ndarray:
    if value.shape[-1] != prior.mean.shape[0]:
        raise ValueError("prior and coefficient matrix column counts differ")

    weights = jnp.asarray(constant_mask, dtype=value.dtype)
    mean = jnp.asarray(prior.mean, dtype=value.dtype)
    precision_diag = jnp.asarray(np.diag(prior.precision), dtype=value.dtype)
    covariance_diag = np.diag(prior.covariance)
    log_variance = jnp.asarray(np.log(covariance_diag), dtype=value.dtype)
    delta = value - mean
    per_value = -0.5 * (
        jnp.log(2.0 * jnp.pi) + log_variance + precision_diag * delta**2
    )
    return jnp.sum(per_value * weights)


def _complete_prior(
    mean: np.ndarray,
    covariance: np.ndarray,
    design_matrix: np.ndarray,
) -> GaussianCoefficientPrior:
    covariance = _symmetrize(covariance)
    precision = _regularized_inverse(covariance)
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(log_det):
        covariance = _make_positive_definite(covariance)
        precision = _regularized_inverse(covariance)
        sign, log_det = np.linalg.slogdet(covariance)
    return GaussianCoefficientPrior(
        mean=np.asarray(mean, dtype=float),
        covariance=covariance,
        precision=_symmetrize(precision),
        log_det_covariance=float(log_det),
        constant_mask=_constant_column_mask(design_matrix),
    )


def _logit_prior_from_beta_kl(mean: float, std: float) -> tuple[float, float]:
    a, b = _beta_parameters_from_mean_std(mean, std)
    x = np.arange(0.005, 0.995 + 0.0005, 0.001)
    logit_x = np.log(x / (1.0 - x))
    log_beta_pdf = (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x) - betaln(a, b)

    sigma0 = max(std / (mean * (1.0 - mean)), 1e-6)
    initial = np.array([math.log(mean / (1.0 - mean)), math.log(sigma0)])

    def objective(params: np.ndarray) -> float:
        loc = params[0]
        scale = math.exp(params[1])
        log_logistic_normal = (
            -np.log(x * (1.0 - x))
            - math.log(scale)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * ((logit_x - loc) / scale) ** 2
        )
        kl = np.sum((log_beta_pdf - log_logistic_normal) * np.exp(log_beta_pdf))
        return float(kl) if np.isfinite(kl) else np.inf

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=((-50.0, 50.0), (math.log(1e-6), math.log(100.0))),
    )
    params = result.x if result.success and np.all(np.isfinite(result.x)) else initial
    return float(params[0]), float(math.exp(params[1]))


def _beta_parameters_from_mean_std(mean: float, std: float) -> tuple[float, float]:
    if not 0.0 < mean < 1.0:
        raise ValueError("logit-link feature prior mean must be between 0 and 1")
    variance = std**2
    max_variance = mean * (1.0 - mean)
    if not 0.0 < variance < max_variance:
        raise ValueError("invalid beta prior implied by logit feature mean/std")
    common = max_variance / variance - 1.0
    return mean * common, (1.0 - mean) * common


def _link_eval_scalar(value: float, link_type: str | float) -> float:
    if not isinstance(link_type, str):
        if value <= 0:
            raise ValueError("power-link feature prior mean must be positive")
        return value ** float(link_type)

    name = link_type.lower()
    if name == "loglog":
        return math.log(-math.log(value))
    if name in {"comploglog", "cloglog"}:
        return math.log(-math.log1p(-value))
    if name == "reciprocal":
        return 1.0 / value
    if name == "log1":
        return math.log(value - 1.0)
    raise ValueError(f"unknown link type: {link_type}")


def _inverse_link_eval_scalar(value: float, link_type: str | float) -> float:
    if not isinstance(link_type, str):
        if value <= 0:
            raise ValueError("power-link linear predictor must be positive")
        return value ** (1.0 / float(link_type))

    name = link_type.lower()
    if name == "identity":
        return value
    if name == "log":
        return math.exp(value)
    if name == "logit":
        return 1.0 / (1.0 + math.exp(-value))
    if name == "loglog":
        return math.exp(-math.exp(value))
    if name in {"comploglog", "cloglog"}:
        return -math.expm1(-math.exp(value))
    if name == "reciprocal":
        return 1.0 / value
    if name == "log1":
        return math.exp(value) + 1.0
    raise ValueError(f"unknown link type: {link_type}")


def _link_derivative_scalar(value: float, link_type: str | float) -> float:
    if not isinstance(link_type, str):
        power = float(link_type)
        return power * value ** (power - 1.0)

    name = link_type.lower()
    if name == "identity":
        return 1.0
    if name == "log":
        return 1.0 / value
    if name == "logit":
        return 1.0 / (value * (1.0 - value))
    if name == "loglog":
        return 1.0 / (math.log(value) * value)
    if name in {"comploglog", "cloglog"}:
        return -1.0 / ((1.0 - value) * math.log1p(-value))
    if name == "reciprocal":
        return -1.0 / value**2
    if name == "log1":
        return 1.0 / (value - 1.0)
    raise ValueError(f"unknown link type: {link_type}")


def _as_2d_design(design_matrix: np.ndarray) -> np.ndarray:
    X = np.asarray(design_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("design_matrix must be 2D")
    if X.shape[0] == 0:
        raise ValueError("design_matrix must contain at least one observation")
    return X


def _is_unit_info(shrinkage: Shrinkage) -> bool:
    return isinstance(shrinkage, str) and shrinkage.lower() == "unitinfo"


def _constant_column_mask(design_matrix: np.ndarray) -> np.ndarray:
    X = _as_2d_design(design_matrix)
    return np.all(np.isclose(X, X[:1, :]), axis=0)


def _regularized_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    scale = max(float(np.mean(np.abs(np.diag(matrix)))) if matrix.size else 1.0, 1.0)
    for jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        adjusted = matrix + jitter * scale * np.eye(matrix.shape[0])
        try:
            return np.linalg.inv(adjusted)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(matrix + 1e-6 * scale * np.eye(matrix.shape[0]))


def _make_positive_definite(matrix: np.ndarray) -> np.ndarray:
    matrix = _symmetrize(matrix)
    min_eig = float(np.min(np.linalg.eigvalsh(matrix)))
    if min_eig > 0:
        return matrix
    return matrix + (abs(min_eig) + 1e-8) * np.eye(matrix.shape[0])


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
