# GSM Python Models

This file is the working model reference for the Python GSM migration. It
tracks which MATLAB model families have Python kernels, and records the response
conventions, links, and component parameterizations used by the current JAX/VB
code.

## Common Mixture Form

All implemented GSM kernels use the same finite-mixture structure:

```text
p(y_i | x_i, z_i) = sum_k pi_ik f_k(y_i | theta_ik)
```

Component features are covariate dependent. For feature block `r` and component
`k`,

```text
eta_{r,ik} = X_{r,i} beta_{r,k}
theta_{r,ik} = inverse_link_r(eta_{r,ik})
```

Mixture weights use reference-class multinomial-logit gating:

```text
s_{i,1} = 0
s_{i,k} = Z_i gamma_k, k = 2, ..., K
pi_ik = exp(s_ik) / sum_l exp(s_il)
```

So `gating_coef` has shape `(n_components - 1, n_cov_mix)`. With one component,
`gating_coef` is empty and the only mixture weight is one.

## Shared Implementation Conventions

- Coefficients are component-specific arrays with shape
  `(n_components, n_covariates_for_feature)`.
- `feature_names`, `link_types`, `covs`, and `covs_mix` in `gsm.config` describe
  which design matrix columns feed each feature and the gating model.
- `add_constant=True` means the data loader inserts the intercept column. Config
  files should not hard-code a constant column in the data source.
- Standardization follows the MATLAB-style `standardize` flag:
  `0` none, `1` z-score, `2` map to approximately `[-1, 1]`, `3` center only.
- VB is mean-field Gaussian over the model coefficient tree. A posterior object
  contains `mean` and `log_std` dataclasses with the same fields as the model
  parameter dataclass.
- Gradients and ELBO optimization are computed with JAX/Optax.
- ARD is optional through `FitConfig(use_ard=True)`. It is currently applied as
  an integrated Gamma shrinkage penalty for non-constant columns, while constant
  columns retain their Gaussian prior treatment.
- MATLAB link-scale intercept priors and unit-information/g-prior conversions
  live in `gsm/priors.py`.

## Link Functions

| Link | Inverse link | Typical use |
| --- | --- | --- |
| `identity` | `eta` | real-valued location |
| `log` | `exp(eta)` | positive scale, variance, rate, mean, dispersion |
| `logit` | `1 / (1 + exp(-eta))` | probabilities and beta means |
| `log1` | `1 + exp(eta)` | dispersion constrained above one |
| `loglog` | `exp(-exp(eta))` | migrated helper, not central in current kernels |
| `cloglog` / `comploglog` | `1 - exp(-exp(eta))` | migrated helper |
| `reciprocal` | `1 / eta` | migrated helper |

## Response Conventions

| Model family | Response convention |
| --- | --- |
| Real-valued continuous models | One numeric response column. |
| Positive continuous models | One strictly positive response column. |
| `BetaReg` | One response column in `[0, 1]`; exact 0/1 are nudged only inside the density evaluation. |
| Count models | One finite, non-negative, integer-valued response column. |
| `Bin` / `BetaBin` | Two-column response matrix `[successes, trials]`. |
| Survival kernels | Interval data `(t0, t, event)` where `event` is 0/1. VB wiring is still pending. |

## Implemented Model Parameterizations

### Gaussian / `heteroGauss`

- Python setting: `GaussianMixtureSetting`.
- Response support: real.
- Parameter dataclass: `GaussianMixtureParams`.
- Coefficients: `mean_coef`, `log_variance_coef`, `gating_coef`.
- Features:
  - `Mean`: identity link; component normal mean `mu`.
  - `Variance`: log link; component variance `sigma2`.
- Density: `Normal(mu, sigma2)`.
- Predictive moments use the standard mixture mean and second moment.
- Default S&P 500 setting uses an intercept-only mean, covariate-dependent
  variance, and covariate-dependent gating.

### Beta Regression / `BetaReg`

- Python setting: `BetaRegMixtureSetting`.
- Response support: unit interval.
- Parameter dataclass: `BetaRegMixtureParams`.
- Coefficients: `mean_coef`, `dispersion_coef`, `gating_coef`.
- Features:
  - `Mean`: logit link; beta mean `mu`.
  - `Disp`: log link; MATLAB precision/dispersion `phi`.
- Density parameterization:
  - `alpha = phi * mu`
  - `beta = phi * (1 - mu)`
- Component variance: `mu * (1 - mu) / (1 + phi)`.
- Rajan defaults are available through `rajan_betareg_mixture_setting(...)`.

### Lognormal / `LogNorm`

- Python setting: `LogNormalMixtureSetting`.
- Response support: strictly positive.
- Parameter dataclass: `LogNormalMixtureParams`.
- Coefficients: `mean_coef`, `scale_coef`, `gating_coef`.
- Features:
  - `Mean`: log link.
  - `Scale`: log link.
- `parameterization="standard"`.
- Density uses `log(y) ~ Normal(location, scale^2)`, where the implemented
  feature named `Mean` is passed as the lognormal location and `Scale` is the
  lognormal scale.
- Predictive component mean is `exp(location + 0.5 * scale^2)`.

### Response-Scale Lognormal / `LogNormRep`

- Python setting: `LogNormalRepMixtureSetting`.
- Uses the same kernel and parameter dataclass as `LogNorm`.
- Response support: strictly positive.
- Features:
  - `Mean`: log link; response-scale mean `m`.
  - `Scale`: log link; response-scale standard deviation `s`.
- `parameterization="response"`.
- Converted internally to standard lognormal parameters:
  - `log_scale = sqrt(log1p((s / m)^2))`
  - `location = log(m) - 0.5 * log_scale^2`

### Gamma / `Gamma`

- Python setting: `GammaMixtureSetting`.
- Response support: strictly positive.
- Parameter dataclass: `GammaMixtureParams`.
- Coefficients: `mean_coef`, `variance_coef`, `gating_coef`.
- Features:
  - `Mean`: log link; response mean `mu`.
  - `Variance`: log link; response variance `v`.
- `parameterization="mean_variance"`.
- Converted internally to `Gamma(shape, scale)`:
  - `shape = mu^2 / v`
  - `scale = v / mu`

### Shape/Scale Gamma / `GammaRep`

- Python setting: `GammaRepMixtureSetting`.
- Uses the same kernel and parameter dataclass as `Gamma`.
- Response support: strictly positive.
- Features:
  - `Shape`: log link; gamma shape `a`.
  - `Scale`: log link; gamma scale `b`.
- `parameterization="shape_scale"`.
- Component mean is `a * b`; component variance is `a * b^2`.

### Symmetric Student-t / `studT`

- Python setting: `StudentTMixtureSetting`.
- Response support: real.
- Parameter dataclass: `StudentTMixtureParams`.
- Coefficients: `mean_coef`, `df_coef`, `scale_coef`, `gating_coef`.
- Features:
  - `Mean`: identity link; location `mu`.
  - `DF`: log link; degrees of freedom `nu`.
  - `Scale`: log link; scale `sigma`.
- Component variance is available only when `nu > 2`:
  `nu / (nu - 2) * sigma^2`.

### Split Normal / `AsymNorm`

- Python setting: `SplitNormalMixtureSetting`.
- Python model name: `SplitNormal`.
- Response support: real.
- Parameter dataclass: `SplitNormalMixtureParams`.
- Coefficients: `mean_coef`, `scale_coef`, `skewness_coef`, `gating_coef`.
- Features:
  - `Mean`: identity link; split point/location `mu`.
  - `Sigma`: log link; left scale `sigma`.
  - `Skewness`: log link; right-to-left scale ratio `lambda`.
- The density uses left scale `sigma` when `y <= mu` and right scale
  `sigma * lambda` when `y > mu`. `lambda = 1` is symmetric.

### Split Student-t / `AsymStudT`

- Python setting: `SplitTMixtureSetting`.
- Python model name: `SplitT`.
- Response support: real.
- Parameter dataclass: `SplitTMixtureParams`.
- Coefficients: `mean_coef`, `df_coef`, `scale_coef`, `skewness_coef`,
  `gating_coef`.
- Features:
  - `Mean`: identity link; split point/location `mu`.
  - `DF`: log link; degrees of freedom `nu`.
  - `Scale`: log link; left scale `sigma`.
  - `Skewness`: log link; right-to-left scale ratio `lambda`.
- `lambda = 1` gives the symmetric Student-t kernel.
- Component mean exists when `nu > 1`; component variance exists when `nu > 2`.

### Poisson / `pois`

- Python setting: `PoissonMixtureSetting`.
- Response support: non-negative counts.
- Parameter dataclass: `PoissonMixtureParams`.
- Coefficients: `mean_coef`, `gating_coef`.
- Feature:
  - `Mean`: log link; Poisson mean/rate `mu`.
- Component variance equals `mu`.
- mdvisits defaults are available through
  `mdvisits_poisson_mixture_setting(...)`.

### Negative Binomial / `negBin`

- Python setting: `NegBinMixtureSetting`.
- Response support: non-negative counts.
- Parameter dataclass: `NegBinMixtureParams`.
- Coefficients: `mean_coef`, `dispersion_coef`, `gating_coef`.
- Features:
  - `Mean`: log link; mean `mu`.
  - `Disp`: log link; dispersion/size `phi`.
- Density parameterization:
  - `y ~ NegBin(phi, phi / (phi + mu))`
- Component variance: `mu + mu^2 / phi`.
- mdvisits defaults are available through
  `mdvisits_negbin_mixture_setting(...)`.

### Generalized Poisson / `GenPois`

- Python setting: `GenPoissonMixtureSetting`.
- Response support: non-negative counts.
- Parameter dataclass: `GenPoissonMixtureParams`.
- Coefficients: `mean_coef`, `dispersion_coef`, `gating_coef`.
- Features:
  - `Mean`: log link; mean `mu`.
  - `Disp`: log link; MATLAB mean/dispersion parameter `alpha`.
- `parameterization="standard"`.
- Component variance: `mu * (1 + mu * alpha)^2`.
- mdvisits defaults are available through
  `mdvisits_genpoisson_mixture_setting(...)`.

### Alternative Generalized Poisson / `GenPoisAlt`

- Python setting: `GenPoissonAltMixtureSetting`.
- Uses the same kernel and parameter dataclass as `GenPois`.
- Response support: non-negative counts.
- Features:
  - `Mean`: log link; mean `mu`.
  - `Disp`: `log1` link; dispersion `d = 1 + exp(eta)`.
- `parameterization="alternative"`.
- Component variance: `mu * d^2`.
- This is the Czado-style parameterization retained as a separate setting.

### Binomial / `Bin`

- Python setting: `BinomialMixtureSetting`.
- Response convention: two columns `[successes, trials]`.
- Parameter dataclass: `BinomialMixtureParams`.
- Coefficients: `mean_coef`, `gating_coef`.
- Feature:
  - `Mean`: logit link; success probability `p`.
- Density: `successes ~ Binomial(trials, p)`.
- Predictive mean and variance are for the success count.

### Beta-Binomial / `BetaBin`

- Python setting: `BetaBinMixtureSetting`.
- Response convention: two columns `[successes, trials]`.
- Parameter dataclass: `BetaBinMixtureParams`.
- Coefficients: `mean_coef`, `dispersion_coef`, `gating_coef`.
- Features:
  - `Mean`: logit link; success probability `p`.
  - `Disp`: log link; MATLAB precision `phi`.
- Density parameterization:
  - `alpha = phi * p`
  - `beta = phi * (1 - p)`
  - `successes ~ BetaBinomial(trials, alpha, beta)`
- Predictive mean and variance are for the success count.

### Exponential Hazard / `ExpHazard`

- Python kernel: `gsm/models/exphazard.py`.
- VB wrapper: not yet implemented.
- Response convention: interval survival rows `(t0, t, event)`.
- Parameter dataclass: `ExpHazardMixtureParams`.
- Coefficients: `rate_coef`, `gating_coef`.
- Feature:
  - `Rate`: log link; constant hazard rate over `(t0, t]`.
- Interval likelihood:
  - `event=0`: `log P(T > t | T > t0) = -rate * (t - t0)`
  - `event=1`: `log P(t0 < T <= t | T > t0) = log(1 - exp(-rate * (t - t0)))`

### Weibull Survival / `WeibullSurvival`

- Python kernel: `gsm/models/weibull_survival.py`.
- VB wrapper: not yet implemented.
- Response convention: interval survival rows `(t0, t, event)`.
- Parameter dataclass: `WeibullSurvivalMixtureParams`.
- Coefficients: `rate_coef`, `shape_coef`, `gating_coef`.
- Features:
  - `Rate`: log link.
  - `Shape`: log link.
- Cumulative hazard increment over `(t0, t]`:
  - `rate * (t^shape - t0^shape)`
- The interval likelihood uses the same survived/event convention as
  `ExpHazard`.

## Migration Status

| Model | Python name | Kernel | VB | Held-out ELPD | MATLAB density check | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `heteroGauss` | `GaussianMixtureSetting` | Yes | Yes | Yes | Partial | S&P 500 setting available. |
| `beta/BetaReg` | `BetaRegMixtureSetting` | Yes | Yes | Yes | Partial | Rajan setting available. |
| `LogNorm` | `LogNormalMixtureSetting` | Yes | Yes | Yes | Yes | Standard lognormal parameterization. |
| `LogNormRep` | `LogNormalRepMixtureSetting` | Yes | Yes | Yes | Yes | Response-scale parameterization. |
| `gamma/Gamma` | `GammaMixtureSetting` | Yes | Yes | Yes | Yes | Mean/variance parameterization. |
| `gamma/GammaRep` | `GammaRepMixtureSetting` | Yes | Yes | Yes | Python extension | Direct shape/scale parameterization. |
| `studT` | `StudentTMixtureSetting` | Yes | Yes | Yes | Yes | Symmetric Student-t. |
| `AsymNorm` | `SplitNormalMixtureSetting` | Yes | Yes | Yes | Partial | Python name uses split-normal. |
| `AsymStudT` | `SplitTMixtureSetting` | Yes | Yes | Yes | Partial | Python name uses split-t. |
| `pois` | `PoissonMixtureSetting` | Yes | Yes | Yes | Partial | mdvisits setting available; density formula matched. |
| `negBin` | `NegBinMixtureSetting` | Yes | Yes | Yes | Partial | mdvisits setting available; mean/dispersion parameterization matched. |
| `Bin` | `BinomialMixtureSetting` | Yes | Yes | Yes | Yes | Two-column response `[successes, trials]`. |
| `BetaBin` | `BetaBinMixtureSetting` | Yes | Yes | Yes | Yes | Two-column response `[successes, trials]`; MATLAB phi precision parameterization. |
| `GenPois` | `GenPoissonMixtureSetting` | Yes | Yes | Yes | Yes | mdvisits setting available; MATLAB mean/dispersion parameterization. |
| `GenPoisAlt` | `GenPoissonAltMixtureSetting` | Yes | Yes | Yes | Yes | Czado-style parameterization with `log1` dispersion link. |
| `ExpHazard` | `exphazard.py` | Yes | No | No | Yes | Interval-survival likelihood kernel reimplemented from the old archive; VB response convention still pending. |
| `WeibullSurvival` | `weibull_survival.py` | Yes | No | No | Yes | Interval-survival likelihood kernel reimplemented from the old archive; VB response convention still pending. |

## Next Validation Work

- Add fixed-parameter equivalence tests for Gaussian, SplitNormal, SplitT, and
  BetaReg against the MATLAB formulas.
- Add real-data comparison reports for continuous models using
  `scripts/compare_continuous_models.py`.
- Add real-data mdvisits comparison reports for Poisson and NegBin.
- Extend the mdvisits count comparison to include GenPois.
- Add a real binomial-trials example before extending Bin/BetaBin comparison
  scripts.
- Decide the survival response convention before wiring ExpHazard/WeibullSurvival
  into VB and held-out ELPD.
