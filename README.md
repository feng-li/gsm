# gsm

The Python module `gsm` implements the covariate-dependent smooth mixture models developed by [Mattias Villani](https://mattiasvillani.com/), [Feng Li](https://feng.li), [Robert Kohn](https://www.unsw.edu.au/staff/robert-kohn), et al. The current implementation focuses on light-weighted fitting with variational Bayes in JAX with both CPU and GPU support.

The package is intentionally small while the implementation is underway. The Python
target is variational inference rather than Metropolis-Hastings with Newton
updates in Villani, Li, et al's original published papers.

## General Purpose

This repository supports research and migration work for Bayesian
mixture-of-experts (MoE) models for conditional density forecasting. In the original
MATLAB code, the method is framed as Generalized Smooth Mixtures: each mixture
component is an expert distribution, the gating function assigns
covariate-dependent mixture probabilities, and each distributional feature can
also depend on covariates through link functions.

The financial forecasting use case is full predictive distribution estimation,
not only point forecasting. The S&P 500 project scripts use smooth mixtures of
asymmetric Student t and asymmetric normal experts to forecast return
distributions with time-varying scale, skewness, tail behavior, and mixture
probabilities. This follows the mixture-of-experts (MoE) density-forecasting
literature cited in the MATLAB code, especially the smooth adaptive Gaussian
mixture work of Villani, Kohn, and Giordani; the generalized smooth-mixture
framework of Kohn, Villani, and Nott; and the asymmetric Student t financial
forecasting application of Li, Villani, and Kohn.

The Python package keeps that modeling goal but replaces the original
Metropolis-Hastings/Newton sampler with JAX-based variational Bayes, standard
Python data tooling, and posterior-sampled held-out ELPD for forecast
comparison.

## Features

- CSV and MATLAB `.mat` data loading helpers.
- JAX-compatible link and inverse-link functions.
- Gaussian mixture kernel with identity-linked means, log-linked variances, and
  multinomial-logit gating.
- Beta-regression mixture kernel for the Rajan debt-ratio example.
- Poisson and negative-binomial mixture kernels for count-response examples.
- Binomial and beta-binomial mixture kernels with two-column responses:
  `successes, trials`.
- Generalized Poisson mixture kernels for `GenPois` and `GenPoisAlt`.
- Gamma/GammaRep mixture kernel with shared mean-variance or shape-scale
  parameterization.
- Symmetric Student-t mixture kernel with mean, degrees of freedom, and scale.
- Exponential-hazard and Weibull interval-survival mixture kernels.
- Mean-field Gaussian variational posterior over mixture coefficients.
- Optional ARD shrinkage for non-constant covariates.
- Posterior-sampled held-out ELPD/LPDS for chronological train/test evaluation.

## FEBAMA

`gsm.febama` ports the R `febama` application layer for feature-based Bayesian
forecast model averaging. FEBAMA is a density-forecast combination method: base
forecasters produce predictive distributions, time-series features drive
softmax model weights, and the fitted combination is scored by log predictive
density.

FEBAMA is kept separate from `gsm.models` because its expert distributions come
from external forecasting models or precomputed predictive densities rather than
from one GSM mixture kernel. The current Python slice includes:

- JAX softmax-weight and log predictive score kernels.
- A predictive-distribution registry for Gaussian, Student-t, split-normal,
  split-t, lognormal, gamma, and Poisson forecasts.
- Simple base forecasters: naive and random-walk-with-drift.
- Optional `statsforecast` AutoETS/AutoARIMA and `arch` GARCH/EGARCH adapters.
- `tsfeatures` integration, feature cleaning/scaling, and precomputed feature
  table loading.
- MAP fitting for FEBAMA gating coefficients.

## Installation

From this directory:

```bash
python -m pip install -e ".[dev]"
```

For FEBAMA examples with optional forecasters and `tsfeatures`, install:

```bash
python -m pip install -e ".[dev,febama]"
```

The package requires Python 3.11 or newer. The editable install exposes the
`gsm` Python package and installs runtime dependencies declared in
`pyproject.toml`.

## Quick Check

Run the test suite:

```bash
python -m pytest
```

Run the default S&P 500 Gaussian mixture fit:

```bash
python scripts/run_gaussian_sp500.py --max-iter 50 --restarts 1
```

Run the Rajan beta-regression mixture fit:

```bash
python scripts/run_betareg_rajan.py --max-iter 50 --restarts 1
```

Run chronological held-out ELPD with posterior predictive sampling:

```bash
python scripts/run_gaussian_sp500.py \
  --max-iter 50 \
  --restarts 1 \
  --holdout-fraction 0.2 \
  --elbo-samples 8 \
  --predictive-samples 100
```

Compare implemented continuous-response models:

```bash
python scripts/compare_continuous_models.py \
  --groups returns positive rajan \
  --max-iter 50 \
  --restarts 1
```

Compare implemented count-response models:

```bash
python scripts/compare_count_models.py \
  --max-iter 50 \
  --restarts 1
```

Run the minimal FEBAMA example on the default S&P 500 return data:

```bash
python scripts/run_febama_example.py --max-origins 4 --test-size 1 --max-iter 100
```

Enable ARD shrinkage:

```bash
python scripts/run_gaussian_sp500.py \
  --max-iter 50 \
  --restarts 1 \
  --holdout-fraction 0.2 \
  --ard
```

## Package Layout

```text
gsm/
  config.py                 # model and fit dataclasses
  data.py                   # CSV/.mat loading, splits, standardization
  evaluation.py             # posterior-sampled predictive scores
  links.py                  # link and inverse-link functions
  variational.py            # public VB facade and model dispatch
  vi/
    common.py               # shared VB result and preprocessing helpers
    engine.py               # shared optimizer and posterior sampling helpers
    betabinomial.py         # BetaBin VB fitting
    binomial.py             # Bin VB fitting
    gamma.py                # Gamma/GammaRep VB fitting
    genpoisson.py           # GenPois/GenPoisAlt VB fitting
    gaussian.py             # Gaussian-mixture VB fitting
    lognormal.py            # LogNorm/LogNormRep VB fitting
    negbin.py               # negative-binomial VB fitting
    poisson.py              # Poisson VB fitting
    betareg.py              # BetaReg VB fitting
    splitnormal.py          # split-normal VB fitting
    splitt.py               # split-t VB fitting
    studentt.py             # symmetric Student-t VB fitting
  models/
    betabinomial.py         # beta-binomial mixture log-density and predictions
    betareg.py              # beta-regression mixture log-density
    binomial.py             # binomial mixture log-density and predictions
    exphazard.py            # exponential-hazard interval-survival log-density
    gamma.py                # gamma mixture log-density and predictions
    genpoisson.py           # generalized Poisson mixture log-density and predictions
    gaussian.py             # Gaussian mixture log-density and predictions
    lognormal.py            # lognormal mixture log-density and predictions
    negbin.py               # negative-binomial mixture log-density and predictions
    poisson.py              # Poisson mixture log-density and predictions
    splitnormal.py          # split-normal mixture log-density and predictions
    splitt.py               # split-t mixture log-density and predictions
    studentt.py             # symmetric Student-t mixture log-density and predictions
    weibull_survival.py     # Weibull interval-survival log-density
  febama/
    api.py                  # FEBAMA workflow wrappers for precomputed LPD/features
    config.py               # FEBAMA configuration object
    data.py                 # FEBAMA data containers
    distributions.py        # predictive distribution registry
    features.py             # tsfeatures adapter and feature cleaning/scaling
    forecasters.py          # naive, drift, AutoETS/AutoARIMA, GARCH/EGARCH
    inference.py            # MAP fitting for gating coefficients
    scoring.py              # JAX softmax weights and log predictive score
scripts/
  BetaReg_config.py         # default Rajan beta-regression specification
  Gaussian_config.py        # default S&P 500 Gaussian mixture specification
  compare_continuous_models.py
  compare_count_models.py      # mdvisits Poisson/NegBin comparison
  run_betareg_rajan.py      # Rajan BetaReg command-line runner
  run_febama_example.py     # minimal FEBAMA command-line example
  run_gaussian_sp500.py     # command-line runner
tests/
  test_*.py                 # focused migration tests
data/
  mdvisits_reduced.csv
  Rajan.csv
  sp500_1990-2009.csv
  sp500_1990-2009_calendar.csv
```


## Literature
- Li, L., Kang, Y., & Li, F. (2023). Bayesian forecast combination using time-varying features. International Journal of Forecasting, 39(3), 1287–1302. https://doi.org/10.1016/j.ijforecast.2022.06.002
- Villani, M., Kohn, R., & Nott, D. J. (2012). Generalized Smooth Finite Mixtures. Journal of Econometrics, 171(2), 121–133. https://doi.org/10.1016/j.jeconom.2012.06.012
- Nott, D. J., Tan, S. L., Villani, M., & Kohn, R. (2012). Regression Density Estimation With Variational Methods and Stochastic Approximation. Journal of Computational and Graphical Statistics, 21(3), 797–820. https://doi.org/10.1080/10618600.2012.679897
- Li, F., Villani, M., & Kohn, R. (2011). Modeling conditional densities using finite smooth mixtures. In K. Mengersen, C. Robert, & M. Titterington (Eds.), Mixtures: Estimation and applications (pp. 123–144). John Wiley & Sons Inc, Chichester. https://doi.org/10.1002/9781119995678.ch6
- Li, F., Villani, M. and Kohn, R. (2010). Flexible modeling of conditional distributions using smooth mixtures of asymmetric Student t densities. Journal
  of Statistical Planning and Inference, 140(12), 3638-3654. https://doi.org/10.1016/j.jspi.2010.04.031.
- Villani, M., Kohn, R., & Giordani, P. (2009). Regression density estimation using smooth adaptive Gaussian mixtures. Journal of Econometrics, 153(2), 155–173. https://doi.org/10.1016/j.jeconom.2009.05.004


## Basic API

```python
from gsm.config import FitConfig, sp500_gaussian_mixture_setting
from gsm.evaluation import fit_heldout_gaussian_mixture_lpds
from gsm.data import load_csv_dataset

setting = sp500_gaussian_mixture_setting(n_components=3)
dataset = load_csv_dataset(
    "data/sp500_1990-2009_calendar.csv",
    response_column="Returns",
    add_constant=setting.add_constant,
)

fit = FitConfig(
    max_iter=200,
    learning_rate=1e-2,
    n_restarts=1,
    n_elbo_samples=8,
    n_predictive_samples=100,
    use_ard=False,
)

result = fit_heldout_gaussian_mixture_lpds(
    dataset,
    setting,
    fit,
    test_size=0.2,
)

print(result.test_score.elpd)
print(result.test_score.mean_elpd)
```

`result.train_result.params` contains posterior means for compatibility with
earlier code. `result.train_result.posterior` contains the fitted mean-field
Gaussian posterior with coefficient means and log standard deviations.

## FEBAMA Basic API

For precomputed component log predictive densities and feature matrices:

```python
from gsm.febama import clean_features, fit_febama, prepare_lpd_features, score_febama

lpd_features = prepare_lpd_features(
    lpd,
    features,
    model_names=("naive", "rw_drift"),
    feature_names=("x_acf1", "entropy"),
)
lpd_features = clean_features(lpd_features)
fit = fit_febama(lpd_features, coefficient_prior_scale=10.0)
score = score_febama(lpd_features, fit)

print(score.total)
```

For live feature extraction, use `compute_tsfeatures(...)` or the example
script `scripts/run_febama_example.py`. The script accepts `--data`, so data
paths stay outside the FEBAMA config.

## Data

The default Gaussian mixture script uses:

```text
data/sp500_1990-2009_calendar.csv
```

The count-model comparison script uses:

```text
data/mdvisits_reduced.csv
```

Binomial-family models use a two-column response convention:

```text
y[:, 0] = successes
y[:, 1] = trials
```

For CSV files, use `load_binomial_csv_dataset(...)` with default response
column names `successes` and `trials`.
