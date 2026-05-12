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
- Gamma/GammaRep mixture kernel with shared mean-variance or shape-scale
  parameterization.
- Symmetric Student-t mixture kernel with mean, degrees of freedom, and scale.
- Mean-field Gaussian variational posterior over mixture coefficients.
- Optional ARD shrinkage for non-constant covariates.
- Posterior-sampled held-out ELPD/LPDS for chronological train/test evaluation.

## Installation

From this directory:

```bash
python -m pip install -e ".[dev]"
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
    gamma.py                # Gamma/GammaRep VB fitting
    gaussian.py             # Gaussian-mixture VB fitting
    lognormal.py            # LogNorm/LogNormRep VB fitting
    betareg.py              # BetaReg VB fitting
    splitnormal.py          # split-normal VB fitting
    splitt.py               # split-t VB fitting
    studentt.py             # symmetric Student-t VB fitting
  models/
    betareg.py              # beta-regression mixture log-density
    gamma.py                # gamma mixture log-density and predictions
    gaussian.py             # Gaussian mixture log-density and predictions
    lognormal.py            # lognormal mixture log-density and predictions
    splitnormal.py          # split-normal mixture log-density and predictions
    splitt.py               # split-t mixture log-density and predictions
    studentt.py             # symmetric Student-t mixture log-density and predictions
    poisson.py              # early migration kernel
    negbin.py               # early migration kernel
scripts/
  BetaReg_config.py         # default Rajan beta-regression specification
  Gaussian_config.py        # default S&P 500 Gaussian mixture specification
  compare_continuous_models.py
  run_betareg_rajan.py      # Rajan BetaReg command-line runner
  run_gaussian_sp500.py     # command-line runner
tests/
  test_*.py                 # focused migration tests
data/
  Rajan.csv
  sp500_1990-2009.csv
  sp500_1990-2009_calendar.csv
```


## Literature

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

## Data

The default Gaussian mixture script uses:

```text
data/sp500_1990-2009_calendar.csv
```
