# gsm-python

`gsm-python` is the Python migration package for the GSMMatlab codebase. The
current implementation focuses on one working path: covariate-dependent Gaussian
mixture models fitted with variational Bayes in JAX.

The package is intentionally small while the migration is underway. MATLAB MCMC
code is used as a reference for model semantics and validation, but the Python
target is variational inference rather than Metropolis-Hastings with Newton
updates.

## General Purpose

This repository supports research and migration work for Bayesian
mixture-of-experts models for conditional density forecasting. In the original
MATLAB code, the method is framed as Generalized Smooth Mixtures: each mixture
component is an expert distribution, the gating function assigns
covariate-dependent mixture probabilities, and each distributional feature can
also depend on covariates through link functions.

The financial forecasting use case is full predictive distribution estimation,
not only point forecasting. The S&P 500 project scripts use smooth mixtures of
asymmetric Student t and asymmetric normal experts to forecast return
distributions with time-varying scale, skewness, tail behavior, and mixture
probabilities. This follows the mixture-of-experts density-forecasting
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
- MATLAB-compatible covariate standardization modes.
- JAX-compatible link and inverse-link functions.
- Gaussian mixture kernel with identity-linked means, log-linked variances, and
  multinomial-logit gating.
- Mean-field Gaussian variational posterior over Gaussian mixture coefficients.
- Optional ARD shrinkage for non-constant covariates.
- Posterior-sampled held-out ELPD/LPDS for chronological train/test evaluation.
- Early Poisson and negative-binomial model kernels for later migration work.

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

Run chronological held-out ELPD with posterior predictive sampling:

```bash
python scripts/run_gaussian_sp500.py \
  --max-iter 50 \
  --restarts 1 \
  --holdout-fraction 0.2 \
  --elbo-samples 8 \
  --predictive-samples 100
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
  variational.py            # Gaussian-mixture VB fitting
  models/
    gaussian_mixture.py     # Gaussian mixture log-density and predictions
    poisson.py              # early migration kernel
    negbin.py               # early migration kernel
scripts/
  Gaussian_config.py        # default S&P 500 Gaussian mixture specification
  run_gaussian_sp500.py     # command-line runner
tests/
  test_*.py                 # focused migration tests
data/
  Rajan.csv
  sp500_1990-2009.csv
  sp500_1990-2009_calendar.csv
```

## MATLAB Core References

The Python implementation uses these MATLAB files as the main semantic
references:

| MATLAB file | Python migration role |
| --- | --- |
| `../MatlabCode/GSM.m` | Original GSM driver: data setup, priors, MCMC, prediction, and LPDS orchestration. |
| `../MatlabCode/models/heteroGauss/HeteroGauss.m` | Gaussian mixture model specification: features, links, covariates, priors, and evaluation defaults. |
| `../MatlabCode/models/heteroGauss/HeteroGaussPDF.m` | Gaussian component PDF/CDF and component summary statistics. |
| `../MatlabCode/models/heteroGauss/HeteroGaussLogPost.m` | Gaussian log-likelihood/log-posterior reference formula. |
| `../MatlabCode/SetUpData.m` | Data selection, feature-specific design matrices, standardization, and spline expansion. |
| `../MatlabCode/StandardizeCovs.m` | MATLAB standardization modes reproduced by `gsm.data.standardize_covariates`. |
| `../MatlabCode/SetUpPrior.m` | Original feature and gating prior construction, including unit-information priors. |
| `../MatlabCode/convertPrior.m` | Link-scale conversion of elicited feature priors. |
| `../MatlabCode/PredPDFGSM.m` | Mixture predictive PDF/CDF and predictive moments. |
| `../MatlabCode/EvalFitGSM.m` | MATLAB LPDS, normalized residual, predictive summary, and LPDS NSE calculation. |
| `../MatlabCode/setUpCrossVal.m` | Original `orderly`, `systematic`, `random`, and `last` cross-validation splits. |

## Literature Cited In MATLAB Comments

The MATLAB source repeatedly cites the following references. These entries are
kept here as migration context and should be checked against the final papers
before formal publication use.

- Villani, M., Kohn, R. and Giordani, P. (2009). Regression Density Estimation
  using Smooth Adaptive Gaussian Mixtures, Journal of Econometrics.
  Cited in `../MatlabCode/GSM.m`, `../MatlabCode/EvalFitGSM.m`,
  `../MatlabCode/PlotPredGSM.m`, and `../MatlabCode/EvalPredPDFGSM.m`.
- Villani, M., Kohn, R. and Nott, D. (2009). A General Approach to Regression
  Density Estimation for Discrete and Continuous Data.
  Cited in `../MatlabCode/GSM.m`, `../MatlabCode/EvalFitGSM.m`, and
  beta-binomial gradient/Hessian files.
- Kohn, R. and Villani, M. (2009). A General Approach to Regression Density
  Estimation using Smooth Mixtures.
  Cited across model PDF files and project scripts, including
  `../MatlabCode/models/heteroGauss/HeteroGaussPDF.m` and
  `../MatlabCode/projects/discreteGLM/MainSMODPois.m`.
- Kohn, R. and Villani, M. (2009). A General Approach to Regression Density
  Estimation using Smooth Mixtures of Over-Dispersed Models.
  Cited in model log-posterior and gradient/Hessian files, including
  `../MatlabCode/models/heteroGauss/HeteroGaussLogPost.m`.
- Villani, M., Kohn, R. and Giordani, P. (2008). Regression Density Estimation
  using Smooth Adaptive Gaussian Mixtures.
  Cited in Newton proposal and gradient/Hessian files as an earlier working
  paper style reference.
- Li, F., Villani, M. and Kohn, R. (2010). Flexible modeling of conditional
  distributions using smooth mixtures of asymmetric Student t densities. Journal
  of Statistical Planning and Inference, 140(12), 3638-3654.
  https://doi.org/10.1016/j.jspi.2010.04.031.
  This appears as an incomplete JSPI citation in `../MatlabCode/GSM.m`,
  `../MatlabCode/GSM4Sim.m`, and `../MatlabCode/GSMWithEstLink.m`.
- Li, F., Villani, M. and Kohn, R. (2009).
  This appears as an incomplete citation in `../MatlabCode/PriorPredGSM.m`.

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

The CSV includes calendar dates converted from the original MATLAB decimal date
format. The constant/intercept column is not stored in the CSV; it is inserted by
the loader when `add_constant=True`.

## Current Scope

This package does not yet implement full GSMMatlab parity. Missing or incomplete
areas include:

- MATLAB-style g-priors and link-scale prior conversion.
- Structured/block Gaussian variational families.
- Posterior inclusion probabilities beyond ARD shrinkage.
- Spline basis construction.
- MATLAB `orderly`, `systematic`, and `random` cross-validation parity.
- Predictive CDF, PIT residuals, normalized residual diagnostics, and LPDS NSE.
- Parity fixtures comparing Python outputs directly against saved MATLAB runs.

The next migration step should be adding MATLAB-compatible prior construction for
the Gaussian mixture before extending the same VI and evaluation pattern to other
GSM families.
