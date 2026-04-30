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
