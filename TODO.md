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
