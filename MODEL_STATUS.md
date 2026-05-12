# GSM Python Model Status

This table tracks migration status for model kernels found in
`MatlabCode/models` and the current Python VB target.

| Model | Python name | Kernel | VB | Held-out ELPD | MATLAB density check | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `heteroGauss` | `GaussianMixtureSetting` | Yes | Yes | Yes | Partial | S&P 500 setting available. |
| `beta/BetaReg` | `BetaRegMixtureSetting` | Yes | Yes | Yes | Partial | Rajan setting available. |
| `LogNorm` | `LogNormalMixtureSetting` | Yes | Yes | Yes | Yes | Shared with LogNormRep. |
| `LogNormRep` | `LogNormalRepMixtureSetting` | Yes | Yes | Yes | Yes | Response-scale parameterization. |
| `gamma/Gamma` | `GammaMixtureSetting` | Yes | Yes | Yes | Yes | Mean/variance parameterization. |
| `gamma/GammaRep` | `GammaRepMixtureSetting` | Yes | Yes | Yes | Python extension | Direct shape/scale parameterization. |
| `studT` | `StudentTMixtureSetting` | Yes | Yes | Yes | Yes | Symmetric Student-t. |
| `AsymNorm` | `SplitNormalMixtureSetting` | Yes | Yes | Yes | Partial | Python name uses split-normal. |
| `AsymStudT` | `SplitTMixtureSetting` | Yes | Yes | Yes | Partial | Python name uses split-t. |
| `pois` | `PoissonMixtureSetting` | Yes | Yes | Yes | Partial | mdvisits setting available; density formula matched. |
| `negBin` | `NegBinMixtureSetting` | Yes | Yes | Yes | Partial | mdvisits setting available; mean/dispersion parameterization matched. |
| `Bin` | Not implemented | No | No | No | No | Needs two-column response `[successes, trials]`. |
| `BetaBin` | Not implemented | No | No | No | No | Needs two-column response `[successes, trials]`. |
| `GenPois` | Not implemented | No | No | No | No | Needs count-mixture design. |
| `GenPoisAlt` | Not implemented | No | No | No | No | Needs `log1` link and count-mixture design. |

## Next Validation Work

- Add fixed-parameter equivalence tests for Gaussian, SplitNormal, SplitT, and
  BetaReg against the MATLAB formulas.
- Add real-data comparison reports for continuous models using
  `scripts/compare_continuous_models.py`.
- Add real-data mdvisits comparison reports for Poisson and NegBin before
  implementing `Bin` and `BetaBin`.
