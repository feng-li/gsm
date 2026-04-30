# GSMMatlab Python Migration Skeleton

This folder is the starting point for the Python migration. It is intentionally small:

- `gsm/data.py` handles `.mat` loading and basic split helpers.
- `gsm/links.py` contains JAX-compatible link and inverse-link functions.
- `gsm/models/` starts model-specific log-probability kernels.
- `gsm/variational.py` is the future variational Bayes entry point.

The first target is not feature parity with the full MATLAB repository. The first target is one
small VB path that can load existing data, evaluate model kernels, optimize an ELBO, and produce
predictive densities.

## Quick Start

```bash
cd PythonCode
python -m pip install -e ".[dev]"
pytest
```

