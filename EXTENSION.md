# Extending `gsm` With New Mixture Distributions

This package currently uses a deliberately simple pattern: each distribution
kernel has a small JAX model file, a setting dataclass, a prior builder, a VB
fit path, and optional predictive scoring support. New distributions should
follow that pattern before adding abstractions.

Shared cross-model wiring lives in `gsm/model_registry.py`, common mixture
weighting in `gsm/models/mixture.py`, common feature-prior construction in
`gsm/priors.py`, and common mean-field fitting in `gsm/vi/engine.py`.

Use this guide when adding another GSM/MoE expert distribution such as Gamma,
Beta, Poisson, negative binomial, asymmetric kernels, or another custom density.

## Model Contract

Every mixture distribution should expose the following functions from
`gsm/models/<model_name>.py`:

```python
@dataclass(frozen=True)
class NewMixtureParams:
    feature1_coef: jnp.ndarray
    feature2_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params, X_feature1, X_feature2):
    ...


def component_log_prob(y, feature1, feature2):
    ...


def responsibilities(params, y, X_feature1, X_feature2, Z):
    ...


def log_prob_observations(params, y, X_feature1, X_feature2, Z):
    ...


def log_prob(params, y, X_feature1, X_feature2, Z):
    ...


def predict_mean_variance(params, X_feature1, X_feature2, Z):
    ...
```

The expected shapes are:

- `y`: `(n_obs,)` or `(n_obs, 1)`
- feature design matrices: `(n_obs, n_covariates_for_feature)`
- coefficient matrices: `(n_components, n_covariates_for_feature)`
- `gating_coef`: `(n_components - 1, n_covariates_for_gating)`
- feature values returned by `component_features`: `(n_obs, n_components)`
- `component_log_prob`: `(n_obs, n_components)`
- `log_prob_observations`: `(n_obs,)`

Reuse `gsm.models.mixture.log_mixture_weights` unless the model needs a
different gating construction. The MATLAB GSM convention uses a reference first
component, so the first component has zero gating coefficients.

## JAX Rules

The density kernel must be differentiable by JAX:

- Use `jax.numpy` as `jnp` inside likelihood and prediction code.
- Avoid NumPy, SciPy, Python loops over observations, or mutation inside the
  objective.
- Enforce support with `jnp.where`, not Python `if` checks on arrays.
- Keep constrained parameters positive with link functions and small lower
  bounds where needed.
- Return `-jnp.inf` for observations outside the support when a pointwise
  density is requested.

For example, a positive-response model should handle invalid observations like:

```python
y = jnp.asarray(y).reshape((-1, 1))
log_density = ...
return jnp.where(y > 0.0, log_density, -jnp.inf)
```

The VB data preparation path can still reject invalid responses early with a
plain `ValueError`.

## Implementation Checklist

### 1. Add The Model Kernel

Create:

```text
gsm/models/<model_name>.py
```

Implement:

- parameter dataclass
- `component_features`
- `component_log_prob`
- `responsibilities`
- `log_prob_observations`
- `log_prob`
- `predict_mean_variance`

If analytic predictive moments are not available, either implement a clear
approximation or return `NotImplementedError` and avoid wiring the model into
code paths that require `predictive_mean` and `predictive_variance`.

Also export the module from:

```text
gsm/models/__init__.py
```

### 2. Add A Setting Dataclass

Edit:

```text
gsm/config.py
```

Add a dataclass with the same fields used by the existing models:

```python
@dataclass(frozen=True)
class NewMixtureSetting:
    model_name: str = "NewModel"
    data_file_name: str = "..."
    feature_names: tuple[str, ...] = (...)
    link_types: tuple[str, ...] = (...)
    covs: tuple[tuple[int, ...], ...] = (...)
    covs_mix: tuple[int, ...] = (...)
    on_trial: tuple[tuple[int, ...], ...] = (...)
    on_trial_mix: tuple[int, ...] = ()
    add_constant: bool = False
    n_components: int = 2
    standardize: int = 1
    prior_mean_feat: tuple[float, ...] = (...)
    prior_std_feat: tuple[float, ...] = (...)
    prior_shrink: tuple[float | str, ...] = (...)
    prior_shrink_mix: float | str = "UnitInfo"
    prior_inclusion: tuple[float, ...] = (...)
    prior_inclusion_mix: float = 0.5
```

Use Python's zero-based indexing.

### 3. Add Priors

Edit:

```text
gsm/priors.py
```

Add a coefficient-prior dataclass:

```python
@dataclass(frozen=True)
class NewMixtureCoefficientPriors:
    feature1: GaussianCoefficientPrior
    feature2: GaussianCoefficientPrior
    gating: GaussianCoefficientPrior | None
```

Then add `build_new_mixture_priors(inputs, setting)`. For ordinary feature
priors, use `_build_feature_priors(inputs, setting, specs)` and pass the
returned feature priors into the dataclass. Use
`build_gaussian_coefficient_prior` directly only when a feature needs special
arguments.

Use `build_gating_prior` for the gating coefficients.

Use the MATLAB-style feature prior values from the corresponding `.m` settings
file. The helper converts feature priors to link scale with
`convert_prior_to_link_scale`.

### 4. Add Variational Support

Edit:

```text
gsm/vi/<model_name>.py
```

Add the same small set of functions the current models use:

- `NewMixtureInputs`
- `NewMixtureStandardization`
- `NewMixturePosterior`
- `prepare_new_mixture_inputs`
- `fit_new_mixture_standardization`
- `initialize_new_mixture_params`
- `initialize_new_mixture_variational_params`
- `tree_to_new_params`
- `tree_to_new_posterior`
- `new_mixture_elbo`
- `new_mixture_variational_elbo`
- `sample_new_mixture_posterior`
- `fit_new_mixture_vb`
- `_build_new_variational_result`

The model-specific `fit_new_mixture_vb` wrapper should prepare inputs and
priors, then call `fit_mean_field_mixture_vb(...)` from `gsm/vi/engine.py`.

Finally export the public names from `gsm/variational.py` and register the
model fitter in `gsm/model_registry.py`; `fit_variational` dispatches through
that registry.

The ELBO should have this form:

```python
log_likelihood = new_log_prob(params, y, X_feature1, X_feature2, Z)
log_prior = (
    coefficient_log_prior(...)
    + coefficient_log_prior(...)
    + coefficient_log_prior(...)
)
return log_likelihood + log_prior
```

The variational objective should keep the current mean-field Gaussian pattern:

```python
sample_tree = sample_param_trees(
    variational_tree["mean"],
    variational_tree["log_std"],
    noise_tree,
)
log_joint = jax.vmap(sample_log_joint)(sample_tree)
return jnp.mean(log_joint) + mean_field_gaussian_entropy(variational_tree["log_std"])
```

### 5. Add Predictive Scoring

Edit:

```text
gsm/model_registry.py
```

Add a `ModelAdapter` entry to `MODEL_ADAPTERS` with the model's setting type,
input preparation, standardization, posterior sampling, parameter conversion,
fitter, and pointwise log-probability function.

This enables posterior-sampled held-out ELPD/LPDS through the existing
`fit_heldout_model_lpds` and `predictive_log_score` functions.

### 6. Add Tests

Create:

```text
tests/test_<model_name>.py
```

At minimum, test:

- the setting dataclass matches the MATLAB settings file
- `component_log_prob` matches the analytic density formula
- invalid support points return `-inf` or are rejected by preparation code
- `log_prob` is finite for valid data
- `predict_mean_variance` returns expected values for a one-component case
- `fit_variational` runs on a tiny synthetic dataset
- posterior sampling shapes are correct

Run:

```bash
python -m pytest tests/test_<model_name>.py
python -m pytest
```

## Minimal Skeleton

This is the shortest useful structure for a two-feature mixture:

```python
from dataclasses import dataclass

import jax.numpy as jnp
from jax.nn import logsumexp

from gsm.links import inverse_link
from gsm.models.mixture import log_mixture_weights


@dataclass(frozen=True)
class NewMixtureParams:
    location_coef: jnp.ndarray
    scale_coef: jnp.ndarray
    gating_coef: jnp.ndarray


def component_features(params, X_location, X_scale):
    location = inverse_link(X_location @ params.location_coef.T, "identity")
    scale = inverse_link(X_scale @ params.scale_coef.T, "log")
    return location, jnp.maximum(scale, jnp.finfo(scale.dtype).tiny)


def component_log_prob(y, location, scale):
    y = jnp.asarray(y).reshape((-1, 1))
    return ...


def log_prob_observations(params, y, X_location, X_scale, Z):
    location, scale = component_features(params, X_location, X_scale)
    return logsumexp(
        log_mixture_weights(params.gating_coef, Z)
        + component_log_prob(y, location, scale),
        axis=1,
    )


def log_prob(params, y, X_location, X_scale, Z):
    return jnp.sum(log_prob_observations(params, y, X_location, X_scale, Z))
```

## Common Mistakes

- Forgetting that MATLAB or R covariate indices are one-based.
- Using SciPy density functions inside JAX objectives.
- Returning scalar log likelihoods where pointwise scores are needed.
- Applying Python `if` statements to JAX arrays.
- Adding a fitter without registering the model adapter used for scoring.
- Ignoring the distribution support, especially for positive-only responses.
- Initializing log-linked intercepts on the feature scale instead of the link
  scale.

## Current Models To Copy From

- Gaussian mixture: `gsm/models/gaussian.py`
- Split-normal mixture: `gsm/models/splitnormal.py`
- Split-t mixture: `gsm/models/splitt.py`
- LogNorm and LogNormRep: `gsm/models/lognormal.py`

For a simple two-feature positive-response model, `lognormal.py` is the closest
template. For a richer asymmetric continuous model, start from `splitt.py` or
`splitnormal.py`.
