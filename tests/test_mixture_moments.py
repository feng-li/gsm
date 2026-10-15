import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from gsm.models.mixture import mixture_moments


def test_centered_mixture_moments_preserve_small_variance():
    mean, variance = mixture_moments(
        jnp.array([[.25, .75]]), jnp.array([[1e8 - 1., 1e8 + 1.]]), jnp.ones((1, 2)),
    )
    np.testing.assert_allclose(mean, [1e8 + .5])
    np.testing.assert_allclose(variance, [1.75])


@pytest.mark.parametrize("module_name, class_name, coefficients, expected, extras", [
    ("gaussian", "GaussianMixtureParams", {"mean_coef": 1e8, "log_variance_coef": 0.}, 1., {}),
    ("studentt", "StudentTMixtureParams", {"mean_coef": 1e8, "df_coef": np.log(10.),
                                          "scale_coef": 0.}, 1.25, {}),
    ("splitt", "SplitTMixtureParams", {"mean_coef": 1e8, "df_coef": np.log(10.),
                                      "scale_coef": 0., "skewness_coef": 0.}, 1.25, {}),
    ("splitnormal", "SplitNormalMixtureParams", {"mean_coef": 1e8, "scale_coef": 0.,
                                                 "skewness_coef": 0.}, 1., {}),
    ("lognormal", "LogNormalMixtureParams", {"mean_coef": np.log(1e8), "scale_coef": 0.},
     1., {"parameterization": "response"}),
    ("gamma", "GammaMixtureParams", {"mean_coef": np.log(1e8), "variance_coef": 0.}, 1., {}),
    ("poisson", "PoissonMixtureParams", {"mean_coef": np.log(1e16)}, 1e16, {}),
    ("negbin", "NegBinMixtureParams", {"mean_coef": np.log(1e16),
                                      "dispersion_coef": np.log(1e30)}, 1e16 + 100., {}),
    ("genpoisson", "GenPoissonMixtureParams", {"mean_coef": np.log(1e16),
                                               "dispersion_coef": np.log(1e-30)},
     1e16 * (1. + 1e-14)**2, {}),
    ("binomial", "BinomialMixtureParams", {"mean_coef": 0.}, 2.5e15, {"trials": [1e16]}),
    ("betabinomial", "BetaBinMixtureParams", {"mean_coef": 0.,
                                             "dispersion_coef": np.log(1e30)},
     2.5e15 * (1. + 1e-14), {"trials": [1e16]}),
    ("betareg", "BetaRegMixtureParams", {"mean_coef": 0., "dispersion_coef": np.log(1e20)},
     .25 / (1. + 1e20), {}),
])
def test_all_model_families_use_stable_variance(module_name, class_name, coefficients, expected, extras):
    module = importlib.import_module("gsm.models." + module_name)
    params = getattr(module, class_name)(
        **{name: jnp.array([[value]]) for name, value in coefficients.items()},
        gating_coef=jnp.zeros((0, 1)),
    )
    designs = [jnp.ones((1, 1))] * (len(coefficients) + 1)
    _, variance = module.predict_mean_variance(params, *designs, **extras)
    np.testing.assert_allclose(variance, [expected], rtol=1e-12, atol=0.)
