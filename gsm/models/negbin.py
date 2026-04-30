"""Negative-binomial model kernel using the MATLAB mean/dispersion parameterization."""

import jax.numpy as jnp
from jax.scipy.special import gammaln


def log_prob(y, mean, dispersion):
    """Log PMF for y ~ NegBin(phi, phi / (phi + mu))."""

    y = jnp.asarray(y).reshape((-1,))
    mu = jnp.asarray(mean).reshape((-1,))
    phi = jnp.asarray(dispersion).reshape((-1,))
    return jnp.sum(
        gammaln(y + phi)
        - gammaln(phi)
        - gammaln(y + 1.0)
        + phi * (jnp.log(phi) - jnp.log(phi + mu))
        + y * (jnp.log(mu) - jnp.log(phi + mu))
    )

