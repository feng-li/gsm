"""Poisson model kernel."""

import jax.numpy as jnp
from jax.scipy.special import gammaln

from gsm.links import inverse_link


def features_from_coefficients(beta, X, link_type: str | float = "log"):
    return inverse_link(X @ beta, link_type)


def log_prob(y, mean):
    y = jnp.asarray(y).reshape((-1,))
    mean = jnp.asarray(mean).reshape((-1,))
    return jnp.sum(y * jnp.log(mean) - mean - gammaln(y + 1.0))

