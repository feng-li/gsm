"""JAX-compatible link functions."""

from __future__ import annotations

import jax.numpy as jnp


def link(value, link_type: str | float):
    """Map model feature values to the linear predictor scale."""

    if not isinstance(link_type, str):
        return value**link_type

    name = link_type.lower()
    if name == "identity":
        return value
    if name == "log":
        return jnp.log(value)
    if name == "logit":
        return jnp.log(value / (1.0 - value))
    if name == "loglog":
        return jnp.log(-jnp.log(value))
    if name in {"comploglog", "cloglog"}:
        return jnp.log(-jnp.log1p(-value))
    if name == "reciprocal":
        return 1.0 / value
    raise ValueError(f"unknown link type: {link_type}")


def inverse_link(eta, link_type: str | float):
    """Map linear predictors to model feature values."""

    if not isinstance(link_type, str):
        return eta ** (1.0 / link_type)

    name = link_type.lower()
    if name == "identity":
        return eta
    if name == "log":
        return jnp.exp(eta)
    if name == "logit":
        return 1.0 / (1.0 + jnp.exp(-eta))
    if name == "loglog":
        return jnp.exp(-jnp.exp(eta))
    if name in {"comploglog", "cloglog"}:
        return -jnp.expm1(-jnp.exp(eta))
    if name == "reciprocal":
        return 1.0 / eta
    raise ValueError(f"unknown link type: {link_type}")

