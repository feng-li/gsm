"""Python migration skeleton for Generalized Smooth Mixture models."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

__all__ = []
