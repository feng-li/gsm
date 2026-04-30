import jax.numpy as jnp
import numpy as np

from gsm.links import inverse_link, link


def test_log_link_roundtrip():
    value = jnp.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(inverse_link(link(value, "log"), "log"), value)


def test_logit_link_roundtrip():
    value = jnp.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(inverse_link(link(value, "logit"), "logit"), value)

