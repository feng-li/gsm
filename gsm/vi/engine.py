"""Shared mean-field variational optimization helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import optax


def optax_maximize(
    params,
    objective,
    max_iter: int,
    learning_rate: float,
    tol: float,
):
    loss_and_grad = jax.value_and_grad(lambda p: -objective(p))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    history: list[float] = []
    converged = False

    for _ in range(1, max_iter + 1):
        loss, grad = loss_and_grad(params)
        history.append(float(-loss))
        if len(history) > 5 and abs(history[-1] - history[-2]) < tol:
            converged = True
            break

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

    return params, np.asarray(history), converged


def sample_noise_like(
    param_tree: dict[str, jnp.ndarray],
    key,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    leaves, tree_def = jax.tree_util.tree_flatten(param_tree)
    keys = jax.random.split(key, len(leaves))
    noise_leaves = [
        jax.random.normal(sample_key, (n_samples, *leaf.shape), dtype=leaf.dtype)
        for sample_key, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(tree_def, noise_leaves)


def sample_param_trees(
    mean_tree: dict[str, jnp.ndarray],
    log_std_tree: dict[str, jnp.ndarray],
    noise_tree: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    return jax.tree_util.tree_map(
        lambda mean, log_std, noise: mean + jnp.exp(log_std) * noise,
        mean_tree,
        log_std_tree,
        noise_tree,
    )


def mean_field_gaussian_entropy(log_std_tree: dict[str, jnp.ndarray]) -> jnp.ndarray:
    log_two_pi_e = jnp.log(2.0 * jnp.pi * jnp.e)
    return sum(
        jnp.sum(log_std + 0.5 * log_two_pi_e)
        for log_std in jax.tree_util.tree_leaves(log_std_tree)
    )


def jitter_params(params: dict[str, jnp.ndarray], seed: int) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: value + jnp.asarray(0.01 * rng.standard_normal(value.shape))
        for name, value in params.items()
    }
