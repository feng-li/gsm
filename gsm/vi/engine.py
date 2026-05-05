"""Shared mean-field variational optimization helpers."""

from typing import Any, Callable

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


def initialize_mean_field_variational_params(
    mean_tree: dict[str, jnp.ndarray],
    init_log_std: float,
) -> dict[str, dict[str, jnp.ndarray]]:
    log_std_tree = jax.tree_util.tree_map(
        lambda value: jnp.full_like(value, init_log_std),
        mean_tree,
    )
    return {"mean": mean_tree, "log_std": log_std_tree}


def monte_carlo_variational_elbo(
    variational_tree: dict[str, dict[str, jnp.ndarray]],
    noise_tree: dict[str, jnp.ndarray],
    sample_log_joint: Callable[[dict[str, jnp.ndarray]], jnp.ndarray],
) -> jnp.ndarray:
    sample_tree = sample_param_trees(
        variational_tree["mean"],
        variational_tree["log_std"],
        noise_tree,
    )
    log_joint = jax.vmap(sample_log_joint)(sample_tree)
    return jnp.mean(log_joint) + mean_field_gaussian_entropy(variational_tree["log_std"])


def sample_posterior_tree(
    posterior: Any,
    params_to_tree: Callable[[Any], dict[str, jnp.ndarray]],
    seed: int,
    n_samples: int,
) -> dict[str, jnp.ndarray]:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    mean_tree = params_to_tree(posterior.mean)
    log_std_tree = params_to_tree(posterior.log_std)
    noise_tree = sample_noise_like(mean_tree, jax.random.PRNGKey(seed), n_samples)
    return sample_param_trees(mean_tree, log_std_tree, noise_tree)


def optimize_restarts(
    fit,
    initialize_variational_params: Callable[[], dict[str, dict[str, jnp.ndarray]]],
    objective_for_noise: Callable[
        [dict[str, jnp.ndarray]],
        Callable[[dict[str, dict[str, jnp.ndarray]]], jnp.ndarray],
    ],
    build_result: Callable[[dict[str, dict[str, jnp.ndarray]], np.ndarray, bool], Any],
):
    if fit.n_elbo_samples < 1:
        raise ValueError("n_elbo_samples must be positive")

    best_result = None
    for restart in range(fit.n_restarts):
        variational_params = initialize_variational_params()
        if restart:
            variational_params["mean"] = jitter_params(
                variational_params["mean"],
                fit.seed + restart,
            )

        noise_tree = sample_noise_like(
            variational_params["mean"],
            jax.random.PRNGKey(fit.seed + 1009 * (restart + 1)),
            fit.n_elbo_samples,
        )
        variational_params, history, converged = optax_maximize(
            variational_params,
            objective_for_noise(noise_tree),
            max_iter=fit.max_iter,
            learning_rate=fit.learning_rate,
            tol=fit.tol,
        )
        result = build_result(variational_params, history, converged)
        if best_result is None or result.elbo_history[-1] > best_result.elbo_history[-1]:
            best_result = result

    if best_result is None:
        raise RuntimeError("no variational optimization runs were executed")
    return best_result
