"""Shared mean-field variational optimization helpers."""

from dataclasses import fields, replace
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gsm.priors import expected_gaussian_matrix_log_prob, gaussian_matrix_log_prob


def optax_maximize(
    params,
    objective,
    max_iter: int,
    learning_rate: float,
    tol: float,
    *,
    noise_sampler=None,
    seed: int = 0,
    validation_objective=None,
    validation_interval: int = 1,
    diagnostics: dict | None = None,
):
    """Optimize in compiled blocks, retaining the best finite validation iterate.

    History includes validation checkpoints and ends at the returned parameters.
    Stochastic objectives accept ``(params, noise)`` and fresh noise each step.
    """

    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative")
    if validation_interval < 1:
        raise ValueError("validation_interval must be positive")
    evaluate_and_grad = jax.jit(jax.value_and_grad(validation_objective or objective))
    if noise_sampler is None:
        loss_and_grad = jax.value_and_grad(lambda p, noise: -objective(p))
    else:
        loss_and_grad = jax.value_and_grad(lambda p, noise: -objective(p, noise))
    optimizer = optax.adam(learning_rate)
    initial_score, initial_grad = evaluate_and_grad(params)
    if not bool(_tree_all_finite((params, initial_score))):
        raise FloatingPointError("non-finite initial parameters or objective")
    state = dict(
        params=params,
        best_params=params,
        opt_state=optimizer.init(params),
        key=jax.random.PRNGKey(seed),
        active=jnp.asarray(True),
        score=initial_score,
        best_score=initial_score,
        best_grad_norm=optax.tree.norm(initial_grad),
        step=jnp.asarray(0),
        best_step=jnp.asarray(0),
        stable=jnp.asarray(0),
        failed=jnp.asarray(False),
        checked=jnp.asarray(False),
    )

    def update(state):
        key, sample_key = jax.random.split(state["key"])
        noise = () if noise_sampler is None else noise_sampler(sample_key)
        loss, grad = loss_and_grad(state["params"], noise)
        updates, opt_state = optimizer.update(grad, state["opt_state"], state["params"])
        candidate = optax.apply_updates(state["params"], updates)
        finite = _tree_all_finite((loss, grad, candidate, opt_state))
        step = state["step"] + 1
        checked = finite & ((step % validation_interval == 0) | (step == max_iter))
        score, validation_grad = jax.lax.cond(
            checked,
            lambda: evaluate_and_grad(candidate),
            lambda: (state["score"], jax.tree.map(jnp.zeros_like, candidate)),
        )
        finite = finite & _tree_all_finite((score, validation_grad))
        improved = finite & checked & (score > state["best_score"])
        small_change = jnp.abs(score - state["score"]) <= tol * (1 + jnp.abs(score))
        small_gradient = optax.tree.norm(validation_grad) <= jnp.sqrt(tol)
        best_grad_norm = jnp.where(
            improved, optax.tree.norm(validation_grad), state["best_grad_norm"],
        )
        stable = jnp.where(
            checked, jnp.where(small_change & small_gradient & finite, state["stable"] + 1, 0),
            state["stable"],
        )
        return dict(
            params=jax.tree.map(
                lambda new, old: jnp.where(finite, new, old), candidate, state["params"],
            ),
            best_params=jax.tree.map(
                lambda new, old: jnp.where(improved, new, old), candidate, state["best_params"],
            ),
            opt_state=opt_state,
            key=key,
            active=finite & ~((stable >= 5) & (best_grad_norm <= jnp.sqrt(tol))),
            score=jnp.where(finite, score, state["score"]),
            best_score=jnp.where(improved, score, state["best_score"]),
            best_grad_norm=best_grad_norm,
            step=step,
            best_step=jnp.where(improved, step, state["best_step"]),
            stable=stable,
            failed=~finite,
            checked=checked & finite,
        )

    @partial(jax.jit, static_argnames=("n_steps",))
    def run_block(state, n_steps):
        def step(state, _):
            state = dict(state, checked=jnp.asarray(False))
            state = jax.lax.cond(state["active"], update, lambda state: state, state)
            return state, (state["score"], state["checked"], state["step"])

        return jax.lax.scan(step, state, None, length=n_steps)

    history = [float(initial_score)]
    history_steps = [0]
    for start in range(0, max_iter, 25):
        state, (scores, checked, steps) = run_block(state, min(25, max_iter - start))
        checked = np.asarray(checked)
        history.extend(np.asarray(scores)[checked].tolist())
        history_steps.extend(np.asarray(steps)[checked].tolist())
        if not bool(state["active"]):
            break
    best_step = int(state["best_step"])
    selected = np.asarray(history_steps) <= best_step
    converged = (
        bool(state["stable"] >= 5) and bool(state["best_grad_norm"] <= jnp.sqrt(tol))
        and not bool(state["failed"])
    )
    if diagnostics is not None:
        diagnostics.update(
            n_iter=int(state["step"]), best_iteration=best_step,
            termination_reason=("non-finite update" if bool(state["failed"]) else
                                "converged" if converged else "maximum iterations reached"),
            elbo_iterations=np.asarray(history_steps)[selected],
        )
    return state["best_params"], np.asarray(history)[selected], converged


def _tree_all_finite(tree):
    return jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)
    ]))


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

    if fit.n_restarts < 1:
        raise ValueError("n_restarts must be positive")
    n_validation = getattr(fit, "n_validation_samples", 64)
    if n_validation < 2:
        raise ValueError("n_validation_samples must be at least two")
    best_run = None
    best_rank = (False, -np.inf)
    validation_noise = None
    for restart in range(fit.n_restarts):
        variational_params = initialize_variational_params()
        if restart:
            variational_params["mean"] = jitter_params(
                variational_params["mean"],
                fit.seed + restart,
            )

        if validation_noise is None:
            validation_noise = sample_noise_like(
                variational_params["mean"], jax.random.PRNGKey(fit.seed + 200003),
                n_validation,
            )
        try:
            diagnostics = {}
            variational_params, history, converged = optax_maximize(
                variational_params,
                lambda q, noise: objective_for_noise(noise)(q),
                max_iter=fit.max_iter,
                learning_rate=fit.learning_rate,
                tol=fit.tol,
                noise_sampler=lambda key: sample_noise_like(
                    variational_params["mean"], key, fit.n_elbo_samples,
                ),
                seed=fit.seed + 1009 * (restart + 1),
                validation_objective=objective_for_noise(validation_noise),
                validation_interval=getattr(fit, "validation_interval", 25),
                diagnostics=diagnostics,
            )
        except FloatingPointError:
            continue
        if not np.isfinite(history[-1]) or not bool(_tree_all_finite(variational_params)):
            continue
        rank = (diagnostics.get("termination_reason") != "non-finite update", float(history[-1]))
        if rank > best_rank:
            best_run = (variational_params, history, converged, diagnostics)
            best_rank = rank
            best_params = variational_params

    if best_run is None:
        raise RuntimeError("all variational optimization runs had non-finite objectives")
    best_params, history, converged, diagnostics = best_run
    best_result = build_result(best_params, history, converged)
    if diagnostics:
        if hasattr(best_result, "success"):
            diagnostics.update(
                success=diagnostics["termination_reason"] != "non-finite update",
                message=diagnostics["termination_reason"],
            )
        best_result = replace(best_result, **diagnostics)
    noise_batches = sample_noise_like(
        best_params["mean"], jax.random.PRNGKey(fit.seed + 300007), 8 * n_validation,
    )
    noise_batches = jax.tree.map(
        lambda value: value.reshape((8, n_validation, *value.shape[1:])), noise_batches,
    )
    scores = np.asarray(jax.jit(lambda batches: jax.lax.map(
        lambda noise: objective_for_noise(noise)(best_params), batches,
    ))(noise_batches))
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("non-finite ELBO on independent validation samples")
    return replace(
        best_result,
        validation_elbo=float(np.mean(scores)),
        elbo_mc_error=float(np.std(scores, ddof=1) / np.sqrt(scores.size)),
    )


def fit_mean_field_mixture_vb(
    fit,
    inputs,
    initialize_variational_params: Callable[[], dict[str, dict[str, jnp.ndarray]]],
    variational_elbo: Callable[..., jnp.ndarray],
    coefficient_priors,
    build_result: Callable[[dict[str, dict[str, jnp.ndarray]], np.ndarray, bool], Any],
    extra_args: tuple[Any, ...] = (),
    standardization=None,
    predict_moments=None,
    moment_policy=None,
):
    """Run the shared mean-field Gaussian VB optimization pattern."""

    if fit.n_predictive_samples < 0:
        raise ValueError("n_predictive_samples must be non-negative")
    def objective_for_noise(noise_tree):
        def objective(q):
            value = variational_elbo(
                q,
                inputs,
                *extra_args,
                noise_tree,
                coefficient_prior_scale=fit.coefficient_prior_scale,
                use_ard=fit.use_ard,
                ard_shape=fit.ard_shape,
                ard_rate=fit.ard_rate,
                coefficient_priors=coefficient_priors,
            )
            if coefficient_priors is not None and not fit.use_ard:
                # Replace sampled Gaussian prior terms with their exact expectation.
                for field in fields(coefficient_priors):
                    name = field.name + "_coef"
                    prior = getattr(coefficient_priors, field.name)
                    if prior is None:
                        continue
                    mean, log_std = q["mean"][name], q["log_std"][name]
                    samples = mean + jnp.exp(log_std) * noise_tree[name]
                    sampled_prior = jax.vmap(lambda sample: gaussian_matrix_log_prob(sample, prior))(
                        samples,
                    )
                    value = value + expected_gaussian_matrix_log_prob(mean, log_std, prior)
                    value = value - jnp.mean(sampled_prior)
            return value

        return objective

    result = optimize_restarts(
        fit,
        initialize_variational_params,
        objective_for_noise,
        build_result,
    )
    result = replace(result, standardization=standardization)
    if predict_moments is not None and fit.n_predictive_samples > 0:
        if moment_policy is not None:
            mean, variance, warning = moment_policy(
                result.posterior, result.predictive_mean, result.predictive_variance,
            )
            if warning and not np.any(np.isfinite(mean) | np.isfinite(variance)):
                return replace(
                    result, predictive_mean=mean, predictive_variance=variance,
                    predictive_moment_warning=warning, n_predictive_samples=fit.n_predictive_samples,
                )
        samples = sample_posterior_tree(
            result.posterior,
            lambda params: vars(params),
            fit.seed + 400009,
            fit.n_predictive_samples,
        )
        mean, variance = jax.jit(lambda draws: jax.lax.map(
            predict_moments, draws, batch_size=min(16, fit.n_predictive_samples),
        ))(samples)
        predictive_mean = np.asarray(jnp.mean(mean, axis=0))
        predictive_variance = np.asarray(jnp.mean(variance, axis=0) + jnp.var(mean, axis=0))
        warning = None
        if moment_policy is not None:
            predictive_mean, predictive_variance, warning = moment_policy(
                result.posterior, predictive_mean, predictive_variance,
            )
        result = replace(
            result,
            predictive_mean=predictive_mean,
            predictive_variance=predictive_variance,
            n_predictive_samples=fit.n_predictive_samples,
            predictive_moment_warning=warning,
        )
    return result
