"""Compare Gaussian-mixture VB fits with and without ARD."""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np

PYTHON_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

from gsm.evaluation import HeldoutLPDSResult, fit_heldout_gaussian_mixture_lpds
from scripts.Gaussian_config import FIT, MODEL, load_default_dataset


@dataclass(frozen=True)
class ComparisonSummary:
    label: str
    ard: bool
    iterations: int
    converged: bool
    initial_objective: float
    final_objective: float
    objective_improvement: float
    train_elpd: float
    test_elpd: float
    mean_test_elpd: float
    heldout: HeldoutLPDSResult


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ARD and non-ARD Gaussian-mixture VB fits.",
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=FIT.learning_rate)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=FIT.seed)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--elbo-samples", type=int, default=FIT.n_elbo_samples)
    parser.add_argument("--predictive-samples", type=int, default=FIT.n_predictive_samples)
    parser.add_argument("--ard-shape", type=float, default=FIT.ard_shape)
    parser.add_argument("--ard-rate", type=float, default=FIT.ard_rate)
    parser.add_argument(
        "--posterior-init-log-std",
        type=float,
        default=FIT.posterior_init_log_std,
    )
    parser.add_argument(
        "--shrinkage-threshold",
        type=float,
        default=0.05,
        help="absolute posterior-mean coefficient threshold for ARD shrinkage summary",
    )
    args = parser.parse_args()

    dataset = load_default_dataset()
    base_fit = replace(
        FIT,
        seed=args.seed,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        n_restarts=args.restarts,
        ard_shape=args.ard_shape,
        ard_rate=args.ard_rate,
        n_elbo_samples=args.elbo_samples,
        n_predictive_samples=args.predictive_samples,
        posterior_init_log_std=args.posterior_init_log_std,
    )

    summaries = [
        _run_one("non_ard", False, dataset, base_fit, args),
        _run_one("ard", True, dataset, base_fit, args),
    ]

    print(f"rows: {dataset.y.shape[0]}")
    print(f"components: {MODEL.n_components}")
    print(f"train rows: {summaries[0].heldout.train_indices.shape[0]}")
    print(f"test rows: {summaries[0].heldout.test_indices.shape[0]}")
    print(f"elbo samples: {args.elbo_samples}")
    print(f"predictive samples: {args.predictive_samples}")
    print()
    _print_table(summaries)
    print()
    _print_ard_shrinkage_summary(
        summaries[1].heldout,
        dataset.x_names,
        threshold=args.shrinkage_threshold,
    )


def _run_one(label: str, use_ard: bool, dataset, base_fit, args) -> ComparisonSummary:
    fit = replace(base_fit, use_ard=use_ard)
    heldout = fit_heldout_gaussian_mixture_lpds(
        dataset,
        MODEL,
        fit,
        test_size=args.holdout_fraction,
        n_predictive_samples=args.predictive_samples,
        seed=args.seed + 10000,
    )
    history = heldout.train_result.elbo_history
    return ComparisonSummary(
        label=label,
        ard=use_ard,
        iterations=len(history),
        converged=heldout.train_result.converged,
        initial_objective=float(history[0]),
        final_objective=float(history[-1]),
        objective_improvement=float(history[-1] - history[0]),
        train_elpd=heldout.train_score.elpd,
        test_elpd=heldout.test_score.elpd,
        mean_test_elpd=heldout.test_score.mean_elpd,
        heldout=heldout,
    )


def _print_table(summaries: list[ComparisonSummary]) -> None:
    print(
        "version  ard    iter  conv   initial_obj    final_obj      "
        "improvement   train_elpd    test_elpd   mean_test"
    )
    for summary in summaries:
        print(
            f"{summary.label:<8} "
            f"{str(summary.ard):<6} "
            f"{summary.iterations:>4}  "
            f"{str(summary.converged):<5} "
            f"{summary.initial_objective:>13.6f} "
            f"{summary.final_objective:>12.6f} "
            f"{summary.objective_improvement:>12.6f} "
            f"{summary.train_elpd:>12.6f} "
            f"{summary.test_elpd:>11.6f} "
            f"{summary.mean_test_elpd:>10.6f}"
        )


def _print_ard_shrinkage_summary(
    heldout: HeldoutLPDSResult,
    x_names: tuple[str, ...],
    threshold: float,
) -> None:
    params = heldout.train_result.params
    print(f"ARD shrinkage summary: abs(posterior mean coefficient) < {threshold:g}")
    _print_block_summary("mean", params.mean_coef, MODEL.covs[0], x_names, threshold)
    _print_block_summary(
        "log_variance",
        params.log_variance_coef,
        MODEL.covs[1],
        x_names,
        threshold,
    )
    _print_block_summary("gating", params.gating_coef, MODEL.covs_mix, x_names, threshold)


def _print_block_summary(
    block_name: str,
    coefficients,
    covariate_indices: tuple[int, ...],
    x_names: tuple[str, ...],
    threshold: float,
) -> None:
    coef = np.asarray(coefficients)
    names = tuple(x_names[index] for index in covariate_indices)
    if coef.size == 0:
        print(f"{block_name}: no free coefficients")
        return

    max_abs_by_covariate = np.max(np.abs(coef), axis=0)
    nonconstant = [
        (name, value)
        for name, value in zip(names, max_abs_by_covariate)
        if name.lower() != "const"
    ]
    if not nonconstant:
        print(f"{block_name}: no non-constant covariates")
        return

    shrunk = [name for name, value in nonconstant if value < threshold]
    retained = [(name, value) for name, value in nonconstant if value >= threshold]
    retained_text = ", ".join(f"{name}={value:.4f}" for name, value in retained) or "none"
    print(f"{block_name}: shrunk {len(shrunk)}/{len(nonconstant)}")
    print(f"{block_name}: retained {retained_text}")


if __name__ == "__main__":
    main()
