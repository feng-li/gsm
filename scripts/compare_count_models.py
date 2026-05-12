"""Compare mdvisits Poisson and negative-binomial mixture VB fits."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PYTHON_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

from gsm.config import (
    FitConfig,
    mdvisits_negbin_mixture_setting,
    mdvisits_poisson_mixture_setting,
)
from gsm.data import load_csv_dataset, subset_dataset
from gsm.evaluation import HeldoutLPDSResult, fit_heldout_model_lpds


DATA_PATH = PYTHON_CODE_ROOT / "data" / "mdvisits_reduced.csv"


@dataclass(frozen=True)
class ModelComparison:
    label: str
    model_name: str
    components: int
    iterations: int
    converged: bool
    initial_objective: float
    final_objective: float
    objective_improvement: float
    train_elpd: float
    test_elpd: float
    mean_test_elpd: float
    pred_mean_min: float
    pred_mean_max: float
    pred_variance_min: float
    pred_variance_max: float
    mean_max_responsibility: float
    mean_responsibility_entropy: float
    heldout: HeldoutLPDSResult


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare mdvisits Poisson and NegBin GSM mixtures by held-out ELPD.",
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--elbo-samples", type=int, default=8)
    parser.add_argument("--predictive-samples", type=int, default=100)
    parser.add_argument("--posterior-init-log-std", type=float, default=-5.0)
    parser.add_argument("--ard", action="store_true", help="use ARD in both fits")
    parser.add_argument("--ard-shape", type=float, default=1e-2)
    parser.add_argument("--ard-rate", type=float, default=1e-2)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="use only the first N rows; 0 uses all rows",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_csv_dataset(
        args.data,
        response_column="numvisit",
        add_constant=False,
        date_column=None,
    )
    if args.max_rows and dataset.y.shape[0] > args.max_rows:
        dataset = subset_dataset(dataset, np.arange(args.max_rows))

    fit = FitConfig(
        seed=args.seed,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        n_restarts=args.restarts,
        tol=0.0,
        use_ard=args.ard,
        ard_shape=args.ard_shape,
        ard_rate=args.ard_rate,
        n_elbo_samples=args.elbo_samples,
        n_predictive_samples=args.predictive_samples,
        posterior_init_log_std=args.posterior_init_log_std,
    )

    models = [
        ("poisson", mdvisits_poisson_mixture_setting(args.components)),
        ("negbin", mdvisits_negbin_mixture_setting(args.components)),
    ]
    comparisons = [
        _run_model(label, setting, dataset, fit, args)
        for label, setting in models
    ]

    _print_run_header(args, dataset, comparisons)
    _print_table(comparisons)

    if args.output_csv is not None:
        _write_csv(args.output_csv, comparisons)
        print()
        print(f"wrote: {args.output_csv}")


def _run_model(label, setting, dataset, fit: FitConfig, args) -> ModelComparison:
    heldout = fit_heldout_model_lpds(
        dataset,
        setting,
        fit,
        test_size=args.holdout_fraction,
        n_predictive_samples=args.predictive_samples,
        seed=args.seed + 10000,
    )
    history = heldout.train_result.elbo_history
    pred_mean = _finite_range(heldout.train_result.predictive_mean)
    pred_var = _finite_range(heldout.train_result.predictive_variance)
    resp_max, resp_entropy = _responsibility_summary(heldout.train_result.responsibilities)
    return ModelComparison(
        label=label,
        model_name=setting.model_name,
        components=setting.n_components,
        iterations=len(history),
        converged=heldout.train_result.converged,
        initial_objective=float(history[0]),
        final_objective=float(history[-1]),
        objective_improvement=float(history[-1] - history[0]),
        train_elpd=heldout.train_score.elpd,
        test_elpd=heldout.test_score.elpd,
        mean_test_elpd=heldout.test_score.mean_elpd,
        pred_mean_min=pred_mean[0],
        pred_mean_max=pred_mean[1],
        pred_variance_min=pred_var[0],
        pred_variance_max=pred_var[1],
        mean_max_responsibility=resp_max,
        mean_responsibility_entropy=resp_entropy,
        heldout=heldout,
    )


def _finite_range(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.nan, np.nan
    return float(np.min(array[finite])), float(np.max(array[finite]))


def _responsibility_summary(values) -> tuple[float, float]:
    resp = np.asarray(values, dtype=float)
    if resp.size == 0:
        return np.nan, np.nan
    clipped = np.clip(resp, 1e-300, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return float(np.mean(np.max(resp, axis=1))), float(np.mean(entropy))


def _print_run_header(
    args,
    dataset,
    comparisons: list[ModelComparison],
) -> None:
    y = dataset.y.reshape(-1)
    print(f"data: {args.data}")
    print(f"rows: {dataset.y.shape[0]}")
    print(f"train rows: {comparisons[0].heldout.train_indices.shape[0]}")
    print(f"test rows: {comparisons[0].heldout.test_indices.shape[0]}")
    print(f"components: {args.components}")
    print(f"response mean: {float(np.mean(y)):.6f}")
    print(f"response variance: {float(np.var(y, ddof=1)):.6f}")
    print(f"elbo samples: {args.elbo_samples}")
    print(f"predictive samples: {args.predictive_samples}")
    print(f"ard: {bool(args.ard)}")
    if args.max_rows:
        print(f"max rows: {args.max_rows}")
    print()


def _print_table(comparisons: list[ModelComparison]) -> None:
    print(
        "model     kernel  comp iter conv  final_obj    test_elpd  mean_test  "
        "pred_mean_range     pred_var_range      max_resp  resp_entropy"
    )
    for item in comparisons:
        print(
            f"{item.label:<9} "
            f"{item.model_name:<7} "
            f"{item.components:>4} "
            f"{item.iterations:>4} "
            f"{str(item.converged):<5} "
            f"{item.final_objective:>10.4f} "
            f"{item.test_elpd:>11.4f} "
            f"{item.mean_test_elpd:>10.4f} "
            f"[{item.pred_mean_min:.4g}, {item.pred_mean_max:.4g}] "
            f"[{item.pred_variance_min:.4g}, {item.pred_variance_max:.4g}] "
            f"{item.mean_max_responsibility:>9.4f} "
            f"{item.mean_responsibility_entropy:>12.4f}"
        )


def _write_csv(path: Path, comparisons: list[ModelComparison]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "kernel",
                "components",
                "iterations",
                "converged",
                "initial_objective",
                "final_objective",
                "objective_improvement",
                "train_elpd",
                "test_elpd",
                "mean_test_elpd",
                "pred_mean_min",
                "pred_mean_max",
                "pred_variance_min",
                "pred_variance_max",
                "mean_max_responsibility",
                "mean_responsibility_entropy",
            ],
        )
        writer.writeheader()
        for item in comparisons:
            writer.writerow(
                {
                    "model": item.label,
                    "kernel": item.model_name,
                    "components": item.components,
                    "iterations": item.iterations,
                    "converged": item.converged,
                    "initial_objective": item.initial_objective,
                    "final_objective": item.final_objective,
                    "objective_improvement": item.objective_improvement,
                    "train_elpd": item.train_elpd,
                    "test_elpd": item.test_elpd,
                    "mean_test_elpd": item.mean_test_elpd,
                    "pred_mean_min": item.pred_mean_min,
                    "pred_mean_max": item.pred_mean_max,
                    "pred_variance_min": item.pred_variance_min,
                    "pred_variance_max": item.pred_variance_max,
                    "mean_max_responsibility": item.mean_max_responsibility,
                    "mean_responsibility_entropy": item.mean_responsibility_entropy,
                }
            )


if __name__ == "__main__":
    main()
