"""Compare Gaussian, split-t, and split-normal mixture VB fits."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

PYTHON_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

from gsm.config import (
    FitConfig,
    sp500_gaussian_mixture_setting,
    sp500_splitnormal_mixture_setting,
    sp500_splitt_mixture_setting,
)
from gsm.data import load_csv_dataset
from gsm.evaluation import HeldoutLPDSResult, fit_heldout_model_lpds


DATA_PATH = PYTHON_CODE_ROOT / "data" / "sp500_1990-2009_calendar.csv"


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
    heldout: HeldoutLPDSResult


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare S&P 500 Gaussian, split-t, and split-normal mixtures.",
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument(
        "--splitnormal-components",
        type=int,
        default=None,
        help="override split-normal components; defaults to --components",
    )
    parser.add_argument("--elbo-samples", type=int, default=8)
    parser.add_argument("--predictive-samples", type=int, default=100)
    parser.add_argument("--posterior-init-log-std", type=float, default=-5.0)
    parser.add_argument("--ard", action="store_true", help="use ARD in all three fits")
    parser.add_argument("--ard-shape", type=float, default=1e-2)
    parser.add_argument("--ard-rate", type=float, default=1e-2)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    splitnormal_components = args.splitnormal_components or args.components
    models = [
        ("gaussian", sp500_gaussian_mixture_setting(args.components)),
        ("splitt", sp500_splitt_mixture_setting(args.components)),
        ("splitnormal", sp500_splitnormal_mixture_setting(splitnormal_components)),
    ]
    dataset = load_csv_dataset(
        args.data,
        response_column="Returns",
        add_constant=True,
    )
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

    comparisons = [
        _run_model(label, setting, dataset, fit, args)
        for label, setting in models
    ]

    print(f"data: {args.data}")
    print(f"rows: {dataset.y.shape[0]}")
    print(f"train rows: {comparisons[0].heldout.train_indices.shape[0]}")
    print(f"test rows: {comparisons[0].heldout.test_indices.shape[0]}")
    print(f"elbo samples: {args.elbo_samples}")
    print(f"predictive samples: {args.predictive_samples}")
    print(f"ard: {fit.use_ard}")
    print()
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
        heldout=heldout,
    )


def _print_table(comparisons: list[ModelComparison]) -> None:
    print(
        "model        kernel       comp  iter  conv   initial_obj    final_obj      "
        "improvement   train_elpd    test_elpd   mean_test"
    )
    for item in comparisons:
        print(
            f"{item.label:<12} "
            f"{item.model_name:<11} "
            f"{item.components:>4}  "
            f"{item.iterations:>4}  "
            f"{str(item.converged):<5} "
            f"{item.initial_objective:>13.6f} "
            f"{item.final_objective:>12.6f} "
            f"{item.objective_improvement:>12.6f} "
            f"{item.train_elpd:>12.6f} "
            f"{item.test_elpd:>11.6f} "
            f"{item.mean_test_elpd:>10.6f}"
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
                }
            )


if __name__ == "__main__":
    main()
