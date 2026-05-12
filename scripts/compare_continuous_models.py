"""Compare continuous-response GSM models with held-out ELPD."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PYTHON_CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_CODE_ROOT.parent
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

from gsm.config import (
    FitConfig,
    StudentTMixtureSetting,
    gamma_mixture_setting,
    gamma_rep_mixture_setting,
    lognormal_mixture_setting,
    lognormal_rep_mixture_setting,
    rajan_betareg_mixture_setting,
    sp500_gaussian_mixture_setting,
    sp500_splitnormal_mixture_setting,
    sp500_splitt_mixture_setting,
)
from gsm.data import load_csv_dataset, load_mat_dataset, subset_dataset
from gsm.evaluation import HeldoutLPDSResult, fit_heldout_model_lpds


RETURNS_DATA_PATH = PYTHON_CODE_ROOT / "data" / "sp500_1990-2009_calendar.csv"
POSITIVE_DATA_PATH = REPO_ROOT / "Data" / "ElData.mat"
RAJAN_DATA_PATH = PYTHON_CODE_ROOT / "data" / "Rajan.csv"


@dataclass(frozen=True)
class ModelSpec:
    group: str
    label: str
    setting: object


@dataclass(frozen=True)
class ModelComparison:
    group: str
    label: str
    model_name: str
    data_path: Path
    rows: int
    train_rows: int
    test_rows: int
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
    pred_mean_finite_fraction: float
    pred_variance_min: float
    pred_variance_max: float
    pred_variance_finite_fraction: float
    heldout: HeldoutLPDSResult


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare implemented continuous GSM models by held-out ELPD.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=("all",),
        choices=("all", "returns", "positive", "rajan"),
        help="model groups to run",
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--elbo-samples", type=int, default=8)
    parser.add_argument("--predictive-samples", type=int, default=100)
    parser.add_argument("--posterior-init-log-std", type=float, default=-5.0)
    parser.add_argument("--ard", action="store_true", help="use ARD in all fits")
    parser.add_argument("--ard-shape", type=float, default=1e-2)
    parser.add_argument("--ard-rate", type=float, default=1e-2)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="use only the first N rows; 0 uses all rows",
    )
    parser.add_argument("--returns-data", type=Path, default=RETURNS_DATA_PATH)
    parser.add_argument("--positive-data", type=Path, default=POSITIVE_DATA_PATH)
    parser.add_argument("--rajan-data", type=Path, default=RAJAN_DATA_PATH)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    groups = _selected_groups(args.groups)
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

    comparisons: list[ModelComparison] = []
    for group in groups:
        dataset, data_path = _load_group_dataset(group, args)
        for spec in _group_model_specs(group, args.components):
            comparisons.append(_run_model(spec, dataset, data_path, fit, args))

    _print_run_header(args, comparisons)
    _print_table(comparisons)

    if args.output_csv is not None:
        _write_csv(args.output_csv, comparisons)
        print()
        print(f"wrote: {args.output_csv}")


def _selected_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    if "all" in groups:
        return ("returns", "positive", "rajan")
    return tuple(dict.fromkeys(groups))


def _load_group_dataset(group: str, args):
    if group == "returns":
        dataset = load_csv_dataset(
            args.returns_data,
            response_column="Returns",
            add_constant=True,
        )
        path = args.returns_data
    elif group == "positive":
        dataset = load_mat_dataset(args.positive_data)
        path = args.positive_data
    elif group == "rajan":
        dataset = load_csv_dataset(
            args.rajan_data,
            response_column="debtratio",
            add_constant=False,
        )
        path = args.rajan_data
    else:
        raise ValueError(f"unknown group: {group}")

    if args.max_rows and dataset.y.shape[0] > args.max_rows:
        dataset = subset_dataset(dataset, np.arange(args.max_rows))
    return dataset, Path(path)


def _group_model_specs(group: str, components: int) -> list[ModelSpec]:
    if group == "returns":
        studentt = StudentTMixtureSetting(
            data_file_name="sp500_1990-2009_calendar.csv",
            covs=((0,), tuple(range(10)), tuple(range(10))),
            covs_mix=tuple(range(10)),
            add_constant=True,
            n_components=components,
            standardize=2,
        )
        return [
            ModelSpec(group, "gaussian", sp500_gaussian_mixture_setting(components)),
            ModelSpec(group, "splitnormal", sp500_splitnormal_mixture_setting(components)),
            ModelSpec(group, "splitt", sp500_splitt_mixture_setting(components)),
            ModelSpec(group, "studentt", studentt),
        ]
    if group == "positive":
        return [
            ModelSpec(group, "lognormal", lognormal_mixture_setting(components)),
            ModelSpec(group, "lognormal_rep", lognormal_rep_mixture_setting(components)),
            ModelSpec(group, "gamma", gamma_mixture_setting(components)),
            ModelSpec(group, "gamma_rep", gamma_rep_mixture_setting(components)),
        ]
    if group == "rajan":
        setting = rajan_betareg_mixture_setting(components)
        return [ModelSpec(group, "betareg", setting)]
    raise ValueError(f"unknown group: {group}")


def _run_model(
    spec: ModelSpec,
    dataset,
    data_path: Path,
    fit: FitConfig,
    args,
) -> ModelComparison:
    heldout = fit_heldout_model_lpds(
        dataset,
        spec.setting,
        fit,
        test_size=args.holdout_fraction,
        n_predictive_samples=args.predictive_samples,
        seed=args.seed + 10000,
    )
    history = heldout.train_result.elbo_history
    pred_mean = _finite_range(heldout.train_result.predictive_mean)
    pred_var = _finite_range(heldout.train_result.predictive_variance)
    return ModelComparison(
        group=spec.group,
        label=spec.label,
        model_name=spec.setting.model_name,
        data_path=data_path,
        rows=dataset.y.shape[0],
        train_rows=heldout.train_indices.shape[0],
        test_rows=heldout.test_indices.shape[0],
        components=spec.setting.n_components,
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
        pred_mean_finite_fraction=pred_mean[2],
        pred_variance_min=pred_var[0],
        pred_variance_max=pred_var[1],
        pred_variance_finite_fraction=pred_var[2],
        heldout=heldout,
    )


def _finite_range(values) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.nan, np.nan, 0.0
    return (
        float(np.min(array[finite])),
        float(np.max(array[finite])),
        float(np.mean(finite)),
    )


def _print_run_header(args, comparisons: list[ModelComparison]) -> None:
    print(f"groups: {', '.join(_selected_groups(args.groups))}")
    print(f"models: {len(comparisons)}")
    print(f"components: {args.components}")
    print(f"holdout fraction: {args.holdout_fraction}")
    print(f"elbo samples: {args.elbo_samples}")
    print(f"predictive samples: {args.predictive_samples}")
    print(f"ard: {bool(args.ard)}")
    if args.max_rows:
        print(f"max rows: {args.max_rows}")
    print()


def _print_table(comparisons: list[ModelComparison]) -> None:
    print(
        "group     model           kernel       rows  train  test comp iter conv "
        "final_obj    test_elpd   mean_test  pred_mean_range       pred_var_range"
    )
    for item in comparisons:
        print(
            f"{item.group:<9} "
            f"{item.label:<15} "
            f"{item.model_name:<11} "
            f"{item.rows:>5} "
            f"{item.train_rows:>6} "
            f"{item.test_rows:>5} "
            f"{item.components:>4} "
            f"{item.iterations:>4} "
            f"{str(item.converged):<5} "
            f"{item.final_objective:>11.4f} "
            f"{item.test_elpd:>11.4f} "
            f"{item.mean_test_elpd:>10.4f} "
            f"[{item.pred_mean_min:.4g}, {item.pred_mean_max:.4g}] "
            f"[{item.pred_variance_min:.4g}, {item.pred_variance_max:.4g}]"
        )


def _write_csv(path: Path, comparisons: list[ModelComparison]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "model",
                "kernel",
                "data_path",
                "rows",
                "train_rows",
                "test_rows",
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
                "pred_mean_finite_fraction",
                "pred_variance_min",
                "pred_variance_max",
                "pred_variance_finite_fraction",
            ],
        )
        writer.writeheader()
        for item in comparisons:
            writer.writerow(
                {
                    "group": item.group,
                    "model": item.label,
                    "kernel": item.model_name,
                    "data_path": str(item.data_path),
                    "rows": item.rows,
                    "train_rows": item.train_rows,
                    "test_rows": item.test_rows,
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
                    "pred_mean_finite_fraction": item.pred_mean_finite_fraction,
                    "pred_variance_min": item.pred_variance_min,
                    "pred_variance_max": item.pred_variance_max,
                    "pred_variance_finite_fraction": item.pred_variance_finite_fraction,
                }
            )


if __name__ == "__main__":
    main()
