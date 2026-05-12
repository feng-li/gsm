"""Run the split-t mixture VB scaffold on the default S&P 500 CSV."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PYTHON_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

from gsm.variational import fit_variational
from scripts.SplitT_config import FIT, MODEL, load_dataset


DEFAULT_DATA_PATH = PYTHON_CODE_ROOT / "data" / "sp500_1990-2009_calendar.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=FIT.learning_rate)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument(
        "--ard",
        action="store_true",
        default=FIT.use_ard,
        help="use ARD shrinkage for non-constant covariates",
    )
    parser.add_argument("--ard-shape", type=float, default=FIT.ard_shape)
    parser.add_argument("--ard-rate", type=float, default=FIT.ard_rate)
    parser.add_argument("--elbo-samples", type=int, default=FIT.n_elbo_samples)
    parser.add_argument(
        "--posterior-init-log-std",
        type=float,
        default=FIT.posterior_init_log_std,
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    fit = replace(
        FIT,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        n_restarts=args.restarts,
        use_ard=args.ard,
        ard_shape=args.ard_shape,
        ard_rate=args.ard_rate,
        n_elbo_samples=args.elbo_samples,
        posterior_init_log_std=args.posterior_init_log_std,
    )
    result = fit_variational(dataset, MODEL, fit)
    print(f"rows: {dataset.y.shape[0]}")
    print(f"data: {args.data}")
    print(f"components: {MODEL.n_components}")
    print(f"ard: {fit.use_ard}")
    print(f"elbo samples: {fit.n_elbo_samples}")
    print(f"iterations: {len(result.elbo_history)}")
    print(f"converged: {result.converged}")
    print(f"initial objective: {result.elbo_history[0]:.6f}")
    print(f"final objective: {result.elbo_history[-1]:.6f}")


if __name__ == "__main__":
    main()
