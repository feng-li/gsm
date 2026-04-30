"""Run the Gaussian mixture VB scaffold on the default S&P 500 CSV."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PYTHON_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

from gsm.variational import fit_variational
from scripts.Gaussian_config import FIT, MODEL, load_default_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=FIT.learning_rate)
    parser.add_argument("--restarts", type=int, default=1)
    args = parser.parse_args()

    dataset = load_default_dataset()
    fit = replace(
        FIT,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        n_restarts=args.restarts,
    )
    result = fit_variational(dataset, MODEL, fit)

    print(f"rows: {dataset.y.shape[0]}")
    print(f"components: {MODEL.n_components}")
    print(f"iterations: {len(result.elbo_history)}")
    print(f"converged: {result.converged}")
    print(f"initial objective: {result.elbo_history[0]:.6f}")
    print(f"final objective: {result.elbo_history[-1]:.6f}")


if __name__ == "__main__":
    main()

