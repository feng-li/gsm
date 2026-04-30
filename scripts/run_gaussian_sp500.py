"""Run the Gaussian mixture VB scaffold on the default S&P 500 CSV."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PYTHON_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

from gsm.variational import fit_variational
from gsm.evaluation import fit_heldout_gaussian_mixture_lpds
from scripts.Gaussian_config import FIT, MODEL, load_default_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=FIT.learning_rate)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument(
        "--coefficient-prior-scale",
        type=float,
        default=FIT.coefficient_prior_scale,
    )
    parser.add_argument(
        "--ard",
        action="store_true",
        default=FIT.use_ard,
        help="use ARD shrinkage for non-constant covariates",
    )
    parser.add_argument("--ard-shape", type=float, default=FIT.ard_shape)
    parser.add_argument("--ard-rate", type=float, default=FIT.ard_rate)
    parser.add_argument("--elbo-samples", type=int, default=FIT.n_elbo_samples)
    parser.add_argument("--predictive-samples", type=int, default=FIT.n_predictive_samples)
    parser.add_argument(
        "--posterior-init-log-std",
        type=float,
        default=FIT.posterior_init_log_std,
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="hold out the last fraction of rows and report train/test ELPD",
    )
    args = parser.parse_args()

    dataset = load_default_dataset()
    fit = replace(
        FIT,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        n_restarts=args.restarts,
        coefficient_prior_scale=args.coefficient_prior_scale,
        use_ard=args.ard,
        ard_shape=args.ard_shape,
        ard_rate=args.ard_rate,
        n_elbo_samples=args.elbo_samples,
        n_predictive_samples=args.predictive_samples,
        posterior_init_log_std=args.posterior_init_log_std,
    )
    if args.holdout_fraction:
        heldout = fit_heldout_gaussian_mixture_lpds(
            dataset,
            MODEL,
            fit,
            test_size=args.holdout_fraction,
        )
        result = heldout.train_result
        print(f"rows: {dataset.y.shape[0]}")
        print(f"train rows: {heldout.train_indices.shape[0]}")
        print(f"test rows: {heldout.test_indices.shape[0]}")
        print(f"components: {MODEL.n_components}")
        print(f"ard: {fit.use_ard}")
        print(f"elbo samples: {fit.n_elbo_samples}")
        print(f"predictive samples: {heldout.test_score.n_posterior_samples}")
        print(f"iterations: {len(result.elbo_history)}")
        print(f"converged: {result.converged}")
        print(f"initial objective: {result.elbo_history[0]:.6f}")
        print(f"final objective: {result.elbo_history[-1]:.6f}")
        print(f"train elpd: {heldout.train_score.elpd:.6f}")
        print(f"train mean elpd: {heldout.train_score.mean_elpd:.6f}")
        print(f"test elpd: {heldout.test_score.elpd:.6f}")
        print(f"test mean elpd: {heldout.test_score.mean_elpd:.6f}")
        return

    result = fit_variational(dataset, MODEL, fit)
    print(f"rows: {dataset.y.shape[0]}")
    print(f"components: {MODEL.n_components}")
    print(f"ard: {fit.use_ard}")
    print(f"elbo samples: {fit.n_elbo_samples}")
    print(f"iterations: {len(result.elbo_history)}")
    print(f"converged: {result.converged}")
    print(f"initial objective: {result.elbo_history[0]:.6f}")
    print(f"final objective: {result.elbo_history[-1]:.6f}")


if __name__ == "__main__":
    main()
