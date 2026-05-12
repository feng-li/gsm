"""Split-normal mixture specification for the Python migration.

This mirrors the S&P 500 asymmetric normal setup in
``MatlabCode/projects/asymStudT/MainSMODAsymNorm.m`` while using the Python VB
scaffold instead of the MATLAB MCMC/Newton updater.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

PYTHON_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

from gsm.config import FitConfig, sp500_splitnormal_mixture_setting
from gsm.data import load_csv_dataset


@dataclass(frozen=True)
class SplitNormalRunDefaults:
    """Selected defaults from ``MainSMODAsymNorm.m``."""

    seed: int = 10000
    n_iter: int = 100000
    n_components: int = 1
    standardize: int = 2
    n_draw_for_eval: int = 1000


RUN_DEFAULTS = SplitNormalRunDefaults()


MODEL = sp500_splitnormal_mixture_setting(n_components=RUN_DEFAULTS.n_components)


FIT = FitConfig(
    seed=RUN_DEFAULTS.seed,
    max_iter=RUN_DEFAULTS.n_iter,
    learning_rate=1e-2,
    n_restarts=3,
    tol=1e-6,
    coefficient_prior_scale=10.0,
    use_ard=False,
    ard_shape=1e-2,
    ard_rate=1e-2,
    n_elbo_samples=8,
    n_predictive_samples=100,
    posterior_init_log_std=-5.0,
)


def load_dataset(data_path: str | Path):
    """Load a caller-provided CSV for the split-normal mixture model."""

    return load_csv_dataset(
        data_path,
        response_column="Returns",
        add_constant=MODEL.add_constant,
    )


if __name__ == "__main__":
    print(f"Model: {MODEL.model_name}, components={MODEL.n_components}")
    print(f"Features: {MODEL.feature_names}")
    print(f"Links: {MODEL.link_types}")
