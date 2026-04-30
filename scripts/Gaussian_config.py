"""Gaussian mixture specification for the Python migration.

This file mirrors three parts of the MATLAB code:

1. Generic run defaults from ``MatlabCode/GSM.m`` before line 90.
2. The Gaussian model shape from ``MatlabCode/models/heteroGauss/HeteroGauss.m``.
3. The S&P 500 covariate layout from ``MainSMODAsymStudT.m``.

The old MATLAB inference settings are kept as references only. The Python target
is variational Bayes, not Metropolis-Hastings with Newton updates.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

PYTHON_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

from gsm.config import FitConfig, sp500_gaussian_mixture_setting
from gsm.data import load_csv_dataset


@dataclass(frozen=True)
class GSMRunDefaults:
    """Generic defaults copied from the user-input block in ``GSM.m``."""

    # Model/data settings
    standardize: int = 1
    use_all_covs: bool = False

    # Old MATLAB MCMC settings, retained only for comparison/reference.
    n_iter: int = 10000
    burn_in_percent: int = 10
    min_n_alloc: int = 0
    alloc_method: str = "kmeans"
    plot_traj: bool = False

    # Model evaluation settings
    n_cross: int = 1
    sample_method: str = "systematic"
    n_last: int = 0
    plot_qq: bool = False
    plot_marginal: bool = False
    plot_seq_lpds: bool = False
    n_draw_for_eval: int = 10
    include_full_run: bool = False

    # Misc settings
    seed: int = 10000
    monitor_iter: bool = False
    n_x_grid: int = 100
    n_y_grid: int = 500
    print_results: bool = True
    plot_mix: bool = False


RUN_DEFAULTS = GSMRunDefaults()


MODEL = sp500_gaussian_mixture_setting(n_components=3)


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


DATA_PATH = PYTHON_CODE_ROOT / "data" / MODEL.data_file_name


def load_default_dataset():
    """Load the default S&P 500 CSV for the Gaussian mixture model."""

    return load_csv_dataset(
        DATA_PATH,
        response_column="Returns",
        add_constant=MODEL.add_constant,
    )


def as_dict():
    """Return a simple serializable view for scripts and notebooks."""

    return {
        "data_path": str(DATA_PATH),
        "run_defaults": RUN_DEFAULTS,
        "model": MODEL,
        "fit": FIT,
    }


if __name__ == "__main__":
    config = as_dict()
    print(f"Data: {config['data_path']}")
    print(f"Model: {MODEL.model_name}, components={MODEL.n_components}")
    print(f"Features: {MODEL.feature_names}")
    print(f"Links: {MODEL.link_types}")
