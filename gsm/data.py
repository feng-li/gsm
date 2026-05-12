"""Data loading and splitting helpers."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class Dataset:
    y: np.ndarray
    X: np.ndarray
    y_name: str
    x_names: tuple[str, ...]
    date: tuple[str, ...] | None = None


def subset_dataset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    """Return a row subset while preserving column metadata."""

    indices = np.asarray(indices, dtype=int)
    date = None if dataset.date is None else tuple(np.asarray(dataset.date)[indices])
    return Dataset(
        y=dataset.y[indices],
        X=dataset.X[indices],
        y_name=dataset.y_name,
        x_names=dataset.x_names,
        date=date,
    )


def load_mat_dataset(path: str | Path) -> Dataset:
    """Load a MATLAB GSM data file with y, X, yName, and XName variables."""

    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    missing = {"y", "X", "yName", "XName"} - set(raw)
    if missing:
        raise KeyError(f"missing required MATLAB variables: {sorted(missing)}")

    y = np.asarray(raw["y"], dtype=float)
    X = np.asarray(raw["X"], dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if X.ndim != 2:
        raise ValueError("X must be a 2D design matrix")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have the same number of observations")

    return Dataset(
        y=y,
        X=X,
        y_name=_matlab_string(raw["yName"]),
        x_names=tuple(_cellstr(raw["XName"])),
    )


def load_csv_dataset(
    path: str | Path,
    response_column: str,
    feature_columns: tuple[str, ...] | None = None,
    date_column: str | None = "Date",
    drop_columns: tuple[str, ...] = ("DateDecimal",),
    add_constant: bool = False,
    constant_name: str = "Const",
) -> Dataset:
    """Load a CSV dataset without pandas.

    If ``feature_columns`` is omitted, every numeric column except the response,
    date column, and dropped columns is used as a covariate.
    """

    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("CSV file has no data rows")
    if response_column not in rows[0]:
        raise KeyError(f"missing response column: {response_column}")

    if feature_columns is None:
        excluded = {response_column, *drop_columns}
        if date_column is not None:
            excluded.add(date_column)
        feature_columns = tuple(
            name for name in rows[0] if name not in excluded and _all_float(rows, name)
        )

    y = np.asarray([float(row[response_column]) for row in rows], dtype=float)[:, None]
    X = np.asarray(
        [[float(row[column]) for column in feature_columns] for row in rows],
        dtype=float,
    )
    x_names = tuple(feature_columns)
    if add_constant:
        if constant_name in x_names:
            raise ValueError(f"constant column already exists: {constant_name}")
        X = np.column_stack([np.ones(X.shape[0]), X])
        x_names = (constant_name, *x_names)

    date = None
    if date_column is not None and date_column in rows[0]:
        date = tuple(row[date_column] for row in rows)

    return Dataset(
        y=y,
        X=X,
        y_name=response_column,
        x_names=x_names,
        date=date,
    )


def standardize_covariates(X: np.ndarray, method: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize like MATLAB ``StandardizeCovs.m``.

    Constant columns are left unchanged and receive ``nan`` standardization constants.
    """

    X = np.asarray(X, dtype=float).copy()
    c1 = np.full(X.shape[1], np.nan)
    c2 = np.full(X.shape[1], np.nan)
    not_constant = np.std(X, axis=0, ddof=1) != 0

    if not np.any(not_constant) or method == 0:
        return X, c1, c2

    if method == 1:
        c1 = np.mean(X, axis=0)
        c2 = np.std(X, axis=0, ddof=1)
        X[:, not_constant] = (X[:, not_constant] - c1[not_constant]) / c2[not_constant]
    elif method == 2:
        c1 = np.min(X, axis=0)
        c2 = np.max(X, axis=0)
        scale = 2.0 / (c2[not_constant] - c1[not_constant])
        offset = 1.0 - scale * c2[not_constant]
        X[:, not_constant] = X[:, not_constant] * scale + offset
    elif method == 3:
        c1 = np.mean(X, axis=0)
        c2 = np.ones(X.shape[1])
        X[:, not_constant] = X[:, not_constant] - c1[not_constant]
    else:
        raise ValueError(f"unknown standardization method: {method}")

    c1[~not_constant] = np.nan
    c2[~not_constant] = np.nan
    return X, c1, c2


def train_test_indices(
    n_obs: int,
    test_size: float = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test integer indices using scikit-learn's splitter."""

    from sklearn.model_selection import train_test_split

    indices = np.arange(n_obs)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def kfold_indices(
    n_obs: int,
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return K-fold train/test index pairs using scikit-learn."""

    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(np.arange(n_obs))]


def _all_float(rows: list[dict[str, str]], column: str) -> bool:
    try:
        for row in rows:
            float(row[column])
    except (TypeError, ValueError):
        return False
    return True


def _matlab_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return "".join(value.astype(str).ravel())
    return str(value)


def _cellstr(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [_matlab_string(item) for item in value.ravel()]
    if isinstance(value, (list, tuple)):
        return [_matlab_string(item) for item in value]
    return [_matlab_string(value)]
