"""Small data containers for FEBAMA workflows."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeriesData:
    """One time series and optional held-out values."""

    x: np.ndarray
    xx: np.ndarray | None = None
    date: tuple[str, ...] | None = None


@dataclass(frozen=True)
class LpdFeatures:
    """Historical component log predictive densities and feature matrix."""

    lpd: np.ndarray
    features: np.ndarray
    feature_mean: np.ndarray | None = None
    feature_sd: np.ndarray | None = None
    model_names: tuple[str, ...] | None = None
    feature_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        lpd = np.asarray(self.lpd, dtype=float)
        features = np.asarray(self.features, dtype=float)
        feature_mean = (
            None if self.feature_mean is None else np.asarray(self.feature_mean, dtype=float)
        )
        feature_sd = None if self.feature_sd is None else np.asarray(self.feature_sd, dtype=float)
        if lpd.ndim != 2:
            raise ValueError("lpd must be a 2D matrix")
        if features.ndim != 2:
            raise ValueError("features must be a 2D matrix")
        if lpd.shape[0] != features.shape[0]:
            raise ValueError("lpd and features must have the same number of rows")
        if self.model_names is not None and len(self.model_names) != lpd.shape[1]:
            raise ValueError("model_names length must match lpd columns")
        if self.feature_names is not None and len(self.feature_names) != features.shape[1]:
            raise ValueError("feature_names length must match feature columns")
        if feature_mean is not None and feature_mean.shape != (features.shape[1],):
            raise ValueError("feature_mean length must match feature columns")
        if feature_sd is not None and feature_sd.shape != (features.shape[1],):
            raise ValueError("feature_sd length must match feature columns")
        object.__setattr__(self, "lpd", lpd)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_sd", feature_sd)
