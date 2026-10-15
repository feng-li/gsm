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
    response: np.ndarray | None = None
    origin: np.ndarray | None = None
    date: tuple[str, ...] | None = None
    lead: np.ndarray | None = None

    def __post_init__(self) -> None:
        lpd = np.asarray(self.lpd, dtype=float)
        features = np.asarray(self.features, dtype=float)
        response = None if self.response is None else np.asarray(self.response, dtype=float)
        feature_mean = (
            None if self.feature_mean is None else np.asarray(self.feature_mean, dtype=float)
        )
        feature_sd = None if self.feature_sd is None else np.asarray(self.feature_sd, dtype=float)
        origin = None if self.origin is None else np.asarray(self.origin)
        lead = None if self.lead is None else np.asarray(self.lead, dtype=int)
        if lpd.ndim != 2:
            raise ValueError("lpd must be a 2D matrix")
        if features.ndim != 2:
            raise ValueError("features must be a 2D matrix")
        if lpd.shape[0] != features.shape[0]:
            raise ValueError("lpd and features must have the same number of rows")
        if np.any(np.isnan(lpd) | np.isposinf(lpd)):
            raise ValueError("lpd must not contain NaN or positive infinity")
        if not np.all(np.any(np.isfinite(lpd), axis=1)):
            raise ValueError("each lpd row must contain at least one finite value")
        if response is not None and response.shape[0] != lpd.shape[0]:
            raise ValueError("response length must match lpd rows")
        if self.model_names is not None and len(self.model_names) != lpd.shape[1]:
            raise ValueError("model_names length must match lpd columns")
        if self.feature_names is not None and len(self.feature_names) != features.shape[1]:
            raise ValueError("feature_names length must match feature columns")
        for names in (self.model_names, self.feature_names):
            if names is not None and len(set(names)) != len(names):
                raise ValueError("column names must be unique")
        if feature_mean is not None and feature_mean.shape != (features.shape[1],):
            raise ValueError("feature_mean length must match feature columns")
        if feature_sd is not None and feature_sd.shape != (features.shape[1],):
            raise ValueError("feature_sd length must match feature columns")
        if origin is not None and origin.shape[0] != lpd.shape[0]:
            raise ValueError("origin length must match lpd rows")
        if self.date is not None and len(self.date) != lpd.shape[0]:
            raise ValueError("date length must match lpd rows")
        if lead is not None and (lead.shape != (lpd.shape[0],) or np.any(lead < 1)):
            raise ValueError("lead must contain one positive horizon step per row")
        object.__setattr__(self, "lpd", lpd)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_sd", feature_sd)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "lead", lead)
