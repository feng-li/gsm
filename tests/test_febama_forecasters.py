import importlib.util

import numpy as np
import pytest

from gsm.febama.distributions import log_prob_matrix
from gsm.febama.forecasters import (
    auto_arima_fore,
    egarch_fore,
    ets_fore,
    garch_fore,
    naive_fore,
    rw_drift_fore,
)


def _series(n=50):
    i = np.arange(n, dtype=float)
    return np.sin(i / 4.0) + 0.03 * i


def _volatile_series(n=100):
    rng = np.random.default_rng(123)
    return 0.1 * rng.standard_normal(n)


def test_naive_and_drift_forecasters_return_gaussian_predictions():
    y = _series()

    predictions = [naive_fore(y, 3), rw_drift_fore(y, 3)]
    lpd = log_prob_matrix(np.asarray([0.1, 0.2, 0.3]), predictions)

    for prediction in predictions:
        assert prediction.name == "gaussian"
        assert prediction.mean().shape == (3,)
        assert prediction.variance().shape == (3,)
        assert np.isfinite(np.asarray(prediction.mean())).all()
        assert np.isfinite(np.asarray(prediction.variance())).all()
        assert np.all(np.asarray(prediction.variance()) > 0.0)
    assert lpd.shape == (3, 2)
    assert np.isfinite(np.asarray(lpd)).all()


def test_simple_forecasters_handle_short_and_constant_series():
    single = naive_fore([2.0], 2)
    constant = rw_drift_fore([1.0, 1.0, 1.0], 2)

    np.testing.assert_allclose(single.mean(), np.asarray([2.0, 2.0]))
    np.testing.assert_allclose(constant.mean(), np.asarray([1.0, 1.0]))
    assert np.all(np.asarray(single.variance()) > 0.0)
    assert np.all(np.asarray(constant.variance()) > 0.0)


def test_forecasters_validate_inputs():
    with pytest.raises(ValueError, match="at least one"):
        naive_fore([], 1)
    with pytest.raises(ValueError, match="finite"):
        rw_drift_fore([1.0, np.nan], 1)
    with pytest.raises(ValueError, match="positive"):
        naive_fore([1.0], 0)


@pytest.mark.skipif(
    importlib.util.find_spec("statsforecast") is None,
    reason="statsforecast is not installed",
)
def test_statsforecast_autoets_and_autoarima_forecasters():
    y = _series(60)

    ets = ets_fore(y, 2, season_length=1)
    arima = auto_arima_fore(
        y,
        2,
        season_length=1,
        seasonal=False,
        max_p=2,
        max_q=2,
        max_order=2,
    )

    assert ets.name == "gaussian"
    assert arima.name == "gaussian"
    assert ets.mean().shape == (2,)
    assert arima.mean().shape == (2,)
    assert np.isfinite(np.asarray(ets.variance())).all()
    assert np.isfinite(np.asarray(arima.variance())).all()


@pytest.mark.skipif(importlib.util.find_spec("arch") is None, reason="arch is not installed")
def test_arch_garch_and_egarch_forecasters():
    y = _volatile_series()

    garch = garch_fore(y, 1)
    egarch = egarch_fore(y, 1)

    assert garch.name == "gaussian"
    assert egarch.name == "gaussian"
    assert garch.mean().shape == (1,)
    assert egarch.mean().shape == (1,)
    assert np.isfinite(np.asarray(garch.variance())).all()
    assert np.isfinite(np.asarray(egarch.variance())).all()
