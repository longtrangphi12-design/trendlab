import numpy as np
import pytest

from trendlab.autocorr import autocorr_flag, autocorr_map, lag1_autocorr
from trendlab.simulate import null_grid


def test_lag1_hand_calculated_line():
    assert lag1_autocorr([1, 2, 3, 4, 5], detrend=False) == pytest.approx(0.4)


def test_lag1_alternating_is_negative():
    assert lag1_autocorr([1, -1, 1, -1], detrend=False) == pytest.approx(-0.75)


def test_lag1_skips_pairs_with_nan():
    x = [1, 2, np.nan, 4, 5]
    assert lag1_autocorr(x, detrend=False) == pytest.approx(0.4)


def test_lag1_constant_or_too_short_is_nan():
    assert np.isnan(lag1_autocorr([3, 3, 3, 3], detrend=False))
    assert np.isnan(lag1_autocorr([3, 3, 3, 3], detrend=True))
    assert np.isnan(lag1_autocorr([1, 2]))


def test_detrending_separates_trend_from_autocorrelation():
    t = np.arange(8)
    x = t + 0.1 * (-1.0) ** t
    assert lag1_autocorr(x, detrend=False) > 0.4
    assert lag1_autocorr(x, detrend=True) == pytest.approx(-0.875)


def test_map_recovers_known_phi():
    g = null_grid(n_time=200, n_lat=10, n_lon=10, phi=0.6, seed=2)
    r1 = autocorr_map(g)
    assert abs(float(r1.mean()) - 0.6) < 0.06


def test_map_shape_and_nan_cell():
    g = null_grid(n_time=24, n_lat=3, n_lon=4, seed=0).copy()
    g.values[:, 0, 0] = np.nan
    r1 = autocorr_map(g)
    assert r1.dims == ("lat", "lon")
    assert np.isnan(r1.isel(lat=0, lon=0).item())
    assert not np.isnan(r1.isel(lat=1, lon=1).item())


def test_flag_threshold_and_nan():
    r1 = np.array([0.1, 0.5, np.nan, 0.3])
    assert list(autocorr_flag(r1, threshold=0.3)) == [False, True, False, False]
