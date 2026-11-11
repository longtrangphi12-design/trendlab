import numpy as np
import pytest

from trendlab.autocorr import tfpw_mann_kendall, tfpw_map
from trendlab.simulate import null_grid
from trendlab.trend import mann_kendall, trend_map


def _noise(n=24, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def test_never_prewhitening_equals_plain_mann_kendall():
    x = _noise()
    r = tfpw_mann_kendall(x, threshold=np.inf)
    assert r["prewhitened"] is False
    assert r["p"] == pytest.approx(mann_kendall(x)["p"])


def test_forced_prewhitening_loses_one_point():
    x = _noise()
    r = tfpw_mann_kendall(x, threshold=-np.inf)
    assert r["prewhitened"] is True
    assert r["n"] == len(x) - 1


def test_perfect_line_falls_back_to_plain():
    x = np.arange(10) * 0.5            # residual is zero, r1 undefined
    r = tfpw_mann_kendall(x, threshold=-np.inf)
    assert r["prewhitened"] is False
    assert r["p"] == pytest.approx(mann_kendall(x)["p"])


def test_gap_removes_the_missing_step_and_the_step_after_it():
    x = _noise(12, seed=1)
    x[5] = np.nan
    r = tfpw_mann_kendall(x, threshold=-np.inf)
    assert r["n"] == 12 - 3            # first step, the gap, the step after the gap


def test_map_matches_plain_when_never_prewhitening():
    g = null_grid(n_time=24, n_lat=4, n_lon=5, phi=0.4, seed=3)
    out = tfpw_map(g, np.inf)
    assert set(out.data_vars) == {"z", "p", "prewhitened"}
    assert out["p"].dims == ("lat", "lon")
    assert not out["prewhitened"].any()
    np.testing.assert_allclose(
        out["p"].values, trend_map(g)["p"].values, equal_nan=True
    )
