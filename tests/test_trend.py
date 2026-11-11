import numpy as np
import pytest

from trendlab.trend import mann_kendall, sen_slope


def test_mk_perfect_increase():
    r = mann_kendall([1, 2, 3, 4, 5])
    assert r["s"] == 10
    assert r["var_s"] == pytest.approx(300 / 18)
    assert r["z"] == pytest.approx(9 / np.sqrt(300 / 18))
    assert r["p"] == pytest.approx(0.0275, abs=1e-3)


def test_mk_perfect_decrease_is_mirror_image():
    up = mann_kendall([1, 2, 3, 4, 5])
    down = mann_kendall([5, 4, 3, 2, 1])
    assert down["s"] == -up["s"]
    assert down["z"] == pytest.approx(-up["z"])
    assert down["p"] == pytest.approx(up["p"])


def test_mk_tie_correction():
    r = mann_kendall([1, 2, 2, 3])
    assert r["s"] == 5
    assert r["var_s"] == pytest.approx(138 / 18)
    assert r["p"] == pytest.approx(0.1486, abs=2e-3)


def test_mk_constant_series_has_no_trend():
    r = mann_kendall([3, 3, 3, 3])
    assert r["s"] == 0 and r["p"] == 1.0


def test_mk_too_short_returns_nan():
    assert np.isnan(mann_kendall([1, 2])["p"])


def test_mk_ignores_nan():
    assert mann_kendall([1, np.nan, 2, 3, 4, 5])["s"] == 10


def test_sen_slope_hand_calculated():
    assert sen_slope([1, 3, 2, 5, 4]) == pytest.approx(0.875)


def test_sen_slope_respects_time_gaps():
    # the NaN sits at position 1, so the points are at t = 0, 2, 3
    assert sen_slope([1, np.nan, 3, 4]) == pytest.approx(1.0)


def test_sen_slope_is_robust_to_one_outlier():
    clean = sen_slope([1, 2, 3, 4, 5, 6, 7, 8])
    dirty = sen_slope([1, 2, 3, 4, 5, 6, 7, 100])
    assert clean == pytest.approx(dirty)
