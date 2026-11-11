import numpy as np

from trendlab.io import validate
from trendlab.simulate import false_positive_rates, null_grid


def test_null_grid_is_valid_and_reproducible():
    a = null_grid(seed=3)
    validate(a)
    np.testing.assert_array_equal(a.values, null_grid(seed=3).values)


def test_null_grid_has_unit_variance():
    for phi in (0.0, 0.5):
        assert abs(float(null_grid(phi=phi, seed=1).std()) - 1.0) < 0.1


def test_white_noise_raw_rate_is_near_alpha():
    r = false_positive_rates(phi=0.0, n_lat=40, n_lon=50, seed=5)
    assert 0.02 < r["raw"] < 0.09


def test_bh_flags_far_fewer_cells_than_raw_on_null_data():
    r = false_positive_rates(phi=0.0, n_lat=40, n_lon=50, seed=5)
    assert r["bh"] <= r["raw"]
    assert r["bh"] <= 0.02
