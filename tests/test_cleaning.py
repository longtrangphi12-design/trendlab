import numpy as np
import pandas as pd
import pytest
import xarray as xr

from trendlab.cleaning import clip_time, mask_sparse_cells, select_months, yearly
from trendlab.io import validate


def _monthly(n_lon=1):
    time = pd.date_range("2001-01-01", periods=36, freq="MS")
    data = np.arange(1.0, 37.0)[:, None, None] * np.ones((36, 1, n_lon))
    return xr.DataArray(
        data,
        coords={"time": time, "lat": [0.0], "lon": np.arange(n_lon) * 10.0},
        dims=("time", "lat", "lon"),
        attrs={"units": "unit_x"},
    )


def _series(da):
    return da.isel(lat=0, lon=0).values


def test_yearly_mean_and_sum_hand_calculated():
    da = _monthly()
    np.testing.assert_allclose(_series(yearly(da, 12, "mean")), [6.5, 18.5, 30.5])
    np.testing.assert_allclose(_series(yearly(da, 12, "sum")), [78, 222, 366])


def test_yearly_mean_respects_valid_fraction():
    da = _monthly().copy()
    da.values[12:15, 0, 0] = np.nan   # year 2 loses 3 of 12 months
    strict = _series(yearly(da, 12, "mean", min_valid_frac=0.8))
    loose = _series(yearly(da, 12, "mean", min_valid_frac=0.7))
    assert np.isnan(strict[1]) and strict[0] == 6.5
    assert loose[1] == pytest.approx(20.0)


def test_yearly_sum_requires_complete_years():
    da = _monthly().copy()
    da.values[0, 0, 0] = np.nan       # year 1 loses one month
    out = _series(yearly(da, 12, "sum", min_valid_frac=0.1))
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [222, 366])


def test_select_months_then_yearly_gives_one_value_per_season():
    summer = select_months(_monthly(), [6, 7, 8])
    assert summer.sizes["time"] == 9
    np.testing.assert_allclose(_series(yearly(summer, 3, "mean")), [7, 19, 31])


def test_clip_time():
    out = clip_time(_monthly(), "2002", "2002")
    assert out.sizes["time"] == 12
    np.testing.assert_allclose(_series(out), np.arange(13.0, 25.0))


def test_mask_sparse_cells():
    da = _monthly(n_lon=2).copy()
    da.values[::2, 0, 1] = np.nan     # second cell loses half its steps
    out = mask_sparse_cells(da, 0.8)
    assert out.isel(lat=0, lon=0).notnull().all()
    assert out.isel(lat=0, lon=1).isnull().all()


def test_output_keeps_standard_format_and_attrs():
    out = yearly(_monthly(), 12, "mean")
    validate(out)
    assert out.attrs["units"] == "unit_x"
    assert "aggregation" in out.attrs
    assert str(out["time"].values[0])[:10] == "2001-01-01"


def test_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError):
        yearly(_monthly(), 12, "median")
