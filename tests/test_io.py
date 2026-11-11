import numpy as np
import pandas as pd
import pytest
import xarray as xr

from trendlab.io import standardize, validate


def _raw():
    time = pd.date_range("2001-01-01", periods=4, freq="YS")
    lat = [10.0, 0.0, -10.0]          # descending on purpose
    lon = [0.0, 90.0, 180.0, 270.0]   # 0..360 on purpose
    data = np.arange(4 * 3 * 4, dtype=float).reshape(4, 3, 4)
    return xr.DataArray(
        data,
        coords={"time": time, "latitude": lat, "longitude": lon},
        dims=("time", "latitude", "longitude"),
        attrs={"units": "unit_x"},
    )


def test_raw_data_fails_validation():
    with pytest.raises(ValueError):
        validate(_raw())


def test_standardize_gives_valid_data():
    out = standardize(_raw())
    validate(out)
    assert list(out["lat"].values) == [-10.0, 0.0, 10.0]
    assert list(out["lon"].values) == [-180.0, -90.0, 0.0, 90.0]


def test_values_follow_their_coordinates():
    out = standardize(_raw()).isel(time=0)
    assert float(out.sel(lat=10.0, lon=0.0)) == 0.0
    assert float(out.sel(lat=10.0, lon=-180.0)) == 2.0
