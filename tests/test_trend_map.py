import numpy as np
import pandas as pd
import pytest
import xarray as xr

from trendlab.trend import mann_kendall, trend_map


def _grid():
    n = 24
    time = pd.date_range("2001-01-01", periods=n, freq="YS")
    a = (np.arange(12).reshape(3, 4) - 5.0) * 0.5   # known slopes, one is 0
    data = a[None, :, :] * np.arange(n)[:, None, None]
    data[:, 0, 0] = np.nan                            # one empty cell
    da = xr.DataArray(
        data,
        coords={"time": time, "lat": [-10.0, 0.0, 10.0],
                "lon": [-90.0, 0.0, 90.0, 180.0]},
        dims=("time", "lat", "lon"),
        attrs={"units": "unit_x"},
    )
    return da, a


def test_slope_map_recovers_known_slopes():
    da, a = _grid()
    out = trend_map(da)
    ok = ~np.isnan(da.isel(time=0).values)
    np.testing.assert_allclose(out["slope"].values[ok], a[ok])


def test_sign_of_z_follows_sign_of_slope():
    da, a = _grid()
    z = trend_map(da)["z"].values
    ok = (a != 0) & ~np.isnan(z)
    assert np.all(np.sign(z[ok]) == np.sign(a[ok]))


def test_flat_cell_has_no_trend_and_empty_cell_is_nan():
    da, _ = _grid()
    out = trend_map(da)
    assert out["p"].isel(lat=1, lon=1).item() == 1.0
    assert np.isnan(out["p"].isel(lat=0, lon=0).item())


def test_cell_matches_the_1d_function():
    da, _ = _grid()
    out = trend_map(da)
    direct = mann_kendall(da.isel(lat=2, lon=3).values)
    assert out["p"].isel(lat=2, lon=3).item() == pytest.approx(direct["p"])


def test_slope_units_attribute():
    da, _ = _grid()
    assert trend_map(da)["slope"].attrs["units"] == "unit_x per time step"


def test_invalid_input_is_rejected():
    da, _ = _grid()
    with pytest.raises(ValueError):
        trend_map(da.drop_attrs() if hasattr(da, "drop_attrs") else da.copy(data=da.values, deep=True).assign_attrs(units=None).drop_attrs(deleted=None) if False else xr.DataArray(da.values, coords=da.coords, dims=da.dims))
