"""Bring gridded data into trendlab's standard form (see docs/data_format.md)."""
import numpy as np
import xarray as xr

DIM_ORDER = ("time", "lat", "lon")
_ALIASES = {"latitude": "lat", "longitude": "lon"}


def standardize(da: xr.DataArray) -> xr.DataArray:
    """Rename coordinates, wrap lon to -180..180, sort, order dims."""
    rename = {old: new for old, new in _ALIASES.items() if old in da.dims}
    da = da.rename(rename)
    da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180)
    da = da.sortby(["time", "lat", "lon"])
    return da.transpose(*DIM_ORDER)


def validate(da: xr.DataArray) -> xr.DataArray:
    """Raise ValueError listing every way `da` breaks the standard."""
    problems = []
    if tuple(da.dims) != DIM_ORDER:
        problems.append(f"dims must be {DIM_ORDER}, got {tuple(da.dims)}")
    else:
        if not np.issubdtype(da["time"].dtype, np.datetime64):
            problems.append("time must be datetime64")
        if float(da["lat"].min()) < -90 or float(da["lat"].max()) > 90:
            problems.append("lat must be within -90..90")
        if float(da["lon"].min()) < -180 or float(da["lon"].max()) > 180:
            problems.append("lon must be within -180..180")
        if not da["lat"].to_index().is_monotonic_increasing:
            problems.append("lat must be ascending")
        if not da["lon"].to_index().is_monotonic_increasing:
            problems.append("lon must be ascending")
    if "units" not in da.attrs:
        problems.append("missing 'units' attribute")
    if problems:
        raise ValueError("; ".join(problems))
    return da
