"""Mann-Kendall trend test and Sen's slope for a 1-D series."""
import numpy as np
from scipy import stats


def mann_kendall(x) -> dict:
    """Mann-Kendall test with tie correction. NaNs are dropped.

    Returns dict with: s, var_s, z, p (two-sided), n.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return {"s": np.nan, "var_s": np.nan, "z": np.nan, "p": np.nan, "n": n}

    # diff[i, j] = x[j] - x[i]; keep only the pairs with i < j
    diff = x[None, :] - x[:, None]
    s = float(np.sign(diff[np.triu_indices(n, k=1)]).sum())

    _, counts = np.unique(x, return_counts=True)
    tie_term = float((counts * (counts - 1) * (2 * counts + 5)).sum())
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18

    if var_s == 0:  # every value identical: no trend, nothing to test
        return {"s": s, "var_s": var_s, "z": 0.0, "p": 1.0, "n": n}

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * stats.norm.sf(abs(z))
    return {"s": s, "var_s": var_s, "z": float(z), "p": float(p), "n": n}


def sen_slope(x, t=None) -> float:
    """Median of pairwise slopes. `t` defaults to 0, 1, 2, ... (unit steps).

    NaNs are dropped together with their time positions, so a gap in the
    series is treated as a gap in time, not squeezed shut.
    """
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float) if t is None else np.asarray(t, dtype=float)
    keep = ~np.isnan(x)
    x, t = x[keep], t[keep]
    if len(x) < 2:
        return np.nan
    iu = np.triu_indices(len(x), k=1)
    slopes = (x[None, :] - x[:, None])[iu] / (t[None, :] - t[:, None])[iu]
    return float(np.median(slopes))


def trend_map(da):
    """Run Mann-Kendall and Sen's slope on every grid cell along "time".

    `da` must follow docs/data_format.md. Returns a Dataset with s, z, p,
    slope on the (lat, lon) grid. Slope is per time step, not per year.
    """
    import xarray as xr
    from trendlab.io import validate

    validate(da)

    def _one(series):
        r = mann_kendall(series)
        return r["s"], r["z"], r["p"], sen_slope(series)

    s, z, p, slope = xr.apply_ufunc(
        _one,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
    )
    out = xr.Dataset({"s": s, "z": z, "p": p, "slope": slope})
    out["slope"].attrs["units"] = f"{da.attrs['units']} per time step"
    return out
