"""Lag-1 autocorrelation as a warning flag, and trend-free pre-whitening.

The flag does NOT correct p-values. It only measures how strongly a series
resembles its own previous step, so users can see where the Mann-Kendall
independence assumption is doubtful. `tfpw_mann_kendall` is the correction.
"""
import numpy as np
import xarray as xr

from trendlab.trend import mann_kendall, sen_slope


def lag1_autocorr(x, detrend=True) -> float:
    """Lag-1 autocorrelation of a 1-D series. NaN where it is not defined.

    With detrend=True the Sen's-slope line is removed first, so a real trend
    does not masquerade as autocorrelation. Pairs touching a NaN are skipped.
    """
    x = np.asarray(x, dtype=float)
    if detrend:
        t = np.arange(len(x), dtype=float)
        slope = sen_slope(x, t)
        if np.isnan(slope):
            return np.nan
        keep = ~np.isnan(x)
        intercept = np.median(x[keep] - slope * t[keep])
        x = x - (intercept + slope * t)

    valid = ~np.isnan(x)
    if valid.sum() < 3:
        return np.nan
    d = x - x[valid].mean()
    denom = np.sum(d[valid] ** 2)
    if denom == 0:
        return np.nan
    pair = valid[:-1] & valid[1:]
    return float(np.sum(d[:-1][pair] * d[1:][pair]) / denom)


def autocorr_map(da, detrend=True):
    """Lag-1 autocorrelation of every grid cell along "time"."""
    from trendlab.io import validate

    validate(da)
    r1 = xr.apply_ufunc(
        lambda s: lag1_autocorr(s, detrend),
        da,
        input_core_dims=[["time"]],
        vectorize=True,
    )
    r1.name = "lag1"
    r1.attrs = {"long_name": "lag-1 autocorrelation", "units": "1"}
    return r1


def autocorr_flag(r1, threshold=0.3):
    """True where r1 exceeds the threshold. NaN gives False.

    The threshold is a judgement call, not a statistical test.
    """
    return r1 > threshold


def tfpw_mann_kendall(x, threshold) -> dict:
    """Mann-Kendall with trend-free pre-whitening, applied only if needed.

    Steps: remove the Sen line, measure r1 on the residual; if r1 is above
    `threshold`, pre-whiten the residual (y_t - r1 * y_(t-1)), add the trend
    back and run Mann-Kendall on that. Otherwise run plain Mann-Kendall.
    Returns the Mann-Kendall dict plus `r1` and `prewhitened`.
    The reported slope should still be Sen's slope of the original series.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    t = np.arange(n, dtype=float)
    slope = sen_slope(x, t)
    r1 = lag1_autocorr(x, detrend=True)

    if np.isnan(slope) or np.isnan(r1) or not r1 > threshold:
        out = mann_kendall(x)
        out.update(r1=r1, prewhitened=False)
        return out

    keep = ~np.isnan(x)
    intercept = np.median(x[keep] - slope * t[keep])
    y = x - (intercept + slope * t)        # residual around the Sen line
    w = np.full(n, np.nan)
    w[1:] = y[1:] - r1 * y[:-1]            # NaN where a step or its predecessor is missing
    out = mann_kendall(w + slope * t)      # add the trend back
    out.update(r1=r1, prewhitened=True)
    return out


def tfpw_map(da, threshold):
    """TFPW Mann-Kendall on every grid cell. Returns z, p, prewhitened."""
    from trendlab.io import validate

    validate(da)

    def _one(series):
        r = tfpw_mann_kendall(series, threshold)
        return r["z"], r["p"], float(r["prewhitened"])

    z, p, pw = xr.apply_ufunc(
        _one,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[[], [], []],
        vectorize=True,
    )
    return xr.Dataset({"z": z, "p": p, "prewhitened": pw > 0.5})
