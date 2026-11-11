"""Synthetic data with NO trend, to measure false-positive behaviour."""
import numpy as np
import pandas as pd
import xarray as xr

from trendlab.autocorr import lag1_autocorr
from trendlab.multitest import benjamini_hochberg, significant
from trendlab.trend import trend_map


def null_grid(n_time=24, n_lat=20, n_lon=25, phi=0.0, seed=0):
    """Grid of independent AR(1) series, unit variance, no trend.

    phi=0 gives white noise; larger phi gives stronger autocorrelation.
    Cells are independent of each other (a simplification).
    """
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_time, n_lat, n_lon))
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, n_time):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi**2) * e[t]
    return xr.DataArray(
        x,
        coords={
            "time": pd.date_range("2001-01-01", periods=n_time, freq="YS"),
            "lat": np.linspace(-10, 10, n_lat),
            "lon": np.linspace(-90, 90, n_lon),
        },
        dims=("time", "lat", "lon"),
        attrs={"units": "unit_x"},
    )


def false_positive_rates(phi=0.0, alpha=0.05, **grid_kwargs):
    """Fraction of cells flagged significant on trend-free data.

    Returns the rate with raw p-values and after Benjamini-Hochberg.
    Ideal behaviour on trend-free data: raw close to alpha, bh close to 0.
    """
    p = trend_map(null_grid(phi=phi, **grid_kwargs))["p"].values
    return {
        "phi": phi,
        "raw": float(significant(p, alpha).mean()),
        "bh": float(significant(benjamini_hochberg(p), alpha).mean()),
    }


def autocorr_null_threshold(n_time, q=0.95, n_sim=2000, detrend=True, seed=0):
    """Threshold on lag-1 autocorrelation calibrated on independent series.

    Simulates `n_sim` white-noise series of length `n_time`, measures lag-1
    autocorrelation exactly as the real pipeline does, and returns the
    q-quantile. A cell above it is more autocorrelated than a fraction q of
    independent series of the same length.
    """
    rng = np.random.default_rng(seed)
    r = np.array([
        lag1_autocorr(rng.standard_normal(n_time), detrend)
        for _ in range(n_sim)
    ])
    return float(np.nanquantile(r, q))
