"""Combine trend, multiple-testing and autocorrelation results in one table."""
import numpy as np
import xarray as xr

from trendlab.autocorr import autocorr_flag, autocorr_map
from trendlab.io import validate
from trendlab.multitest import benjamini_hochberg
from trendlab.simulate import autocorr_null_threshold
from trendlab.trend import trend_map


def analyze(da, alpha=0.05, q=0.95, n_sim=2000, seed=0, min_valid_frac=0.8):
    """One row per grid cell with trend results and warnings.

    `da` must follow docs/data_format.md and already be cleaned to the
    resolution you want to test (e.g. one value per year).
    """
    validate(da)
    n_time = da.sizes["time"]

    trend = trend_map(da)
    r1 = autocorr_map(da)
    threshold = autocorr_null_threshold(n_time, q=q, n_sim=n_sim, seed=seed)

    p = trend["p"]
    p_bh = xr.DataArray(benjamini_hochberg(p.values), coords=p.coords, dims=p.dims)

    ds = xr.Dataset({
        "n_valid": da.notnull().sum("time"),
        "slope": trend["slope"],
        "z": trend["z"],
        "p_raw": p,
        "p_bh": p_bh,
        "r1": r1,
    })
    df = ds.to_dataframe().reset_index()

    df["significant_raw"] = df["p_raw"] <= alpha
    df["significant_bh"] = df["p_bh"] <= alpha
    df["autocorr_flag"] = autocorr_flag(df["r1"], threshold)

    too_few = (df["n_valid"] < 3).to_numpy()
    gaps = ((df["n_valid"] >= 3) & (df["n_valid"] < min_valid_frac * n_time)).to_numpy()
    checks = [
        (too_few, "too few valid points"),
        (gaps, "many gaps"),
        (df["autocorr_flag"].to_numpy(), "high autocorrelation"),
    ]
    labels = np.full(len(df), "", dtype=object)
    for mask, text in checks:
        labels = np.where(mask, np.where(labels == "", text, labels + "; " + text), labels)
    df["warning"] = labels

    units = da.attrs.get("units", "")
    df.attrs.update({
        "alpha": alpha,
        "autocorr_quantile": q,
        "autocorr_threshold": threshold,
        "n_time": n_time,
        "n_sim": n_sim,
        "seed": seed,
        "min_valid_frac": min_valid_frac,
        "units": units,
        "slope_units": f"{units} per time step",
    })
    return df


def summarize(df):
    """Counts that belong in the first lines of any report."""
    return {
        "n_cells": int(len(df)),
        "n_tested": int(df["p_raw"].notna().sum()),
        "n_significant_raw": int(df["significant_raw"].sum()),
        "n_significant_bh": int(df["significant_bh"].sum()),
        "n_flagged_autocorr": int(df["autocorr_flag"].sum()),
        "n_significant_bh_and_flagged": int(
            (df["significant_bh"] & df["autocorr_flag"]).sum()
        ),
    }
