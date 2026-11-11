import numpy as np
import pytest

from trendlab.report import analyze, summarize
from trendlab.simulate import autocorr_null_threshold, null_grid

COLUMNS = {
    "lat", "lon", "n_valid", "slope", "z", "p_raw", "p_bh", "r1",
    "significant_raw", "significant_bh", "autocorr_flag", "warning",
}


def _trend_grid():
    da = null_grid(n_time=24, n_lat=10, n_lon=10, phi=0.0, seed=0).copy()
    da.values[:, :2, :] += 0.3 * np.arange(24)[:, None, None]  # real trend, 20 cells
    return da


def _row(df, da, i, j):
    sel = (df["lat"] == da["lat"].values[i]) & (df["lon"] == da["lon"].values[j])
    return df[sel].iloc[0]


def test_one_row_per_cell_with_expected_columns():
    df = analyze(_trend_grid(), n_sim=200)
    assert len(df) == 100
    assert COLUMNS <= set(df.columns)


def test_pipeline_finds_the_injected_trend():
    da = _trend_grid()
    df = analyze(da, n_sim=200)
    block = df["lat"].isin(da["lat"].values[:2])
    assert df.loc[block, "significant_bh"].sum() >= 18
    assert df.loc[~block, "significant_bh"].sum() <= 3
    assert abs(df.loc[block, "slope"].median() - 0.3) < 0.03


def test_warnings_for_gaps_empty_cells_and_autocorrelation():
    da = null_grid(n_time=24, n_lat=10, n_lon=10, seed=1).copy()
    da.values[:5, 2, 2] = np.nan                                  # 19 of 24 valid
    da.values[:, 3, 3] = np.nan                                   # empty cell
    da.values[:, 4, 4] = np.sin(2 * np.pi * np.arange(24) / 24)   # slow wave
    df = analyze(da, n_sim=200)
    assert "many gaps" in _row(df, da, 2, 2)["warning"]
    empty = _row(df, da, 3, 3)
    assert "too few valid points" in empty["warning"] and np.isnan(empty["p_raw"])
    assert "high autocorrelation" in _row(df, da, 4, 4)["warning"]


def test_summarize_is_consistent():
    df = analyze(_trend_grid(), n_sim=200)
    s = summarize(df)
    assert s["n_cells"] == 100
    assert s["n_significant_bh"] <= s["n_significant_raw"]
    assert s["n_significant_bh_and_flagged"] <= s["n_significant_bh"]


def test_settings_are_recorded_in_attrs():
    df = analyze(_trend_grid(), n_sim=200)
    assert df.attrs["n_time"] == 24
    assert df.attrs["autocorr_threshold"] == pytest.approx(
        autocorr_null_threshold(24, q=0.95, n_sim=200, seed=0)
    )
    assert df.attrs["slope_units"] == "unit_x per time step"
