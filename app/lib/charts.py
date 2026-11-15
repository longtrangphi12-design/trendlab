"""Plotly figures. Pure functions: data in, figure out (no file reads here)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import text as T

FONT = dict(family="Inter, system-ui, sans-serif", size=12, color="#111827")
GRID = "#EEF2F0"


def _base(fig: go.Figure, height: int, **kw) -> go.Figure:
    layout = dict(height=height, margin=dict(l=10, r=10, t=10, b=10),
                  paper_bgcolor="white", plot_bgcolor="white", font=FONT, showlegend=False)
    layout.update(kw)
    fig.update_layout(**layout)
    return fig


def discrete_scale(colors: list[str]) -> list:
    """Step colour scale so integer codes 0..k-1 map to exact colours."""
    k = len(colors)
    scale = []
    for n, c in enumerate(colors):
        scale += [[n / k, c], [(n + 1) / k, c]]
    return scale


# ---------------------------------------------------------------- maps
def raster_map(codes: np.ndarray, colors: list[str], lat: np.ndarray, lon: np.ndarray, *,
               hover: dict | None = None, selected: tuple[float, float] | None = None,
               bands: bool = True, height: int = 620) -> go.Figure:
    """ONE raster (go.Heatmap) of integer category codes; NaN = transparent.

    hover: optional {"lon","lat","text","customdata","template"} for an invisible
    Scattergl layer on the basin cells. It gives the tooltip and makes cells
    clickable (Plotly heatmaps cannot be 'selected', scatter points can).
    """
    k = len(colors)
    fig = go.Figure(go.Heatmap(
        x=lon, y=lat, z=codes, zmin=-0.5, zmax=k - 0.5,
        colorscale=discrete_scale(colors), showscale=False, hoverinfo="skip",
    ))
    if hover is not None:
        fig.add_trace(go.Scattergl(
            x=hover["lon"], y=hover["lat"], mode="markers",
            marker=dict(size=5, color="rgba(0,0,0,0)"),
            text=hover.get("text"), customdata=hover.get("customdata"),
            hovertemplate=hover.get("template"), name="cells",
            selected=dict(marker=dict(color="rgba(0,0,0,0)")),
            unselected=dict(marker=dict(color="rgba(0,0,0,0)")),
        ))
    if bands:
        for y, lab, pos in [(-3, "North (≥ 3°S)", "top right"), (-3, "Middle (3–10°S)", "bottom right"),
                            (-10, "South (< 10°S)", "bottom right")]:
            fig.add_hline(y=y, line=dict(color="#6B7280", width=1.5, dash="dash"),
                          annotation_text=lab, annotation_position=pos,
                          annotation_font=dict(size=11, color="#374151"))
    if selected is not None:
        fig.add_trace(go.Scatter(
            x=[selected[1]], y=[selected[0]], mode="markers", hoverinfo="skip",
            marker=dict(symbol="square-open", size=14, color="#000000", line=dict(width=3)),
        ))
    fig.update_xaxes(showgrid=False, zeroline=False, ticksuffix="°", title=None, constrain="domain",
                     range=[float(lon.min()) - 0.05, float(lon.max()) + 0.05])
    fig.update_yaxes(showgrid=False, zeroline=False, ticksuffix="°", title=None,
                     scaleanchor="x", scaleratio=1, constrain="domain",
                     range=[float(lat.min()) - 0.05, float(lat.max()) + 0.05])
    _base(fig, height, plot_bgcolor="#EAF0EE", dragmode="pan", hovermode="closest",
          clickmode="event+select")
    return fig


# ---------------------------------------------------------------- yearly charts
def _year_bands(fig: go.Figure, drought=(2005, 2010, 2015, 2016), last=(2022, 2023, 2024)):
    for y in drought:
        fig.add_vrect(x0=y - 0.5, x1=y + 0.5, fillcolor=T.C_DROUGHT_BAND, opacity=0.25, line_width=0, layer="below")
    if last:
        fig.add_vrect(x0=min(last) - 0.5, x1=max(last) + 0.5, fillcolor=T.C_PINK_BAND, opacity=0.45,
                      line_width=0, layer="below")


def sen_line(years: np.ndarray, y: np.ndarray, slope: float) -> np.ndarray:
    """Sen line drawn with the FILE's slope; intercept = median(y − slope·t)."""
    t = years - years[0]
    ok = ~np.isnan(y)
    icpt = np.median(y[ok] - slope * t[ok])
    return icpt + slope * t


def yearly_series(years, y, slope, sen_color="#6B7280", *, last=(2022, 2023, 2024),
                  height=230, y_title="mm per year") -> go.Figure:
    years = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)
    fig = go.Figure()
    _year_bands(fig, last=last)
    fig.add_trace(go.Scatter(x=years, y=y, mode="lines+markers", line=dict(color="#1F2937", width=2),
                             marker=dict(size=4), name="Yearly total",
                             hovertemplate="%{x:.0f}: %{y:,.0f} mm<extra></extra>"))
    if slope is not None:
        fig.add_trace(go.Scatter(x=years, y=sen_line(years, y, slope), mode="lines",
                                 line=dict(color=sen_color, width=2, dash="dash"), name="Sen's slope",
                                 hovertemplate=f"Sen's slope {T.fmt_signed(slope)} mm/yr<extra></extra>"))
    fig.update_xaxes(range=[years[0] - 0.6, years[-1] + 0.6], dtick=4, showgrid=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, tickformat=",")
    return _base(fig, height)


def stripes(years, values, mean: float, height=90) -> go.Figure:
    anom = np.asarray(values, dtype=float) - mean
    lim = float(np.abs(anom).max())
    fig = go.Figure(go.Heatmap(
        x=years, y=[0], z=[anom], zmin=-lim, zmax=lim, showscale=False, xgap=0,
        colorscale=[[0, "#1E3A8A"], [0.35, "#93C5FD"], [0.5, "#F3F4F6"], [0.65, "#FDBA74"], [1, "#C2410C"]],
        customdata=[np.asarray(values)],
        hovertemplate="%{x}: %{customdata:,.0f} mm (%{z:+,.0f} vs mean)<extra></extra>",
    ))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(dtick=23, tick0=2001, showgrid=False)
    return _base(fig, height)


def end_year_bars(df: pd.DataFrame, height=240) -> go.Figure:
    colors = ["#FDBA74" if v >= 0 else "#3B82F6" for v in df["sen_mm_yr"]]
    fig = go.Figure(go.Bar(
        x=df["end_year"], y=df["sen_mm_yr"], marker_color=colors,
        text=[T.fmt_signed(v) for v in df["sen_mm_yr"]], textposition="outside",
        customdata=df["p_hr"], cliponaxis=False,
        hovertemplate="2001–%{x}: %{y:+.1f} mm/yr · p = %{customdata:.2f} (not corrected)<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#111827", width=1))
    fig.update_xaxes(dtick=1, showgrid=False, title="Final year of the record")
    fig.update_yaxes(title="Sen's slope (mm/yr)", gridcolor=GRID, zeroline=False)
    return _base(fig, height)


# ---------------------------------------------------------------- regions
def region_intervals(df: pd.DataFrame, height=280) -> go.Figure:
    df = df.iloc[::-1]
    fig = go.Figure()
    for _, r in df.iterrows():
        col = "#1E3A8A" if bool(r["boot_excludes_zero"]) else "#6B7280"
        label = f"<b>{r['name']}</b><br><span style='font-size:11px'>{r['sub']} · {int(r['n_cells']):,} cells</span>"
        fig.add_trace(go.Scatter(x=[r["boot_lo"], r["boot_hi"]], y=[label, label], mode="lines",
                                 line=dict(color=col, width=8), opacity=0.35, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r["sen_mm_yr"]], y=[label], mode="markers",
                                 marker=dict(color=col, size=14),
                                 hovertemplate=(f"Sen {r['sen_mm_yr']:+.1f} mm/yr · bootstrap "
                                                f"{r['boot_lo']:+.1f} to {r['boot_hi']:+.1f}<extra></extra>")))
    fig.add_vline(x=0, line=dict(color="#111827", width=1.5))
    lo, hi = float(min(df["boot_lo"].min(), 0)), float(max(df["boot_hi"].max(), 0))
    pad = 0.05 * (hi - lo)
    fig.update_xaxes(title="Sen's slope, mm per year", gridcolor=GRID, zeroline=False, range=[lo - pad, hi + pad])
    return _base(fig, height)


def end_year_lines(df: pd.DataFrame, height=320) -> go.Figure:
    sty = {"Whole basin": ("#111827", 3, "solid"), "North": ("#6B7280", 2, "dash"),
           "Middle": ("#1E3A8A", 2.5, "solid"), "South": ("#0F766E", 2.5, "solid")}
    fig = go.Figure()
    for name, g in df.groupby("name", sort=False):
        c, w, d = sty.get(name, ("#374151", 2, "solid"))
        fig.add_trace(go.Scatter(x=g["end_year"], y=g["sen_mm_yr"], mode="lines+markers", name=name,
                                 line=dict(color=c, width=w, dash=d), marker=dict(size=5),
                                 customdata=g["p_hr"],
                                 hovertemplate=name + " 2001–%{x}: %{y:+.1f} mm/yr · p = %{customdata:.2f}<extra></extra>"))
        last = g.iloc[-1]
        fig.add_annotation(x=last["end_year"], y=last["sen_mm_yr"], text=f"{name} {last['sen_mm_yr']:+.1f}",
                           xanchor="left", xshift=8, showarrow=False, font=dict(color=c, size=12))
    fig.add_hline(y=0, line=dict(color="#111827", width=1))
    fig.update_xaxes(dtick=1, showgrid=False, range=[2014.6, 2025.6])
    fig.update_yaxes(title="Sen's slope (mm/yr)", gridcolor=GRID, zeroline=False)
    return _base(fig, height)


# ---------------------------------------------------------------- exploratory
def indices_chart(df: pd.DataFrame, height=240) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=df["nino34"], name="Niño 3.4 (Pacific)", mode="lines",
                             line=dict(color=T.C_NINO, width=2.5)))
    fig.add_trace(go.Scatter(x=df["year"], y=df["tna"], name="TNA (tropical North Atlantic)", mode="lines",
                             line=dict(color=T.C_TNA, width=2.5, dash="dash")))
    fig.add_hline(y=0, line=dict(color="#111827", width=1))
    fig.update_yaxes(title="SD units", gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False)
    _base(fig, height, showlegend=True,
          legend=dict(orientation="h", y=-0.2, x=0))
    return fig


# ---------------------------------------------------------------- discovery
def anomaly_lines(years, area_series, basin_series, height=200) -> go.Figure:
    """Your area vs whole basin, each as % from its own mean."""
    a = np.asarray(area_series, float)
    b = np.asarray(basin_series, float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=(b / b.mean() - 1) * 100, name="Whole basin", mode="lines",
                             line=dict(color="#9CA3AF", width=2, dash="dash"),
                             hovertemplate="%{x}: basin %{y:+.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=(a / a.mean() - 1) * 100, name="Your area", mode="lines",
                             line=dict(color="#111827", width=2.5),
                             hovertemplate="%{x}: your area %{y:+.1f}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#9CA3AF", width=1))
    fig.update_yaxes(ticksuffix="%", gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return _base(fig, height, showlegend=True, legend=dict(orientation="h", y=-0.25, x=0))


def dry_share_bars(ts: pd.DataFrame, year: int, height=170) -> go.Figure:
    cols = ["#7C3AED" if y == year else ("#93C5FD" if s >= 0.5 else "#E5E7EB")
            for y, s in zip(ts["year"], ts["dry_share"])]
    fig = go.Figure(go.Bar(x=ts["year"], y=ts["dry_share"] * 100, marker_color=cols,
                           hovertemplate="%{x}: %{y:.0f}% of the basin ≥3% drier<extra></extra>"))
    fig.update_yaxes(ticksuffix="%", gridcolor=GRID, range=[0, 100])
    fig.update_xaxes(showgrid=False, dtick=23, tick0=2001)
    return _base(fig, height)
