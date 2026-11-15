"""Exploratory layers: ENSO/TNA adjustment, dominant index, dry months.

Everything here is descriptive association, never cause. Each card hides itself
if its file is missing.
"""
import numpy as np
import streamlit as st

from lib import charts, data
from lib import text as T

summ = data.summary()
g = data.grid("trend_cells.nc")
LAT, LON = g["lat"], g["lon"]
ec = data.evidence()
inside = ec >= data.GREY
w = data.area_weight()
NOCFG = {"displayModeBar": False}


def exp_title(txt, level="####"):
    st.markdown(f"{level} {txt} <span class='te-expl'>{T.EXPLORATORY}</span>", unsafe_allow_html=True)


def legend(items):
    st.markdown("".join(
        f"<span class='te-key'><span class='te-sw' style='background:{c};"
        f"{'border:1px solid #D1D5DB;' if c in (T.C_GREY, '#D1D5DB') else ''}'></span>{t}</span>"
        for c, t in items), unsafe_allow_html=True)


st.markdown(f"## Exploratory layers <span class='te-expl'>{T.EXPLORATORY}</span>", unsafe_allow_html=True)
st.write("Extra views to explore the main result. They describe statistical associations, not causes, "
         "and they never replace the main map.")

n_down = int(summ["main_cells_down"])

# ------------------------------------------------------------------ adjusted trend
if data.has("drivers_cells.nc"):
    with st.container(border=True):
        exp_title("Trend after accounting for ENSO & TNA")
        options = ["Joint (main)", "Naive (sensitivity)"]
        if data.has("adjusted_drop_cells.nc"):
            options.append("Joint, 2001–2021 only")
        v = st.segmented_control("Version", options, default=options[0], key="adj_version") or options[0]
        if v.startswith("Joint (main)"):
            adj_cls = data.var("drivers_cells.nc", "class_adj_joint")
            share, area = summ["adj_joint_main_decreasing_kept_share"], summ["adj_joint_decreasing_area_share"]
            kept, new, inc = int(summ["adj_joint_main_decreasing_kept"]), int(summ["adj_joint_new_decreasing"]), int(summ["adj_joint_new_increasing"])
            suffix = ""
        elif v.startswith("Naive"):
            adj_cls = data.var("drivers_cells.nc", "class_adj_naive")
            share, area = summ["adj_naive_main_decreasing_kept_share"], summ["adj_naive_decreasing_area_share"]
            kept, new, inc = int(summ["adj_naive_main_decreasing_kept"]), int(summ["adj_naive_new_decreasing"]), int(summ["adj_naive_new_increasing"])
            suffix = ""
        else:
            adj_cls = data.var("adjusted_drop_cells.nc", "class_adj_2001_2021")
            share, area = summ["drop2022_24_main_decreasing_kept_share_adj"], summ["drop2022_24_adj_decreasing_area_share"]
            kept = int(((ec == data.DOWN) & (adj_cls == data.DOWN)).sum())
            new = int(((ec != data.DOWN) & (adj_cls == data.DOWN)).sum())
            inc = int((adj_cls == data.UP).sum())
            suffix = " when the record stops in 2021"

        codes = np.full(ec.shape, np.nan)
        codes[inside] = 0
        codes[inside & (ec == data.DOWN) & (adj_cls == data.DOWN)] = 1
        codes[inside & (ec != data.DOWN) & (adj_cls == data.DOWN)] = 2
        codes[inside & (adj_cls == data.UP)] = 3

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.plotly_chart(charts.raster_map(codes, [T.C_GREY, T.C_KEEP, T.C_GAIN, T.C_INC], LAT, LON, height=480),
                            width="stretch", config=NOCFG)
            legend([(T.C_KEEP, "Decrease before and after"), (T.C_GAIN, "Decrease only after adjustment"),
                    (T.C_INC, "Increase"), (T.C_GREY, "Not enough evidence")])
        with right:
            st.markdown(f"<div style='font-size:34px;font-weight:700;color:#1E3A8A'>{T.pct(share)}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"of the {n_down:,} decreasing cells still decrease ({kept:,} cells){suffix}")
            a, b = st.columns(2)
            a.markdown(f"<div class='te-tile'><div class='l'>Decrease area</div><b>{T.pct(area)}</b> "
                       f"<span class='l'>vs {T.pct(summ['main_area_share_down'])}</span></div>", unsafe_allow_html=True)
            b.markdown(f"<div class='te-tile'><div class='l'>Newly with evidence</div><b>{new:,}</b> cells</div>",
                       unsafe_allow_html=True)
            a, b = st.columns(2)
            a.markdown(f"<div class='te-tile'><div class='l'>Increase cells</div><b>{inc:,}</b> "
                       f"<span class='l'>vs {int(summ['main_cells_up'])}</span></div>", unsafe_allow_html=True)
            if suffix:
                b.markdown(f"<div class='te-tile' style='background:#FDF2F8'><div class='l'>With 2022–24 (joint)</div>"
                           f"<b>{T.pct(summ['adj_joint_main_decreasing_kept_share'])}</b> keep it</div>", unsafe_allow_html=True)
            else:
                b.markdown(f"<div class='te-tile' style='background:#FDF2F8'><div class='l'>Without 2022–24</div>"
                           f"<b>{T.pct(summ['drop2022_24_main_decreasing_kept_share_adj'])}</b> keep it "
                           f"<span class='l'>({T.pct(summ['drop2022_24_main_decreasing_kept_share_raw'])} unadjusted)</span></div>",
                           unsafe_allow_html=True)
            st.markdown("<div style='background:#F3E8FF;border-radius:8px;padding:8px 10px;margin-top:10px;font-size:13px'>"
                        "Newly significant cells reflect reduced noise, not a new trend.</div>", unsafe_allow_html=True)
            st.caption("Removing the part of each year's rainfall that moves with the ocean indices makes the series "
                       "less noisy, so the same small trend becomes easier to detect. It does not explain the decrease "
                       "away, and it does not make it robust to 2022–2024.")

# ------------------------------------------------------------------ dominant index + indices
c1, c2 = st.columns(2, gap="medium")
if data.has("dominant_driver.nc"):
    with c1, st.container(border=True):
        exp_title("Which ocean index are year-to-year swings associated with?", "#####")
        dd = data.var("dominant_driver.nc", "dominant_driver")
        codes = np.where(inside, dd, np.nan).astype(float)
        st.plotly_chart(charts.raster_map(codes, ["#D1D5DB", T.C_NINO, T.C_TNA], LAT, LON, height=400, bands=False),
                        width="stretch", config=NOCFG)
        legend([(T.C_TNA, f"Atlantic (TNA) · {T.pct(summ['dominant_tna_area_share'])}"),
                (T.C_NINO, f"Pacific (ENSO) · {T.pct(summ['dominant_nino34_area_share'])}"),
                ("#D1D5DB", f"Not predictable by ENSO/TNA · {T.pct(summ['dominant_none_area_share'])}")])
        st.write(f"Explanatory power per cell is low: the median cross-validated R² is {summ['r2_cv_median']:.2f}, "
                 f"and only {T.pct(summ['r2_cv_positive_share_of_cells'])} of cells have a cross-validated R² above 0. "
                 "Colour shows the larger effect, only where R² > 0.")
        st.caption("Descriptive statistical association, not causal attribution. Shares are of basin area.")

idx = data.csv("drivers_indices.csv")
if idx is not None:
    with c2, st.container(border=True):
        exp_title("The two ocean indices, 2001–2024", "#####")
        st.caption("Yearly, standardised (SD units) · from MERRA-2 surface temperature")
        st.plotly_chart(charts.indices_chart(idx), width="stretch", config=NOCFG)
        st.write(f"Neither index has a trend over 2001–2024 (p = {summ['index_nino34_p_hr']:.2f} and "
                 f"{summ['index_tna_p_hr']:.2f}), and they are nearly independent of each other "
                 f"(correlation {T.fmt_signed(summ['corr_nino34_tna'], 2)}, VIF {summ['vif_detrended']:.2f} after detrending).")

# ------------------------------------------------------------------ regional coefficients
rd = data.csv("regions_drivers.csv")
if rd is not None:
    with st.container(border=True):
        exp_title("By region: rainfall change per 1 SD of each index", "#####")
        cols = st.columns(len(rd))
        for col, (_, r) in zip(cols, rd.iterrows()):
            col.markdown(
                f"<div class='te-tile'><b>{r['name']}</b><br>"
                f"<span style='color:#15803D'>Niño 3.4</span> {T.fmt_signed(r['beta_nino34_mm_per_sd'], 0)} mm "
                f"({T.fmt_p(r['p_nino34'])})<br>"
                f"<span style='color:#6D28D9'>TNA</span> {T.fmt_signed(r['beta_tna_mm_per_sd'], 0)} mm "
                f"({T.fmt_p(r['p_tna'])})<br>"
                f"<span class='l'>Trend raw → adjusted: {T.fmt_signed(r['sen_raw'])} → {T.fmt_signed(r['sen_adj'])}</span></div>",
                unsafe_allow_html=True)
        tna_all_neg = bool((rd["beta_tna_mm_per_sd"] < 0).all())
        st.caption("mm of yearly rainfall per 1 SD of the index (p in brackets). "
                   + ("A warmer tropical North Atlantic is associated with less rainfall in every region. " if tna_all_neg else "")
                   + "Adjusted trend p-values hold only with 2022–2024 in the record (see Regions → 2001–2021 check).")

# ------------------------------------------------------------------ dry months
if data.has("dry_months_cells.nc"):
    with st.container(border=True):
        exp_title("Dry months per year")
        thr = data.dry_threshold_mm()
        cd = data.var("dry_months_cells.nc", "class_dry")
        left, right = st.columns([3, 2], gap="large")
        with left:
            codes = np.full(ec.shape, np.nan)
            codes[inside] = 0
            codes[inside & (cd == 2)] = 1
            codes[inside & (cd == 3)] = 2
            st.plotly_chart(charts.raster_map(codes, [T.C_GREY, "#B45309", "#0EA5E9"], LAT, LON, height=440),
                            width="stretch", config=NOCFG)
            n_more, n_fewer = int((inside & (cd == 2)).sum()), int((inside & (cd == 3)).sum())
            a_more = float(w[inside & (cd == 2)].sum() / w.sum())
            legend([("#B45309", f"More dry months, with evidence · {n_more:,} cells ({T.pct(a_more)} of area)"),
                    ("#0EA5E9", f"Fewer · {n_fewer:,} cells"), (T.C_GREY, "Not enough evidence")])
            st.caption(f"Dry month = less than {thr} mm, following Levine et al. (2016). "
                       f"Threshold {thr} mm; sensitivity not tested.")
        with right:
            rdm = data.csv("regions_dry_months.csv")
            if rdm is not None:
                for _, r in rdm.iterrows():
                    st.markdown(
                        f"<div class='te-tile' style='margin-bottom:8px;display:flex;justify-content:space-between'>"
                        f"<div><b>{r['name']}</b><div class='l'>{r['mean_dry_months']:.1f} dry months per year on average</div></div>"
                        f"<div style='text-align:right'><b>{T.fmt_signed(r['sen_dry_months_per_decade'], 2)}</b>"
                        f"<div class='l'>per decade · p = {T.fmt_p(r['p_hr'])}</div></div></div>",
                        unsafe_allow_html=True)
            st.caption(f"No strong conclusions from this layer: a fixed {thr} mm cut-off means very different things "
                       "in the wet north and the drier south.")
