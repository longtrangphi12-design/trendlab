"""Map explorer: evidence map, drop-years switch, cell details, basin card."""
import numpy as np
import streamlit as st

from lib import charts, data
from lib import text as T

S = st.session_state
S.setdefault("sel", None)          # (lat, lon) of the selected cell
S.setdefault("details", False)     # right panel open?
S.setdefault("legend", False)      # left panel open?
S.setdefault("variant", "None")    # drop-years switch
S.setdefault("last_click", None)   # last map click already handled

summ = data.summary()
g = data.grid("trend_cells.nc")
LAT, LON = g["lat"], g["lon"]
t1, t2 = data.magnitude_cutoffs()


# ------------------------------------------------------------------ callbacks
def select(lat, lon):
    S.sel = (float(lat), float(lon))
    S.details = True


def pick_variant(key):
    S.variant = key


# A click on the map arrives in S["map"] (the chart's key) on the next run.
# Handle it here, BEFORE drawing, so the marker and the panel update in the same run.
ev = S.get("map")
if ev and ev.get("selection", {}).get("points"):
    p = ev["selection"]["points"][-1]
    sig = (round(p["x"], 2), round(p["y"], 2))
    if sig != S.last_click:
        S.last_click = sig
        select(p["y"], p["x"])

if S.variant not in data.VARIANTS:  # the switch can be cleared by clicking the active option
    S.variant = "None"
variant = S.variant
full = variant == "None"

# ------------------------------------------------------------------ header
st.markdown(f"## {T.APP_TITLE}")
st.markdown(f"<div class='te-sub'>{T.APP_SUBTITLE}</div>", unsafe_allow_html=True)

c1, c2, _ = st.columns([1.2, 1.2, 5])
c1.toggle("Legend & data", key="legend")
c2.toggle("Details panel", key="details")

# ------------------------------------------------------------------ overview strip
reg = data.csv("regions_robustness.csv")
basin = reg[reg["region"] == "toan_luu_vuc"].iloc[0] if reg is not None else None


def bar(segments):
    """segments: list of (share 0-1, colour, min width px)."""
    return "<div class='te-bar'>" + "".join(
        f"<div style='width:{100 * w:.3f}%;min-width:{m}px;background:{c}'></div>" for w, c, m in segments
    ) + "</div>"


def key_item(color, value, label, border=None):
    bd = f"border:1px solid {border};" if border else ""
    return f"<span class='te-key'><span class='te-sw' style='background:{color};{bd}'></span><b>{value}</b> {label}</span>"


with st.container(border=True):
    if full:
        down, grey = summ["main_area_share_down"], summ["main_area_share_not_enough"]
        n_up, n_down = int(summ["main_cells_up"]), int(summ["main_cells_down"])
        st.markdown(f"**Whole basin at a glance** <span class='te-small'>· share of basin area · "
                    f"{data.n_cells():,} cells</span>", unsafe_allow_html=True)
        st.markdown(bar([(down, T.C_DEC[1], 0), (grey, "#D9DDE3", 0), (0.0001, T.C_INC, 3)])
                    + key_item(T.C_DEC[1], T.pct(down), f"decrease with evidence ({n_down:,} cells)")
                    + key_item("#D9DDE3", T.pct(grey), "not enough evidence to detect a trend")
                    + key_item(T.C_INC, f"{n_up} cells", "increase with evidence"),
                    unsafe_allow_html=True)
        if basin is not None:
            verdict = ("→ a basin-wide trend is supported." if basin["boot_excludes_zero"]
                       else "→ no basin-wide trend is supported.")
            st.markdown(
                f"Whole basin as one series: Sen {T.fmt_signed(basin['sen_mm_yr'])} mm/yr, "
                f"p = {T.fmt_p(basin['p_hr'])}, bootstrap interval {T.fmt_signed(basin['boot_lo'])} to "
                f"{T.fmt_signed(basin['boot_hi'])} {verdict} The decreases are concentrated in the "
                "south-central basin and along the Andes.")
    else:
        vc = data.variant_counts(variant)
        share = vc["kept_share"] if vc["kept_share"] is not None else vc["kept"] / vc["n_ev"]
        tot = vc["kept"] + vc["lost"]
        st.markdown(f"**{vc['long']}: {T.pct(share)} of cells with evidence keep it** "
                    f"<span class='te-small'>· out of the {vc['n_ev']:,} cells with evidence in the full period</span>",
                    unsafe_allow_html=True)
        st.markdown(bar([(vc["kept"] / tot, T.C_KEEP, 0), (vc["lost"] / tot, T.C_LOSE, 0)])
                    + key_item(T.C_KEEP, f"{vc['kept']:,}", "keep evidence")
                    + key_item(T.C_LOSE, f"{vc['lost']:,}", "lose it")
                    + key_item(T.C_GAIN, f"{vc['new']:,}", "gain evidence of decrease")
                    + key_item(T.C_INC, f"{vc['inc']:,}", "show an increase"),
                    unsafe_allow_html=True)
        if share < 0.5:
            msg = "Most of the decrease signal depends on these years. Read the map as fragile."
            ba = data.csv("basin_annual.csv")
            if ba is not None and set(vc["dropped_years"]) == set(data.LAST_YEARS):
                driest = ba.nsmallest(2, "basin_mean_mm")["year"].tolist()
                if set(driest) <= set(data.LAST_YEARS):
                    msg = (f"Most of the decrease signal depends on the last three, very dry years "
                           f"({min(driest)} and {max(driest)} are the driest in the series). Read the map as fragile.")
        else:
            msg = "Most cells keep their result: the signal does not depend on these years alone."
        st.markdown(msg)
    st.info(T.IN_SHORT)

# ------------------------------------------------------------------ drop-years switch
st.segmented_control("How fragile is it? Drop years:", list(data.VARIANTS), key="variant",
                     help="Recolours the map: which cells keep their evidence when these years are left out.")
# ------------------------------------------------------------------ layout
widths = ([1.1] if S.legend else []) + [3] + ([1.6] if S.details else [])
cols = st.columns(widths, gap="medium")
col_left = cols[0] if S.legend else None
col_map = cols[1] if S.legend else cols[0]
col_right = cols[-1] if S.details else None

# ------------------------------------------------------------------ legend panel
if col_left is not None:
    with col_left:
        with st.container(border=True):
            st.markdown("**Legend**")
            st.markdown("<div class='te-small'><b>DECREASE WITH EVIDENCE</b></div>"
                        + key_item(T.C_DEC[0], "", f"Lowest third of decreases · &lt; {t1:.1f} %/decade") + "<br>"
                        + key_item(T.C_DEC[1], "", f"Middle third · {t1:.1f}–{t2:.1f} %/decade") + "<br>"
                        + key_item(T.C_DEC[2], "", f"Highest third · &gt; {t2:.1f} %/decade")
                        + f"<div class='te-small'>{T.MAG_NOTE}</div><br>"
                        + "<div class='te-small'><b>OTHER</b></div>"
                        + key_item(T.C_INC, "", f"Increase with evidence ({int(summ['main_cells_up'])} cells, no scale)") + "<br>"
                        + key_item(T.C_GREY, "", T.GREY_LONG, T.C_GREY_BD),
                        unsafe_allow_html=True)
            st.caption(T.EVIDENCE_RULE)
        with st.container(border=True):
            st.markdown("**Data**")
            st.markdown(T.DATA_BOX.format(n=data.n_cells(), m=data.min_valid_years()))

# ------------------------------------------------------------------ map
pts = data.basin_points()
if full:
    codes = data.main_codes()
    colors = [T.C_GREY, T.C_INC, *T.C_DEC, T.C_NODATA]
    labels = np.array([T.GREY_SHORT, "Increase with evidence", "Decrease · lowest third",
                       "Decrease · middle third", "Decrease · highest third", "Not enough valid years"], dtype=object)
    legend_items = [(T.C_DEC[0], "Lowest third of decreases"), (T.C_DEC[1], "Middle third"),
                    (T.C_DEC[2], "Highest third"), (T.C_INC, "Increase"),
                    (T.C_GREY, "Not enough evidence (≠ no change)")]
    title = "Rainfall trend 2001–2024 · evidence after FDR"
else:
    codes = data.variant_codes(variant)
    colors = [T.C_GREY, T.C_KEEP, T.C_LOSE, T.C_GAIN, T.C_INC]
    labels = np.array([T.GREY_SHORT, "Keeps evidence", "Loses evidence",
                       "Evidence only without these years", "Increase"], dtype=object)
    legend_items = [(T.C_KEEP, "Keeps evidence"), (T.C_LOSE, "Loses evidence"),
                    (T.C_GAIN, "Evidence only without these years"), (T.C_INC, "Increase"),
                    (T.C_GREY, "Not enough evidence")]
    title = f"Robustness view · {data.VARIANTS[variant][0]}"

cell_codes = codes[pts["ii"], pts["jj"]]
hover_text = labels[np.nan_to_num(cell_codes).astype(int)]
hover = dict(
    lon=pts["lon"], lat=pts["lat"], text=hover_text, customdata=pts["customdata"],
    template=("<b>%{text}</b><br>lat %{y:.2f}°, lon %{x:.2f}°<br>"
              "Sen %{customdata[0]:+.1f} mm/yr (%{customdata[1]:+.1f} %/decade)<br>"
              "p (Hamed–Rao) %{customdata[2]:.3f} · q (BY) %{customdata[3]:.3f}<extra></extra>"),
)

with col_map:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.caption("Hover for values · click a cell to open its details · scroll to zoom, drag to pan")
        fig = charts.raster_map(codes, colors, LAT, LON, hover=hover, selected=S.sel)
        st.plotly_chart(fig, key="map", on_select="rerun", selection_mode="points",
                        width="stretch", config={"scrollZoom": True, "displaylogo": False})
        st.markdown("".join(key_item(c, "", t, T.C_GREY_BD if c == T.C_GREY else None)
                            for c, t in legend_items), unsafe_allow_html=True)

        st.markdown("**Try:**")
        ex = data.example_cells()
        bcols = st.columns(len(ex))
        for bc, (name, (la, lo)) in zip(bcols, ex.items()):
            bc.button(name, on_click=select, args=(la, lo), width="stretch")

        with st.expander("Pick a cell by coordinates"):
            k1, k2, k3 = st.columns([1, 1, 1])
            lat_in = k1.number_input("Latitude (°, south is negative)", float(LAT.min()), float(LAT.max()),
                                     value=S.sel[0] if S.sel else -8.0, step=0.1, format="%.2f")
            lon_in = k2.number_input("Longitude (°, west is negative)", float(LON.min()), float(LON.max()),
                                     value=S.sel[1] if S.sel else -60.0, step=0.1, format="%.2f")
            k3.write("")
            k3.button("Show this cell", on_click=select, args=(lat_in, lon_in), width="stretch")

# ------------------------------------------------------------------ details panel
CL = T.CLASS_LABEL


def tile(label, value, sub):
    return (f"<div class='te-tile'><div class='l'>{label}</div><div class='v'>{value}</div>"
            f"<div class='l'>{sub}</div></div>")


def badge(txt, bg, fg):
    return f"<span class='te-badge' style='background:{bg};color:{fg}'>{txt}</span>"


def details_panel():
    if S.sel is None:
        st.markdown("**Click a place on the map**")
        st.write("Pick any location inside the basin, or use “Try” under the map. The panel shows that cell’s "
                 "trend, how strong the evidence is, whether 24 years could detect it, and whether it survives "
                 "dropping years.")
        return
    c = data.cell_record(*S.sel)
    if c is None or c.get("outside"):
        st.markdown("**Outside the basin**")
        st.write("That point is outside the Amazon basin. Pick a location inside the coloured area.")
        return

    lat_s = f"{abs(c['lat']):.2f}°{'S' if c['lat'] < 0 else 'N'}"
    lon_s = f"{abs(c['lon']):.2f}°{'W' if c['lon'] < 0 else 'E'}"
    st.markdown(f"<div class='te-small'><b>SELECTED CELL · 0.1°</b><br>{lat_s}, {lon_s} · 2001–2024</div>",
                unsafe_allow_html=True)

    ec, slope = c["ec"], c["slope"]
    down = slope < 0
    if ec == data.DOWN:
        head, ev_badge = "Rainfall here decreased, with statistical evidence", badge("Evidence of decrease", "#1E3A8A", "#FFFFFF")
        plain = (f"Over 2001–2024, yearly rainfall in this cell fell by about {abs(slope):.1f} mm per year "
                 f"({abs(c['pct']):.1f}% per decade). This describes the observation period only; it does not say why.")
    elif ec == data.UP:
        head, ev_badge = "Rainfall here increased, with statistical evidence", badge("Evidence of increase", "#C2410C", "#FFFFFF")
        plain = (f"One of only {int(summ['main_cells_up'])} cells in the basin with evidence of an increase. "
                 "Treat it as a local result, not a regional pattern.")
    elif ec == data.GREY:
        head, ev_badge = "Not enough evidence to detect a trend", badge(T.GREY_LONG, "#E5E7EB", "#111827")
        plain = (f"The slope here is {'negative' if down else 'positive'}, but it is not separable from normal "
                 "year-to-year swings after correcting for testing many cells. This does not mean rainfall did not change.")
    else:
        head, ev_badge = "Not enough valid years", badge("Not tested", "#E5E7EB", "#111827")
        plain = f"Fewer than {data.min_valid_years()} valid years, so no trend test was run here."
    st.markdown(f"### {head}")
    dir_txt = "flat" if slope == 0 else ("down" if down else "up")
    st.markdown(badge(f"Direction of Sen's slope: {dir_txt}", "#DBEAFE" if down else "#FFEDD5",
                      "#1E3A8A" if down else "#9A3412") + ev_badge, unsafe_allow_html=True)
    if ec == data.NO_DATA:
        st.write(plain)
        return

    a, b = st.columns(2)
    a.markdown(tile("Sen's slope", T.fmt_signed(slope), f"mm/yr · {T.fmt_signed(c['pct'])} %/decade"), unsafe_allow_html=True)
    b.markdown(tile("Average rainfall", f"{c['mean']:,.0f}", "mm per year"), unsafe_allow_html=True)
    a, b = st.columns(2)
    a.markdown(tile("q (BY, Hamed–Rao)", "< 0.001" if c["q"] < 0.001 else (f"{c['q']:.4f}" if 0.045 < c["q"] < 0.055 else f"{c['q']:.3f}"), "evidence if ≤ 0.05"), unsafe_allow_html=True)
    b.markdown(tile("Valid years", f"{c['n_valid']} / 24", f"minimum {data.min_valid_years()}"), unsafe_allow_html=True)

    # yearly chart
    if c["series"] is not None:
        st.markdown("**Yearly rainfall in this cell**")
        st.caption("Line = yearly total (mm) · dashed = Sen's slope · grey = drought years · pink = 2022–2024")
        col = "#1E3A8A" if ec == data.DOWN else "#C2410C" if ec == data.UP else "#6B7280"
        st.plotly_chart(charts.yearly_series(data.YEARS, c["series"], slope, col), width="stretch",
                        config={"displayModeBar": False})

    if ec == data.DOWN and c["mag"] in (1, 2, 3):
        mt = {1: f"Lowest third of decreases with evidence (< {t1:.1f} %/decade)",
              2: f"Middle third of decreases with evidence ({t1:.1f}–{t2:.1f} %/decade)",
              3: f"Highest third of decreases with evidence (> {t2:.1f} %/decade)"}[c["mag"]]
        st.markdown(f"<div style='background:#EFF6FF;border-radius:10px;padding:10px 12px;font-size:13px'>"
                    f"<b>{mt}</b><br><span class='te-small'>{T.MAG_NOTE}</span></div>", unsafe_allow_html=True)

    # detectability
    if c["ratio"] is not None:
        r = c["ratio"]
        w, colr = min(100, r * 50), (T.C_TEAL if r >= 1 else "#9CA3AF")
        if r < 1:
            rt = ("So a grey result is expected: the change, if real, is too small for this record length."
                  if ec == data.GREY else "Detected anyway – the MDS is an approximate, optimistic guide.")
        else:
            rt = ("Large enough in principle, but it did not pass the strict FDR correction."
                  if ec == data.GREY else "Comfortably above what the record can detect.")
        with st.container(border=True):
            st.markdown("**Could 24 years detect a change this size?**")
            st.markdown(
                "<div style='position:relative;height:14px;background:#F2F5F4;border-radius:999px;margin:6px 0'>"
                f"<div style='position:absolute;left:0;top:0;bottom:0;width:{w:.1f}%;background:{colr};border-radius:999px'></div>"
                "<div style='position:absolute;left:50%;top:-4px;bottom:-4px;border-left:2px solid #111827'></div></div>"
                "<div class='te-small' style='display:flex;justify-content:space-between'><span>0</span>"
                f"<span>detectable size (MDS {c['mds']:.1f} mm/yr)</span><span>2×</span></div>",
                unsafe_allow_html=True)
            st.markdown(f"Observed slope is **{r:.2f}×** the smallest slope a 24-year series could detect here. {rt}")

    # drop-year rows
    with st.container(border=True):
        st.markdown("**Does the result hold when years are dropped?**")
        for k, dc in c["drops"].items():
            if dc is None:
                continue
            same = dc == ec
            res = "Same result" if same else f"Becomes: {CL.get(dc, '—')}"
            colr = T.C_TEAL if same else "#9D174D"
            r1, r2 = st.columns([3, 2])
            r1.button(data.VARIANTS[k][0], key=f"drop_{k}", on_click=pick_variant, args=(k,),
                      width="stretch", type="primary" if variant == k else "secondary")
            r2.markdown(f"<div style='padding-top:8px;font-weight:600;color:{colr};font-size:13px'>{res}</div>",
                        unsafe_allow_html=True)
        st.caption("Click a row to show that version on the map.")

    # exploratory block
    d = c.get("drivers")
    if d:
        with st.container(border=True):
            st.markdown(f"**Year-to-year swings and ocean indices** <span class='te-expl'>{T.EXPLORATORY}</span>",
                        unsafe_allow_html=True)
            r2 = d["r2cv"]
            if r2 is not None and r2 > 0:
                st.write(f"Ocean indices are associated with about {round(r2 * 100)}% of the year-to-year swings "
                         "here (cross-validated R², trend removed).")
                a, b = st.columns(2)
                a.markdown(tile("Pacific (Niño 3.4)", f"{T.fmt_signed(d['b_nino'], 0)} mm", "per 1 SD"), unsafe_allow_html=True)
                b.markdown(tile("Atlantic (TNA)", f"{T.fmt_signed(d['b_tna'], 0)} mm", "per 1 SD"), unsafe_allow_html=True)
            else:
                st.write("Not predictable by ENSO/TNA here (cross-validated R² ≤ 0).")
            st.markdown(f"Trend after accounting for ENSO & TNA: **{T.fmt_signed(slope)} → "
                        f"{T.fmt_signed(d['slope_adj'])} mm/yr** · {CL.get(d['class_adj'], '—')}")
            st.caption(f"{T.ASSOC_NOTE} Sensitivity (naive) version: {T.fmt_signed(d['slope_adj_naive'])} mm/yr · "
                       f"{CL.get(d['class_adj_naive'], '—')}.")
    st.write(plain)


if col_right is not None:
    with col_right:
        with st.container(border=True):
            details_panel()

# ------------------------------------------------------------------ basin card
ba = data.csv("basin_annual.csv")
ew = data.csv("expanding_window.csv")
if ba is not None:
    with st.container(border=True):
        left, right = st.columns(2, gap="large")
        mean = float(ba["basin_mean_mm"].mean())
        with left:
            st.markdown("**Basin rainfall, year by year**")
            st.caption(f"Area-weighted basin mean · each stripe = one year, compared with the 2001–2024 average "
                       f"({mean:,.0f} mm) · blue = drier · orange = wetter")
            st.plotly_chart(charts.stripes(ba["year"], ba["basin_mean_mm"], mean), width="stretch",
                            config={"displayModeBar": False})
        with right:
            if basin is not None:
                st.caption(f"Line = basin mean (mm/yr) · dashed = Sen's slope {T.fmt_signed(basin['sen_mm_yr'])} mm/yr "
                           f"(p = {T.fmt_p(basin['p_hr'])}"
                           f"{': not a supported trend' if not basin['boot_excludes_zero'] else ''}) · "
                           "grey bands = drought years used in the robustness check")
            st.plotly_chart(charts.yearly_series(ba["year"], ba["basin_mean_mm"],
                                                 None if basin is None else float(basin["sen_mm_yr"]),
                                                 T.C_TEAL, last=(), height=220),
                            width="stretch", config={"displayModeBar": False})
        if ew is not None:
            st.markdown("**If the record had ended earlier: basin trend by final year**")
            st.caption("Sen's slope of the basin series from 2001 to each end year (mm/yr)")
            st.plotly_chart(charts.end_year_bars(ew[ew["region"] == "toan_luu_vuc"]), width="stretch",
                            config={"displayModeBar": False})
            st.caption(T.WINDOWS_NOTE + " Full regional version in the Regions tab.")
        st.markdown(f"<div class='te-small'>{T.DROUGHT_SOURCES}</div>", unsafe_allow_html=True)
