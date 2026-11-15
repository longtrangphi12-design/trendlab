"""Regions: is the decrease regional or local?"""
import streamlit as st

from lib import charts, data
from lib import text as T

MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.markdown("## Regions: is the decrease regional or local?")
st.write("Area-weighted yearly rainfall for three latitude bands and the whole basin, each tested as one series. "
         "The bands were chosen after looking at the maps, so always read them next to the whole-basin result.")

reg = data.csv("regions_robustness.csv")
if reg is None:
    st.warning("regions_robustness.csv not found – this page needs it.")
    st.stop()

adj = data.csv("adjusted_robustness.csv")
e21 = {} if adj is None else {r["region"]: r for _, r in adj.iterrows()}

# ---- dot + interval chart
with st.container(border=True):
    st.markdown("**Sen's slope with its bootstrap interval (mm per year)**")
    st.caption("Dot = Sen's slope · bar = bootstrap interval · an interval that does not cross 0 supports a trend")
    st.plotly_chart(charts.region_intervals(reg), width="stretch", config={"displayModeBar": False})

# ---- four cards
cols = st.columns(len(reg))
for col, (_, r) in zip(cols, reg.iterrows()):
    with col, st.container(border=True):
        st.markdown(f"**{r['name']}**  \n<span class='te-small'>{r['sub']} · {int(r['n_cells']):,} cells</span>",
                    unsafe_allow_html=True)
        if r["boot_excludes_zero"]:
            word = "Decrease" if r["sen_mm_yr"] < 0 else "Increase"
            st.markdown(f"<span class='te-badge' style='background:#1E3A8A;color:#fff'>{word} supported "
                        "(interval excludes 0)</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='te-badge' style='background:#E5E7EB;color:#111827'>No trend supported</span>",
                        unsafe_allow_html=True)
        hm = int(r["hydro_start_month"])
        rows = [
            ("Sen's slope", f"**{T.fmt_signed(r['sen_mm_yr'])} mm/yr**"),
            ("p (Hamed–Rao)", T.fmt_p(r["p_hr"])),
            ("Bootstrap interval", f"{T.fmt_signed(r['boot_lo'])} to {T.fmt_signed(r['boot_hi'])}"),
            ("Detectable size (MDS)", f"{r['mds_mm_yr']:.1f} mm/yr"),
            ("Hydrological year", f"{T.fmt_signed(r['hydro_sen'])} (p = {T.fmt_p(r['hydro_p_hr'])}, from {MON[hm]})"),
            ("Average rainfall", f"{r['mean_mm']:,.0f} mm/yr"),
        ]
        st.markdown("\n".join(f"- {a}: {b}" for a, b in rows))
        if r["region"] in e21:
            a = e21[r["region"]]
            st.markdown(f"<div style='background:#FDF2F8;border-radius:6px;padding:4px 8px;color:#9D174D;font-size:13px'>"
                        f"Record ending 2021: <b>{T.fmt_signed(a['sen_raw_2001_2021'])} mm/yr "
                        f"(p = {T.fmt_p(a['p_raw'])})</b></div>", unsafe_allow_html=True)
        flips = int(r["drop3_sign_flips"])
        if flips == 0:
            ft = "Never changes sign."
        else:
            wins = str(r["drop3_flip_windows"]).replace("-", "–").split(";")
            ft = f"Sign flips in {'one window (' + wins[0] + ')' if flips == 1 else f'{flips} windows'}."
        st.caption(f"Drop any 3 consecutive years: slope ranges {T.fmt_signed(r['drop3_min'])} to "
                   f"{T.fmt_signed(r['drop3_max'])}. {ft}")

# ---- end-year lines
ew = data.csv("expanding_window.csv")
if ew is not None:
    with st.container(border=True):
        st.markdown("**How the trend depends on the final years**")
        st.caption("Sen's slope from 2001 to each end year (mm/yr).")
        st.plotly_chart(charts.end_year_lines(ew), width="stretch", config={"displayModeBar": False})
        st.caption(T.WINDOWS_NOTE)

# ---- 2001–2021 check
if adj is not None:
    with st.container(border=True):
        st.markdown("**Check: 2001–2021 only, before and after accounting for ENSO & TNA**")
        cc = st.columns(len(adj))
        for col, (_, a) in zip(cc, adj.iterrows()):
            col.markdown(
                f"<div class='te-tile'><b>{a['name']}</b><br>"
                f"<span class='l'>Raw</span> {T.fmt_signed(a['sen_raw_2001_2021'], 2)} (p = {a['p_raw']:.2f})<br>"
                f"<span class='l'>After ENSO &amp; TNA</span> {T.fmt_signed(a['sen_adj_2001_2021'], 2)} "
                f"(p = {a['p_adj']:.2f})</div>", unsafe_allow_html=True)
        if (adj[["p_raw", "p_adj"]] >= 0.05).all().all():
            st.markdown("**No region reaches p < 0.05 without 2022–2024**, with or without the ocean-index adjustment. "
                        "Slopes in mm/yr; p from Mann–Kendall with Hamed–Rao.")
        else:
            st.markdown("Slopes in mm/yr; p from Mann–Kendall with Hamed–Rao.")

st.caption("Reading guide: with the full record, the middle and southern bands show decreases whose bootstrap "
           "intervals exclude zero, while the north and the whole basin do not. These decreases depend on 2022–2024, "
           "the regional p-values sit just under 0.05, and the bands were drawn after seeing the map, so treat them as "
           "descriptions of the map, not as separate discoveries. Hydrological year = the same test with the year "
           "starting in the month shown (23 years).")
