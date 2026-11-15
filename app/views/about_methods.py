"""About & methods: central message, how cells are tested, limitations, sources."""
import pandas as pd
import streamlit as st

from lib import data
from lib import text as T

summ = data.summary()
sens = data.json_attr("trend_cells.nc", "sensitivity")
sens.update(data.json_attr("robustness_cells.nc", "cell_sensitivity"))
kept_last = sens.get("drop_2022_24", {}).get("kept_significant_share")
t1, t2 = data.magnitude_cutoffs()
moran = data.morans_i()

st.markdown("## About & methods")

# ---- hero with headline numbers (all read from files)
ew = data.csv("expanding_window.csv")
tiles = [(T.pct(summ["main_area_share_down"]), "of basin area: decrease with evidence")]
if kept_last is not None:
    tiles.append((T.pct(kept_last), "of those cells keep evidence without 2022–2024"))
if ew is not None:
    b = ew[ew["region"] == "toan_luu_vuc"].set_index("end_year")["sen_mm_yr"]
    if 2021 in b and 2024 in b:
        tiles.append((f"{T.fmt_signed(b[2021])} → {T.fmt_signed(b[2024])}", "basin slope (mm/yr) ending in 2021 vs 2024"))
st.markdown(
    "<div class='te-hero'><div class='k'>WHAT THIS EXPLORER SHOWS</div>"
    f"<div class='m'>{T.CENTRAL_MESSAGE}</div>"
    f"<div style='display:grid;grid-template-columns:repeat({len(tiles)},minmax(0,1fr));gap:10px'>"
    + "".join(f"<div class='t'><b>{v}</b><div>{t}</div></div>" for v, t in tiles)
    + "</div></div>", unsafe_allow_html=True)
st.write("")

with st.container(border=True):
    st.markdown("#### How each cell is tested")
    st.markdown(f"""
1. Yearly totals from GPM IMERG Final V07 monthly (mm/hr × hours per month, summed over the calendar year), 0.1° grid, 2001–2024. Cells need at least {data.min_valid_years()} valid years.
2. Trend size: Sen's slope (mm/yr and % per decade).
3. Trend test: Mann–Kendall with the Hamed–Rao correction for year-to-year persistence.
4. Many cells are tested at once, and neighbouring cells are not independent{f" (Moran's I of the slope = {moran:.2f})" if moran is not None else ""}, so q-values use Benjamini–Yekutieli.
5. **The only evidence rule: q ≤ 0.05.** The map, the overview and the details panel all read the same class.
6. Decrease magnitude: thirds of the decreasing cells with evidence (cut-offs {t1:.1f} and {t2:.1f} %/decade, read from the data file). Relative, not absolute severity.
""")

with st.container(border=True):
    st.markdown("#### Why most of the basin is grey")
    st.write(f"Grey means “not enough evidence to detect a trend”, not “no change”. The minimum detectable slope (MDS) "
             f"of a 24-year series has a median of about {data.mds_median():.1f} mm/yr across the basin, so smaller real "
             "changes are usually invisible. The MDS is an optimistic guide (it ignores the FDR step and persistence).")

with st.container(border=True):
    st.markdown("#### Robustness checks")
    names = [("drop_drought", "Drop drought years"), ("drop_2022_24", "Drop 2022–2024"),
             ("drop_2001_03", "Drop 2001–2003"), ("drop_2011_13", "Drop 2011–2013")]
    cols = st.columns(len(names))
    for col, (k, lab) in zip(cols, names):
        s = sens.get(k, {}).get("kept_significant_share")
        bg = "#FDF2F8" if k == "drop_2022_24" else "#F7F9F8"
        col.markdown(f"<div class='te-tile' style='background:{bg}'><div class='l'>{lab}</div>"
                     f"<div class='v'>{'—' if s is None else T.pct(s)}</div><div class='l'>keep evidence</div></div>",
                     unsafe_allow_html=True)
    n_ev = int(summ["main_cells_down"] + summ["main_cells_up"])
    st.caption(f"Share of the {n_ev:,} cells with evidence in the full period that keep the same class. Regional slopes "
               "also tested with a block bootstrap, a hydrological year, and dropping every 3-year window (Regions tab).")

with st.container(border=True):
    st.markdown(f"#### Exploratory driver analysis: main and sensitivity versions <span class='te-expl'>{T.EXPLORATORY}</span>",
                unsafe_allow_html=True)
    nd = int(summ["main_cells_down"])
    a, b = st.columns(2)
    a.markdown(f"**Joint (main)** · class_adj_joint  \n"
               f"{T.pct(summ['adj_joint_main_decreasing_kept_share'])} of decreasing cells remain "
               f"({int(summ['adj_joint_main_decreasing_kept']):,} / {nd:,}); decrease area {T.pct(summ['adj_joint_decreasing_area_share'])}  \n"
               "Coefficients from rainfall ~ year + Niño 3.4 + TNA, so a trend shared with the indices is not attributed to them.")
    b.markdown(f"**Naive (sensitivity)** · class_adj_naive  \n"
               f"{T.pct(summ['adj_naive_main_decreasing_kept_share'])} remain "
               f"({int(summ['adj_naive_main_decreasing_kept']):,} / {nd:,}); decrease area {T.pct(summ['adj_naive_decreasing_area_share'])}  \n"
               "Coefficients from rainfall ~ Niño 3.4 + TNA (no year term), which removes the whole association.")
    st.write("Adjusted trend: rainfall minus the fitted contributions of the Niño 3.4 and tropical North Atlantic (TNA) "
             "indices. The adjusted series is then tested exactly as the main analysis (Sen's slope, Mann–Kendall with "
             "Hamed–Rao variance correction, Benjamini–Yekutieli FDR). Descriptive statistics, not causal attribution.")
    st.caption(f"Indices: Niño 3.4 and TNA from MERRA-2 surface temperature, same year, standardised; no trend in either "
               f"(p = {summ['index_nino34_p_hr']:.2f}, {summ['index_tna_p_hr']:.2f}), nearly independent "
               f"(VIF {summ['vif_detrended']:.2f}). Only the cross-validated R² is reported (median {summ['r2_cv_median']:.2f}). "
               "Newly significant cells reflect reduced noise, not a new trend. Associations, not causes.")

with st.container(border=True):
    st.markdown("#### Limitations")
    last_txt = f"only {T.pct(kept_last)} of cells keep evidence" if kept_last is not None else "far fewer cells keep evidence"
    st.markdown(f"""
- 24 years is a short record; only fairly large changes can be detected.
- The decrease signal is sensitive to 2022–2024: without those years {last_txt} ({T.pct(summ['drop2022_24_main_decreasing_kept_share_adj'])} of decreasing cells after the ENSO/TNA adjustment), and using 2001–2021 no region reaches p < 0.05.
- Cells are strongly spatially dependent{f" (Moran's I = {moran:.2f})" if moran is not None else ""}, which is why the conservative BY correction is used.
- The temporal homogeneity of IMERG (changes in contributing satellite sensors over time) has not been checked.
- Results describe 2001–2024 only. No causes are attributed and nothing is forecast.
""")

with st.container(border=True):
    st.markdown("#### Data, code and tools")
    st.markdown("""
- **GPM IMERG Final Precipitation L3, 1 month, 0.1° V07 (GPM_3IMERGM)** · NASA GES DISC · doi:[10.5067/GPM/IMERG/3B-MONTH/07](https://doi.org/10.5067/GPM/IMERG/3B-MONTH/07) · rainfall
- **MERRA-2 tavgM_2d_slv_Nx (M2TMNXSLV) V5.12.4, variable TS** · NASA GMAO via GES DISC · doi:[10.5067/AP1B0BA5PD2K](https://doi.org/10.5067/AP1B0BA5PD2K) · Niño 3.4 and TNA indices
- **HydroBASINS v1c, South America, level 3 (hybas_sa_lev03_v1c)** · Lehner & Grill (2013), *Hydrological Processes* 27(15): 2171–2186, doi:[10.1002/hyp.9740](https://doi.org/10.1002/hyp.9740) · Amazon basin outline and mask
- **Code:** [github.com/PTBK147/trendlab](https://github.com/PTBK147/trendlab)
""")
    st.markdown("##### Software")
    sw = pd.DataFrame([
        ("Python", "3.14.2", "language", "python.org"), ("NumPy", "2.5.4", "arrays, regression", "numpy.org"),
        ("SciPy", "1.18.1", "statistics", "scipy.org"), ("pandas", "3.1.1", "tables", "pandas.pydata.org"),
        ("xarray", "2026.9.0", "gridded data", "xarray.dev"), ("netCDF4", "1.7.5.1.post1", "NetCDF I/O", "unidata.github.io/netcdf4-python"),
        ("Matplotlib", "3.11.2", "figures", "matplotlib.org"), ("GeoPandas", "1.2.0", "basin shapefile", "geopandas.org"),
        ("regionmask", "0.13.0", "basin mask", "regionmask.readthedocs.io"), ("earthaccess", "0.19.0", "NASA download", "earthaccess.readthedocs.io"),
        ("Streamlit", "1.67.0", "interface", "streamlit.io"), ("Plotly", "7.1.0", "interactive charts", "plotly.com/python"),
    ], columns=["Library", "Version", "Used for", "Link"])
    st.dataframe(sw, hide_index=True, width="stretch")
    st.markdown("##### References used in the interface")
    st.markdown("""
- Levine, N. M. et al. (2016). Ecosystem heterogeneity determines the ecological resilience of the Amazon to climate change. *PNAS* 113(3), 793–797. Used for the 100 mm dry-month threshold.
- Brando, P. M. et al. (2010). *PNAS* 107(33), 14685–14690. Used for the 2005 drought.
- Lewis, S. L. et al. (2011). The 2010 Amazon drought. *Science* 331(6017), 554.
- Jiménez-Muñoz, J. C. et al. (2016). Record-breaking warming and extreme drought in the Amazon rainforest during the course of El Niño 2015–2016. *Scientific Reports* 6, 33130. doi:10.1038/srep33130
- Marengo, J. A. et al. (2024). The drought of Amazonia in 2023–2024. *American Journal of Climate Change*. doi:10.4236/ajcc.2024.133026
""")
    st.caption("These papers document that the drought years happened; this app does not use them to attribute causes.")
