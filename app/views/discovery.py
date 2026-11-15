"""Discovery map: 4 modes (Detective, Grey Zone, Time Machine, Evidence), map tools
(Inspect, Compare A/B, Draw area, Swipe 2024 | 2021), hot/cold hints, rewards, journal.

Discovery never re-tests anything: it reads, bins and counts the result files.
"""
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, game
from lib import text as T

S = st.session_state
DEFAULTS = dict(d_mode="Detective", d_tool="Inspect", d_layer="evidence", d_sel=None, d_a=None, d_b=None,
                d_next="a", d_box=None, d_mi=0, d_done={}, d_fb=None, d_hint=0, d_journal=[], d_sw_pred=50,
                d_sw_rev=False, d_year=2024, d_playing=False, d_next_year=None, d_tans={}, d_ci="Claim 1",
                d_verdict={}, d_seen={}, d_sig=None)
for k, v in DEFAULTS.items():
    S.setdefault(k, v)

summ = data.summary()
g = data.grid("trend_cells.nc")
LAT, LON = g["lat"], g["lon"]
t1, t2 = data.magnitude_cutoffs()
P = data.pools()
INC_IJ = P["inc"]["ij"]
reg = data.csv("regions_robustness.csv")
basin = reg[reg["region"] == "toan_luu_vuc"].iloc[0]
adj = data.csv("adjusted_robustness.csv")
sens = data.json_attr("trend_cells.nc", "sensitivity")
N = dict(t2=t2, n_cells=data.n_cells(), n_down=int(summ["main_cells_down"]), n_up=int(summ["main_cells_up"]),
         fragile=P["fragile"]["n"], robust=P["robust"]["n"], area_down=summ["main_area_share_down"],
         basin_sen=float(basin["sen_mm_yr"]), basin_p=float(basin["p_hr"]),
         kept_2224=sens.get("drop_2022_24", {}).get("kept_significant_share", np.nan),
         none_2021=bool(adj is not None and (adj[["p_raw", "p_adj"]] >= 0.05).all().all()),
         mds_median=data.mds_median())
MISSIONS = game.missions(N)
CLAIMS = {c["short"]: c for c in game.claims(N)}

# apply a year set by the play loop in the previous run (before the slider exists)
if S.d_next_year is not None:
    S.d_year = S.d_next_year
    S.d_next_year = None


# ------------------------------------------------------------------ helpers
def seen(key):
    S.d_seen = {**S.d_seen, key: True}


def event_sig(ev):
    if not ev:
        return None
    sel = ev.get("selection", {})
    if sel.get("box"):
        b = sel["box"][-1]
        return ("box", tuple(np.round(b.get("x", []), 3)), tuple(np.round(b.get("y", []), 3)))
    if sel.get("points"):
        p = sel["points"][-1]
        return ("pt", round(p["x"], 2), round(p["y"], 2))
    return None


def mark_event_seen():
    """After a mode/tool change, ignore the selection that was already on the map."""
    S.d_sig = event_sig(S.get("d_map"))


def on_mode():
    m = S.d_mode or "Detective"
    S.d_mode, S.d_fb, S.d_playing = m, None, False
    if m == "Detective":
        S.d_tool = "Inspect"
    elif m == "Grey Zone":
        S.d_layer = "q"
        seen("layer_q")
    elif m == "Evidence":
        S.d_layer = "evidence"
        seen("layer_evidence")
    mark_event_seen()


def on_tool():
    S.d_tool = S.d_tool or "Inspect"
    S.d_fb = None
    if S.d_tool.startswith("Swipe"):
        seen("swipe")
    mark_event_seen()


def on_layer():
    S.d_layer = S.d_layer or "evidence"
    seen("layer_" + S.d_layer)


def clear_all():
    S.d_sel = S.d_a = S.d_b = S.d_box = S.d_fb = None
    S.d_next = "a"
    mark_event_seen()


def cell(sel):
    if sel is None:
        return None
    c = data.cell_record(*sel)
    return None if (c is None or c.get("outside")) else c


def select(lat, lon):
    """A click on the map: Compare slots, Detective check, Grey Zone progress, journal."""
    c = data.cell_record(lat, lon)
    S.d_sel, S.d_fb = (lat, lon), None
    if S.d_tool == "Compare A/B":
        setattr(S, "d_" + S.d_next, (lat, lon))
        S.d_next = "b" if S.d_next == "a" else "a"
        seen("compare")
    if c is None or c.get("outside"):
        S.d_fb = dict(ok=False, title="Outside the basin.", text="Pick a location inside the coloured area.")
        return
    if c["ec"] == data.GREY:
        seen("grey")
    if S.d_mode == "Detective" and S.d_tool == "Inspect":
        m = MISSIONS[S.d_mi]
        if game.check_mission(m["id"], c, INC_IJ):
            S.d_done = {**S.d_done, m["id"]: True}
            S.d_fb = dict(ok=True, title="Signal discovered!", text=game.success_text(m["id"], c, N),
                          heat="FOUND!", reward=game.REWARD[m["id"]])
            S.d_journal = game.add_journal(S.d_journal, m["id"], *m["journal"][:1],
                                           f"Mission 0{S.d_mi + 1} · Completed", m["journal"][1])
        else:
            h = game.heat(m["id"], c, INC_IJ)
            S.d_fb = dict(ok=False, title=h[1] if h else "Not this one.", text=game.miss_text(c),
                          heat=h[0] if h else None)
    if (S.d_mode == "Grey Zone" and S.d_seen.get("layer_q") and c["ec"] == data.GREY and c["slope"] < 0
            and 0.05 < c["q"] <= 0.15):
        S.d_journal = game.add_journal(
            S.d_journal, "grey", "The grey zone", "Grey Zone · Completed",
            f"A cell with {T.fmt_signed(c['pct'])}% per decade and q = {game.fq(c['q'])} stays grey. "
            "Not enough evidence does not mean no rainfall change.")


# ------------------------------------------------------------------ handle the last map interaction
ev = S.get("d_map")
sig = event_sig(ev)
if sig is not None and sig != S.d_sig:
    S.d_sig = sig
    if sig[0] == "box" and S.d_tool == "Draw area":
        b = ev["selection"]["box"][-1]
        S.d_box = (min(b["y"]), max(b["y"]), min(b["x"]), max(b["x"]))
        seen("area")
    elif sig[0] == "pt" and S.d_tool in ("Inspect", "Compare A/B") and S.d_mode != "Time Machine":
        p = ev["selection"]["points"][-1]
        select(float(p["y"]), float(p["x"]))

mode, tool, layer = S.d_mode, S.d_tool, S.d_layer
is_time = mode == "Time Machine"
is_swipe = tool.startswith("Swipe") and not is_time

# ------------------------------------------------------------------ header + mode switch
st.markdown("## Discovery")
st.segmented_control("Experience", list(game.MODES), key="d_mode", on_change=on_mode,
                     format_func=lambda m: f"{m} · {game.MODES[m][0]}")

left, center, right = st.columns([1.15, 2.6, 1.35], gap="medium")
NOCFG = {"displayModeBar": False}


def badge(txt, bg, fg="#FFFFFF"):
    return f"<span class='te-badge' style='background:{bg};color:{fg}'>{txt}</span>"


def checklist(items):
    """items: [(done, text)] -> HTML list with ✓ / numbers."""
    html = ""
    for i, (ok, t) in enumerate(items):
        dot = (f"<span style='display:inline-flex;width:20px;height:20px;border-radius:999px;align-items:center;"
               f"justify-content:center;font-size:11px;font-weight:700;color:#fff;background:"
               f"{'#0D9488' if ok else '#CBD5E1'}'>{'✓' if ok else i + 1}</span>")
        html += f"<div style='display:flex;gap:8px;margin:4px 0;font-size:13px;color:{'#111827' if ok else '#4B5563'}'>{dot}{t}</div>"
    st.markdown(html, unsafe_allow_html=True)


# ================================================================== LEFT: mode panel + journal
with left:
    with st.container(border=True):
        col = game.MODES[mode][1]
        if mode == "Detective":
            st.markdown(f"<div style='font-size:12px;font-weight:700;letter-spacing:.05em;color:{col}'>"
                        f"CLIMATE DETECTIVE · MISSION {S.d_mi + 1} OF 4</div>", unsafe_allow_html=True)
            nd = len(S.d_done)
            st.progress(nd / 4, text=f"Investigation progress {nd} / 4")
            if nd == 4:
                st.success("**Detective Completed** – all four signals found: strong, fragile, rock-solid and rare.")
            for i, m in enumerate(MISSIONS):
                done = m["id"] in S.d_done
                label = f"{'✓' if done else i + 1} · {m['title']}" + (f"  —  {game.REWARD[m['id']]}" if done else "")
                if st.button(label, key=f"mis_{i}", width="stretch", type="primary" if i == S.d_mi else "secondary"):
                    S.d_mi, S.d_fb, S.d_hint, S.d_tool = i, None, 0, "Inspect"
                    st.rerun()
            m = MISSIONS[S.d_mi]
            st.markdown(f"#### {m['title']}")
            st.write(m["task"])
            if tool != "Inspect":
                if st.button("This mission needs the Inspect tool – switch"):
                    S.d_tool = "Inspect"
                    mark_event_seen()
                    st.rerun()
            h1, h2 = st.columns(2)
            if h1.button("Hint 1", width="stretch"):
                S.d_hint = 0 if S.d_hint >= 1 else 1
            if h2.button("Reveal clue on map", width="stretch"):
                S.d_hint = 1 if S.d_hint >= 2 else 2
            if S.d_hint >= 1:
                st.warning(m["hint"])
            if m["id"] in S.d_done and len(S.d_done) < 4:
                if st.button("Next mission →", type="primary"):
                    for k in range(1, 5):
                        j = (S.d_mi + k) % 4
                        if MISSIONS[j]["id"] not in S.d_done:
                            S.d_mi = j
                            break
                    S.d_fb, S.d_hint, S.d_tool = None, 0, "Inspect"
                    st.rerun()

        elif mode == "Grey Zone":
            st.markdown(f"<div style='font-size:12px;font-weight:700;letter-spacing:.05em;color:{col}'>"
                        "GREY ZONE MYSTERY</div>", unsafe_allow_html=True)
            st.markdown("#### Find the hidden signal")
            st.write("Find a grey cell whose rainfall is going down and whose q-value is close to the evidence line "
                     "(between 0.05 and 0.15). Then work out why it stays grey.")
            c = cell(S.d_sel)
            g1 = bool(S.d_seen.get("layer_q"))
            g2 = bool(c and c["ec"] == data.GREY)
            g3 = bool(g2 and c["slope"] < 0)
            g4 = bool(g3 and 0.05 < c["q"] <= 0.15)
            checklist([(g1, "Open the q-value layer"), (g2, "Select a grey cell (not enough evidence)"),
                       (g3, "Its Sen’s slope is negative (rainfall going down)"),
                       (g4, "Its q is between 0.05 and 0.15 – close to the line")])
            n_ok = sum([g1, g2, g3, g4])
            st.progress(n_ok / 4, text=f"Progress {n_ok} / 4")
            if g4:
                st.info(f"Hidden signal found: {T.fmt_signed(c['pct'])}% per decade with q = {game.fq(c['q'])} "
                        f"(raw p = {game.fq(c['p_hr'])}). The raw test alone looks promising, but after correcting for "
                        f"{N['n_cells']:,} tests the bar is much higher. Its slope is {c['ratio']:.2f}× the size a 24-year "
                        "record could reliably detect here. Grey does not mean “no change” – it means this record cannot "
                        "yet tell the change apart from noise.")
            st.caption(f"{P['grey_near']['n']:,} grey cells match. Bonus: switch to Compare and pick a dark-blue cell "
                       "with a similar slope. Same size, different evidence.")

        elif mode == "Time Machine":
            st.markdown(f"<div style='font-size:12px;font-weight:700;letter-spacing:.05em;color:{col}'>"
                        "TIME MACHINE</div>", unsafe_allow_html=True)
            st.markdown("#### Travel through 24 years")
            st.write("Each map shows that year's rainfall compared with the cell's own 2001–2024 average. "
                     "Press play, or drag the slider.")
            for q in game.time_questions(data.time_stats()):
                with st.container(border=True):
                    a = st.radio(q["q"], q["opts"], index=None, key=f"tq_{q['id']}")
                    if a is not None:
                        (st.success if a == q["ans"] else st.error)(
                            ("Correct. " if a == q["ans"] else "Not quite. ") + q["fb"])

        else:  # Evidence
            st.markdown(f"<div style='font-size:12px;font-weight:700;letter-spacing:.05em;color:{col}'>"
                        "EVIDENCE CHALLENGE</div>", unsafe_allow_html=True)
            ci = st.segmented_control("Claim", list(CLAIMS), key="d_ci",
                                      format_func=lambda k: k + (" ✓" if S.d_verdict.get(k) == CLAIMS[k]["ans"] else ""))
            ci = ci or "Claim 1"
            cl = CLAIMS[ci]
            st.caption("Claim to test")
            st.markdown(f"#### “{cl['claim']}”")
            st.markdown("**Collect evidence on the map**")
            checklist([(bool(S.d_seen.get(k)), t) for k, t in cl["steps"]])
            v = st.radio("Your verdict", cl["opts"], index=None, key=f"verdict_{ci}")
            if v is not None:
                vi = cl["opts"].index(v)
                S.d_verdict = {**S.d_verdict, ci: vi}
                n_seen = sum(bool(S.d_seen.get(k)) for k, _ in cl["steps"])
                msg = ("Good call. " if vi == cl["ans"] else "Not quite. ") + cl["fb"]
                if n_seen < len(cl["steps"]):
                    msg += f" (You collected {n_seen} of {len(cl['steps'])} pieces of evidence – try the rest on the map.)"
                (st.success if vi == cl["ans"] else st.error)(msg)
                if vi == cl["ans"]:
                    S.d_journal = game.add_journal(S.d_journal, "claim" + ci, cl["title"],
                                                   "Evidence Challenge · Completed", cl["fb"].split(". ")[0] + ".")

    with st.container(border=True):
        st.markdown(f"**MY DISCOVERY JOURNAL** <span class='te-small'>· {len(S.d_journal)} discoveries</span>",
                    unsafe_allow_html=True)
        if not S.d_journal:
            st.caption("Complete a mission, solve the Grey Zone, win a claim or save an area – your findings land here.")
        for j in S.d_journal:
            st.markdown(f"<div class='te-tile' style='margin-bottom:6px'><b>{j['title']}</b>"
                        f"<div style='font-size:11px;font-weight:600;color:{T.C_TEAL}'>{j['tag']}</div>"
                        f"<div style='font-size:12px;color:#374151'>{j['text']}</div></div>", unsafe_allow_html=True)


# ================================================================== CENTER: tools + map
with center:
    with st.container(border=True):
        a1, a2 = st.columns([4, 1])
        a1.segmented_control("Tool", game.TOOLS, key="d_tool", on_change=on_tool)
        a2.write("")
        a2.button("Clear selection", on_click=clear_all, width="stretch")
        tool = S.d_tool or "Inspect"
        is_swipe = tool.startswith("Swipe") and not is_time

        if is_time:
            p1, p2 = st.columns([1, 6])
            if p1.button("⏸ Pause" if S.d_playing else "▶ Play", width="stretch"):
                S.d_playing = not S.d_playing
                if S.d_playing:
                    seen("play")
                    if S.d_year >= 2024:
                        S.d_next_year = 2001
                st.rerun()
            p2.slider("Year", 2001, 2024, key="d_year")
        elif not is_swipe:
            st.segmented_control("Look at", list(game.LAYERS), key="d_layer", on_change=on_layer,
                                 format_func=lambda k: game.LAYERS[k][0])
        layer = S.d_layer or "evidence"

        # ---- build the map
        hover = None
        if is_time:
            codes, leg = data.anomaly_codes(S.d_year), game.ANOMALY
            title = f"{S.d_year}: rainfall vs each cell's own 2001–2024 average"
        elif is_swipe:
            leg = game.SWIPE
            if S.d_sw_rev:
                split = st.slider("Move the divider (longitude): left = record ending 2024 · right = ending 2021",
                                  float(LON.min()), float(LON.max()), -62.0, 0.5, format="%.1f°")
                full, short = data.swipe_codes("2024"), data.swipe_codes("2021")
                codes = np.where(LON[None, :] < split, full, short)
                title = "Swipe: 2001–2024 (left) | 2001–2021 (right)"
            else:
                codes = data.swipe_codes("2024")
                title = "2001–2024 · predict first, then reveal the 2001–2021 version"
        else:
            codes, leg = data.layer_codes(layer), game.LAYERS[layer][1]
            title = f"Layer: {game.LAYERS[layer][0]}"
            pts = data.basin_points()
            lab = np.array([game.CL.get(int(e), "") for e in (1, 2, 3)], dtype=object)
            hover = dict(lon=pts["lon"], lat=pts["lat"], text=lab[pts["ec"] - 1], customdata=pts["customdata"],
                         template=("<b>%{text}</b><br>lat %{y:.2f}°, lon %{x:.2f}°<br>"
                                   "Sen %{customdata[0]:+.1f} mm/yr (%{customdata[1]:+.1f} %/decade)<br>"
                                   "p %{customdata[2]:.3f} · q %{customdata[3]:.3f}<extra></extra>"))
        colors = [c for c, _ in leg]
        fig = charts.raster_map(codes, colors, LAT, LON, hover=hover, bands=False, height=600)

        if is_swipe and S.d_sw_rev:
            fig.add_vline(x=split, line=dict(color="#111827", width=3))
            n24 = int(summ["main_cells_down"])
            n21 = data.class_count("adjusted_drop_cells.nc", "class_raw_2001_2021", data.DOWN)
            fig.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper", showarrow=False, xanchor="left",
                               text=f"<b>2001–2024 · {n24:,} decreasing cells</b>", bgcolor="#111827",
                               font=dict(color="white", size=12))
            fig.add_annotation(x=0.99, y=0.99, xref="paper", yref="paper", showarrow=False, xanchor="right",
                               text=f"<b>2001–2021 · {n21:,} decreasing cells</b>", bgcolor="white",
                               bordercolor="#111827", font=dict(color="#111827", size=12))
        # hint ring
        if mode == "Detective" and S.d_hint >= 2 and not is_time:
            rl, ro, rh, rw = data.ring(MISSIONS[S.d_mi]["id"])
            fig.add_shape(type="circle", x0=ro - rw, x1=ro + rw, y0=rl - rh, y1=rl + rh,
                          line=dict(color="#D97706", width=3, dash="dash"))
        # selection marks
        def mark(sel, colr, label):
            fig.add_trace(go.Scatter(x=[sel[1]], y=[sel[0]], mode="markers+text", text=[label],
                                     textposition="top center", hoverinfo="skip",
                                     textfont=dict(color=colr, size=13), marker=dict(
                                         symbol="square-open", size=15, color=colr, line=dict(width=3))))
        if not is_time and not is_swipe:
            if tool == "Compare A/B":
                if S.d_a:
                    mark(S.d_a, T.C_TEAL, "<b>A</b>")
                if S.d_b:
                    mark(S.d_b, "#7C3AED", "<b>B</b>")
            elif tool == "Inspect" and S.d_sel:
                mark(S.d_sel, "#000000", "")
            if tool == "Draw area" and S.d_box:
                l0, l1, o0, o1 = S.d_box
                fig.add_shape(type="rect", x0=o0, x1=o1, y0=l0, y1=l1,
                              line=dict(color="#111827", width=2), fillcolor="rgba(17,24,39,0.08)")

        st.markdown(f"**{title}**")
        sel_mode = "box" if tool == "Draw area" and not is_time else "points"
        if sel_mode == "box":
            fig.update_layout(dragmode="select")   # drag draws a box instead of panning
        st.plotly_chart(fig, key="d_map", on_select="rerun", selection_mode=sel_mode, width="stretch",
                        config={"scrollZoom": True, "displaylogo": False})

        # legend with the selected cell's item highlighted
        hi = None
        if not is_time and not is_swipe and tool == "Inspect":
            hi = game.legend_index(layer, cell(S.d_sel))
        st.markdown("".join(
            f"<span class='te-key' style='padding:2px 6px;border-radius:6px;"
            f"{'background:#FEF3C7;outline:2px solid #111827;font-weight:700;' if i == hi else ''}'>"
            f"<span class='te-sw' style='background:{c};border:1px solid #D1D5DB'></span>{t}</span>"
            for i, (c, t) in enumerate(leg)), unsafe_allow_html=True)
        help_txt = {"Inspect": "Hover for a quick look · click to select",
                    "Compare A/B": "Click two cells: first A, then B",
                    "Draw area": "Drag on the map to draw a box (box-select)",
                    "Swipe 2024 | 2021": "Predict, reveal, then move the divider"}[tool]
        st.caption("Time Machine: drag the slider or press play" if is_time else help_txt)

        if tool == "Draw area" and not is_time:
            with st.expander("Or set the box with sliders"):
                la = st.slider("Latitude range", float(LAT.min()), float(LAT.max()), (-14.0, -6.0), 0.1)
                lo = st.slider("Longitude range", float(LON.min()), float(LON.max()), (-66.0, -56.0), 0.1)
                if st.button("Use this box"):
                    S.d_box = (la[0], la[1], lo[0], lo[1])
                    seen("area")
                    st.rerun()


# ================================================================== RIGHT: findings
with right:
    with st.container(border=True):
        if is_time:
            ts = data.time_stats().set_index("year")
            r = ts.loc[S.d_year]
            st.markdown(f"<div style='font-size:12px;font-weight:700;letter-spacing:.05em;color:#6D28D9'>"
                        f"{S.d_year} AT A GLANCE</div>", unsafe_allow_html=True)
            x1, x2 = st.columns(2)
            x1.markdown(f"<div class='te-tile'><div class='l'>Basin rainfall</div><div class='v'>{r['basin_mm']:,.0f}</div>"
                        "<div class='l'>mm</div></div>", unsafe_allow_html=True)
            x2.markdown(f"<div class='te-tile'><div class='l'>Area drier than usual</div><div class='v'>"
                        f"{T.pct(r['dry_share'], 0)}</div></div>", unsafe_allow_html=True)
            x1, x2 = st.columns(2)
            x1.markdown(f"<div class='te-tile'><div class='l'>East of 62°W</div><div class='v'>{T.fmt_signed(r['east'])}%</div></div>",
                        unsafe_allow_html=True)
            x2.markdown(f"<div class='te-tile'><div class='l'>West of 62°W</div><div class='v'>{T.fmt_signed(r['west'])}%</div></div>",
                        unsafe_allow_html=True)
            st.caption("Share of basin area at least 3% drier than its own average, per year")
            st.plotly_chart(charts.dry_share_bars(data.time_stats(), S.d_year), width="stretch", config=NOCFG)
            st.caption("Yearly anomalies only – trends are not re-run per year.")

        elif tool == "Compare A/B":
            A, B = cell(S.d_a), cell(S.d_b)
            st.markdown(f"<div style='font-size:12px;font-weight:700;letter-spacing:.05em;color:{T.C_TEAL}'>"
                        f"COMPARE · {(A is not None) + (B is not None)} / 2 SELECTED</div>", unsafe_allow_html=True)
            rows = game.compare_rows(A, B)
            st.markdown(
                "<table style='width:100%;font-size:12px;border-collapse:collapse'>"
                "<tr><th style='text-align:left;padding:4px'>Indicator</th>"
                "<th style='text-align:left;padding:4px;background:#CCFBF1;color:#115E59'>Cell A</th>"
                "<th style='text-align:left;padding:4px;background:#EDE9FE;color:#5B21B6'>Cell B</th></tr>"
                + "".join(f"<tr style='border-top:1px solid #EEF2F0'><td style='padding:4px;color:#4B5563'>{k}</td>"
                          f"<td style='padding:4px'><b>{a}</b></td><td style='padding:4px'><b>{b}</b></td></tr>"
                          for k, a, b in rows) + "</table>", unsafe_allow_html=True)
            st.info(game.compare_text(A, B))

        elif is_swipe:
            st.markdown("**BEFORE & AFTER CHALLENGE**")
            actual = summ["drop2022_24_main_decreasing_kept_share_raw"] * 100
            if not S.d_sw_rev:
                st.markdown("#### What happens when we remove 2022–2024?")
                st.write("The map shows 2001–2024. Predict how many of its decreasing cells keep their evidence if the "
                         "record stops in 2021.")
                st.slider("Your prediction (%)", 0, 100, key="d_sw_pred")
                if st.button("Reveal actual result", type="primary"):
                    S.d_sw_rev = True
                    S.d_journal = game.add_journal(
                        S.d_journal, "swipe", "Before & after", "Swipe challenge · Completed",
                        f"Predicted {S.d_sw_pred}%, actual {actual:.1f}%: most decreasing cells lose their evidence "
                        "when 2022–2024 are removed.")
                    st.rerun()
            else:
                off = abs(S.d_sw_pred - actual)
                verdict = "spot on!" if off <= 5 else "close." if off <= 15 else \
                    "surprising, right? Move the divider to see where it changes."
                st.markdown(f"You predicted **{S.d_sw_pred}%**. Actual: **{actual:.1f}%** – {verdict}")
                inc21 = data.class_count("adjusted_drop_cells.nc", "class_raw_2001_2021", data.UP)
                x1, x2 = st.columns(2)
                x1.markdown(f"<div class='te-tile'><div class='l'>Ending 2024</div><div class='v'>"
                            f"{T.pct(summ['main_area_share_down'])}</div><div class='l'>area decreasing · "
                            f"{int(summ['main_cells_up'])} increases</div></div>", unsafe_allow_html=True)
                x2.markdown(f"<div class='te-tile' style='background:#FDF2F8'><div class='l'>Ending 2021</div><div class='v'>"
                            f"{T.pct(summ['drop2022_24_raw_decreasing_area_share'])}</div><div class='l'>area decreasing · "
                            f"{inc21} increases</div></div>", unsafe_allow_html=True)
                st.write(f"Move the divider across the south-central patch. Most of it turns grey when the record stops "
                         f"in 2021: only {actual:.1f}% of the decreasing cells keep their evidence. This is the clearest "
                         "picture of why the result is fragile.")
                if st.button("Predict again"):
                    S.d_sw_rev = False
                    st.rerun()

        elif tool == "Draw area":
            st.markdown("**YOUR AREA**")
            A = data.area_summary(*S.d_box) if S.d_box else None
            if A is None:
                st.write("Drag on the map to draw a box, or use the sliders under the map.")
            else:
                st.write(f"{A['n']:,} basin cells in your box")
                tot = A["n"]
                st.markdown(
                    "<div class='te-bar'>"
                    f"<div style='width:{100 * A['dec'] / tot:.2f}%;background:#3B82F6'></div>"
                    f"<div style='width:{100 * A['grey'] / tot:.2f}%;background:#D9DDE3'></div>"
                    f"<div style='width:{100 * A['inc'] / tot:.2f}%;background:#F97316'></div></div>",
                    unsafe_allow_html=True)
                x1, x2 = st.columns(2)
                x1.markdown(f"<div class='te-tile'><div class='l'>Decrease with evidence</div><b>{100 * A['dec'] / tot:.1f}%</b></div>",
                            unsafe_allow_html=True)
                x2.markdown(f"<div class='te-tile'><div class='l'>Not enough evidence</div><b>{100 * A['grey'] / tot:.1f}%</b></div>",
                            unsafe_allow_html=True)
                x1, x2 = st.columns(2)
                x1.markdown(f"<div class='te-tile'><div class='l'>Increase with evidence</div><b>{100 * A['inc'] / tot:.1f}%</b></div>",
                            unsafe_allow_html=True)
                x2.markdown(f"<div class='te-tile'><div class='l'>Median Sen's slope</div><b>{T.fmt_signed(A['median_pct'])}</b> %/dec</div>",
                            unsafe_allow_html=True)
                frag = (f"{A['fragile']:,} of the {A['dec']:,} decreasing cells lose their evidence without 2022–24. "
                        if A["dec"] else "")
                st.caption(frag + "Shares are of cells, not area-weighted. Each cell keeps its own test; nothing is "
                           "re-tested for your area.")
                if st.button("Save discovery to journal"):
                    S.d_journal = [{"id": f"area{len(S.d_journal)}", "title": f"My area ({tot:,} cells)",
                                    "tag": "Saved area",
                                    "text": f"{100 * A['dec'] / tot:.1f}% decreasing with evidence, "
                                            f"{100 * A['grey'] / tot:.1f}% not enough evidence, median "
                                            f"{T.fmt_signed(A['median_pct'])}% per decade."}] + S.d_journal
                    st.rerun()
                ba = data.csv("basin_annual.csv")
                st.markdown("**Your area vs whole basin** · yearly rainfall, % from own average")
                st.plotly_chart(charts.anomaly_lines(ba["year"], A["series"], ba["basin_mean_mm"]),
                                width="stretch", config=NOCFG)
                st.caption("Solid = your area (simple mean of its cells) · dashed = whole basin")

        else:  # Inspect: selected cell
            st.markdown("**SELECTED CELL**")
            c = cell(S.d_sel)
            if c is None:
                st.write("Hover the map for a quick look, click a cell to select it.")
            else:
                st.caption(f"{abs(c['lat']):.2f}°{'S' if c['lat'] < 0 else 'N'}, {abs(c['lon']):.2f}°W")
                head = {3: "Decrease with evidence", 2: "Increase with evidence"}.get(c["ec"], T.GREY_LONG)
                st.markdown(f"#### {head}")
                x1, x2 = st.columns(2)
                x1.markdown(f"<div class='te-tile'><div class='l'>Sen's slope</div><b>{T.fmt_signed(c['slope'])}</b> mm/yr"
                            f"<div class='l'>{T.fmt_signed(c['pct'])} %/decade</div></div>", unsafe_allow_html=True)
                x2.markdown(f"<div class='te-tile'><div class='l'>p → q (after FDR)</div><b>{game.fq(c['p_hr'])} → "
                            f"{game.fq(c['q'])}</b><div class='l'>evidence if q ≤ 0.05</div></div>", unsafe_allow_html=True)
                x1, x2 = st.columns(2)
                x1.markdown(f"<div class='te-tile'><div class='l'>Detectable size</div><b>{c['ratio']:.2f}×</b>"
                            f"<div class='l'>MDS {c['mds']:.1f} mm/yr</div></div>", unsafe_allow_html=True)
                x2.markdown(f"<div class='te-tile'><div class='l'>Drop-years tests</div><b>"
                            f"{'—' if c['ec'] == 1 else str(c['kept']) + ' / 4'}</b><div class='l'>"
                            f"{'only for cells with evidence' if c['ec'] == 1 else 'keep the same result'}</div></div>",
                            unsafe_allow_html=True)

        fb = S.d_fb
        if fb and not is_time:
            bg, fg = ("#F0FDFA", "#115E59") if fb["ok"] else ("#FDF2F8", "#9D174D")
            html = f"<div style='background:{bg};color:{fg};border-radius:10px;padding:10px 12px;font-size:13px;margin-top:8px'>"
            if fb.get("heat"):
                html += badge(fb["heat"], game.HEAT_COL[fb["heat"]]) + "<br>"
            html += f"<b>{fb['title']}</b> {fb['text']}"
            if fb.get("reward"):
                html += f"<div style='border-top:1px solid #99F6E4;margin-top:8px;padding-top:6px;font-size:12px'>🏅 Mission reward: <b>{fb['reward']}</b></div>"
            st.markdown(html + "</div>", unsafe_allow_html=True)
    st.caption("All values come from the result files. Discovery mode never re-tests anything; it helps you read "
               "the tests that were already run.")

# ------------------------------------------------------------------ play loop (one year per rerun)
if is_time and S.d_playing:
    time.sleep(0.6)
    if S.d_year >= 2024:
        S.d_playing = False
    else:
        S.d_next_year = S.d_year + 1
    st.rerun()
