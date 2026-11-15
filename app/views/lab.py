"""Discovery Lab: Predict, What-if, Match, Rank, Quiz. Every answer comes from the result files."""
import streamlit as st

from lib import charts, data
from lib import text as T

S = st.session_state
for k, v in dict(l_shown=[False, False, False], l_toggled=False, l_rank_done=False).items():
    S.setdefault(k, v)

summ = data.summary()
g = data.grid("trend_cells.nc")
reg = data.csv("regions_robustness.csv")
adj = data.csv("adjusted_robustness.csv")
rd = data.csv("regions_drivers.csv")
ew = data.csv("expanding_window.csv")
sens = data.json_attr("trend_cells.nc", "sensitivity")
basin = reg[reg["region"] == "toan_luu_vuc"].iloc[0]
basin21 = adj[adj["region"] == "toan_luu_vuc"].iloc[0] if adj is not None else None
none_2021 = bool(adj is not None and (adj[["p_raw", "p_adj"]] >= 0.05).all().all())
ewb = ew[ew["region"] == "toan_luu_vuc"].set_index("end_year")
NOCFG = {"displayModeBar": False}


def chip(n, txt, color):
    return (f"<span style='font-size:12px;font-weight:700;color:#fff;background:{color};border-radius:999px;"
            f"padding:3px 10px'>{n}</span> <b style='font-size:18px'>{txt}</b>")


done = []

st.markdown("<div style='font-size:12px;font-weight:700;letter-spacing:.06em;color:#0F766E'>DISCOVERY LAB</div>",
            unsafe_allow_html=True)
h1, h2 = st.columns([4, 1])
h1.markdown("## Can you read the rainfall evidence?")
h1.write("Five short challenges. Every answer comes from the project's result files.")
counter = h2.empty()

# ------------------------------------------------------------------ 01 Predict
with st.container(border=True):
    st.markdown(chip("01 · Slider", "Predict before you reveal", T.C_TEAL), unsafe_allow_html=True)
    preds = [
        dict(q="How much of the Amazon basin area shows a decrease with statistical evidence?",
             lo=0.0, hi=100.0, step=1.0, start=40.0, ans=round(100 * summ["main_area_share_down"], 1), unit="%",
             why=f"Most of the basin ({T.pct(summ['main_area_share_not_enough'])}) is grey: not enough evidence, "
                 "which does not mean no change."),
        dict(q="Drop the last three years (2022–2024). What share of the cells with evidence keep it?",
             lo=0.0, hi=100.0, step=1.0, start=50.0,
             ans=round(100 * sens.get("drop_2022_24", {}).get("kept_significant_share", float("nan")), 1), unit="%",
             why="The decrease signal leans heavily on the three driest years at the end of the record."),
        dict(q="If the record had ended in 2021, what would the whole-basin trend be (mm per year)?",
             lo=-15.0, hi=10.0, step=0.5, start=0.0, ans=round(float(ewb.loc[2021, "sen_mm_yr"]), 1), unit=" mm/yr",
             why=f"Ending in 2021 the basin slope is {T.fmt_signed(ewb.loc[2021, 'sen_mm_yr'])} mm/yr "
                 f"(p = {ewb.loc[2021, 'p_hr']:.2f}). It only turns negative once 2022–2024 are included."),
    ]
    cols = st.columns(3)
    for i, (col, p) in enumerate(zip(cols, preds)):
        with col, st.container(border=True):
            st.markdown(f"**{p['q']}**")
            guess = st.slider("Your guess", p["lo"], p["hi"], p["start"], p["step"], key=f"l_guess_{i}",
                              format="%.0f%%" if p["unit"] == "%" else "%.1f mm/yr")
            if st.button("Reveal", key=f"l_rev_{i}", type="primary"):
                sh = list(S.l_shown)
                sh[i] = True
                S.l_shown = sh
            if S.l_shown[i]:
                span = p["hi"] - p["lo"]
                gx, ax = (guess - p["lo"]) / span * 100, (p["ans"] - p["lo"]) / span * 100
                off = abs(guess - p["ans"])
                verdict = "Spot on!" if off <= span * 0.05 else "Close." if off <= span * 0.15 else "Surprising, right?"
                ans_txt = f"{p['ans']}{p['unit']}" if p["unit"] == "%" else f"{T.fmt_signed(p['ans'])}{p['unit']}"
                st.markdown(
                    "<div style='position:relative;height:12px;background:#F2F5F4;border-radius:999px;margin:8px 0'>"
                    f"<div style='position:absolute;top:-3px;bottom:-3px;left:{gx:.1f}%;border-left:3px solid #9CA3AF'></div>"
                    f"<div style='position:absolute;top:-5px;bottom:-5px;left:{ax:.1f}%;border-left:4px solid #0F766E'></div></div>",
                    unsafe_allow_html=True)
                st.markdown(f"**:green[Answer: {ans_txt}]** · {verdict} {p['why']}")
    done.append(all(S.l_shown))

# ------------------------------------------------------------------ 02 What if
with st.container(border=True):
    st.markdown(chip("02 · Toggles", "What if…?", "#7C3AED"), unsafe_allow_html=True)
    left, right = st.columns([1, 2], gap="large")
    with left:
        incl = st.toggle("Include 2022–2024 (the three driest years at the end)", value=True, key="l_incl")
        adjust = st.toggle("Account for ENSO & TNA (joint) · exploratory", value=False, key="l_adj")
        if not incl or adjust:
            S.l_toggled = True
        key = ("adj" if adjust else "raw") + "_" + ("full" if incl else "2021")
        src = {"raw_full": ("trend_cells.nc", "evidence_class"),
               "raw_2021": ("adjusted_drop_cells.nc", "class_raw_2001_2021"),
               "adj_full": ("drivers_cells.nc", "class_adj_joint"),
               "adj_2021": ("adjusted_drop_cells.nc", "class_adj_2001_2021")}[key]
        area = {"raw_full": summ["main_area_share_down"], "raw_2021": summ["drop2022_24_raw_decreasing_area_share"],
                "adj_full": summ["adj_joint_decreasing_area_share"],
                "adj_2021": summ["drop2022_24_adj_decreasing_area_share"]}[key]
        kept = {"raw_full": 1.0, "raw_2021": summ["drop2022_24_main_decreasing_kept_share_raw"],
                "adj_full": summ["adj_joint_main_decreasing_kept_share"],
                "adj_2021": summ["drop2022_24_main_decreasing_kept_share_adj"]}[key]
        n_dec = data.class_count(*src, data.DOWN)
        n_inc = data.class_count(*src, data.UP)
        if key == "raw_full":
            btxt = (f"Whole basin: Sen {T.fmt_signed(basin['sen_mm_yr'])} mm/yr, p = {basin['p_hr']:.2f} – "
                    + ("a basin-wide trend is supported." if basin["boot_excludes_zero"] else "no basin-wide trend supported."))
        elif key == "raw_2021":
            btxt = (f"Whole basin, 2001–2021: Sen {T.fmt_signed(basin21['sen_raw_2001_2021'])} mm/yr, "
                    f"p = {basin21['p_raw']:.2f}. Without the last three years almost nothing is left.")
        elif key == "adj_full":
            b_adj = float(rd[rd["region"] == "toan_luu_vuc"]["sen_adj"].iloc[0])
            btxt = (f"Whole basin after adjustment: Sen {T.fmt_signed(b_adj)} mm/yr. Its small p-value exists only with "
                    f"2022–2024 in the record (p = {basin21['p_adj']:.2f} without them). Extra cells reflect reduced "
                    "noise, not a new trend.")
        else:
            btxt = (f"Whole basin, adjusted, 2001–2021: Sen {T.fmt_signed(basin21['sen_adj_2001_2021'])} mm/yr, "
                    f"p = {basin21['p_adj']:.2f}." + (" No region reaches p < 0.05." if none_2021 else ""))
        a, b = st.columns(2)
        a.markdown(f"<div class='te-tile'><div class='l'>Basin area decreasing</div><div class='v'>{T.pct(area)}</div></div>",
                   unsafe_allow_html=True)
        b.markdown(f"<div class='te-tile'><div class='l'>Original decreases kept</div><div class='v'>{T.pct(kept)}</div></div>",
                   unsafe_allow_html=True)
        a, b = st.columns(2)
        a.markdown(f"<div class='te-tile'><div class='l'>Decreasing cells</div><div class='v'>{n_dec:,}</div></div>",
                   unsafe_allow_html=True)
        b.markdown(f"<div class='te-tile'><div class='l'>Increasing cells</div><div class='v'>{n_inc:,}</div></div>",
                   unsafe_allow_html=True)
        st.write(btxt)
        st.caption("All four combinations were computed by the analysis scripts; switching only picks a precomputed "
                   "result. ENSO/TNA layers are exploratory.")
    with right:
        st.plotly_chart(charts.raster_map(data.whatif_codes(key), [T.C_GREY, "#2563EB", T.C_INC], g["lat"], g["lon"],
                                          bands=False, height=460), width="stretch", config=NOCFG)
        st.markdown("".join(f"<span class='te-key'><span class='te-sw' style='background:{c};border:1px solid #D1D5DB'></span>{t}</span>"
                            for c, t in [("#2563EB", "Decrease with evidence"), (T.C_INC, "Increase with evidence"),
                                         (T.C_GREY, "Not enough evidence")]), unsafe_allow_html=True)
    done.append(S.l_toggled)

c3, c4 = st.columns(2, gap="medium")
# ------------------------------------------------------------------ 03 Match
with c3, st.container(border=True):
    st.markdown(chip("03 · Match", "Match the evidence", "#B45309"), unsafe_allow_html=True)
    st.caption("Pick the right meaning for each term.")
    pairs = {"Sen’s slope": "How big the trend is (mm per year)",
             "p-value": "Evidence from one cell’s test on its own",
             "q-value": f"Evidence after correcting for testing {data.n_cells():,} cells",
             "Grey cell": "Not enough evidence to detect a trend – not “no change”"}
    meanings = [pairs[k] for k in ["q-value", "Sen’s slope", "Grey cell", "p-value"]]
    n_ok = 0
    for term, right_m in pairs.items():
        pick = st.selectbox(term, meanings, index=None, placeholder="Choose a meaning…", key=f"l_m_{term}")
        if pick is not None:
            if pick == right_m:
                n_ok += 1
                st.markdown(":green[✓ Correct]")
            else:
                st.markdown(":red[✗ Not that one – try another meaning]")
    if n_ok == 4:
        st.success("All matched! Size of a trend and strength of evidence are different questions.")
    done.append(n_ok == 4)

# ------------------------------------------------------------------ 04 Rank
with c4, st.container(border=True):
    st.markdown(chip("04 · Order", "Rank the regions", "#1E3A8A"), unsafe_allow_html=True)
    st.caption("Order from the steepest decrease (1) to the weakest (4).")
    names = reg["name"].tolist()
    correct = reg.sort_values("sen_mm_yr")["name"].tolist()
    sen = dict(zip(reg["name"], reg["sen_mm_yr"]))
    order = [st.selectbox(f"Position {i + 1}", names, index=None, placeholder="Choose a region…", key=f"l_r_{i}")
             for i in range(4)]
    if st.button("Submit order", type="primary"):
        S.l_rank_done = True
    rank_ok = order == correct
    if S.l_rank_done:
        if None in order or len(set(order)) < 4:
            st.warning("Use each region once, in all four positions.")
        elif rank_ok:
            ex = [r for r in reg.itertuples() if r.boot_excludes_zero]
            st.success("Correct: " + " > ".join(f"{n} ({T.fmt_signed(sen[n])})" for n in correct) + ". Only "
                       + " and ".join(r.name.lower() for r in ex) + " bands have bootstrap intervals that exclude zero"
                       + (" – and even they lose it when the record stops in 2021." if none_2021 else "."))
        else:
            st.error("Not quite: " + ", ".join(
                f"{i + 1}. {n} {'✓' if n == correct[i] else '✗'}" for i, n in enumerate(order)) + ". Adjust and submit again.")
    done.append(S.l_rank_done and rank_ok)

# ------------------------------------------------------------------ 05 Quiz
with st.container(border=True):
    st.markdown(chip("05 · Quiz", "Climate detective quiz", "#374151"), unsafe_allow_html=True)
    quiz = [
        dict(q="A cell has a negative Sen’s slope, but q = 0.12. What can we conclude?",
             o=["Rainfall is clearly decreasing there.",
                "A decrease is estimated, but there is not enough evidence to call it a trend.",
                "Rainfall did not change there.", "The data for that cell are invalid."], a=1,
             fb="The evidence rule is q ≤ 0.05. With q = 0.12 the cell is grey: the estimated decrease cannot be separated "
                "from year-to-year noise. That is “not enough evidence”, not “no change”."),
        dict(q="Why can two cells with almost the same slope get different evidence labels?",
             o=["One of them has a data error.",
                "Their year-to-year noise differs, so the same slope stands out more in one cell.",
                "The map colours are random.", "Only the larger cell is tested."], a=1,
             fb=f"Evidence depends on slope relative to the noise. Detectable size (MDS) varies by place – the median is "
                f"about {data.mds_median():.1f} mm/yr – so an equal slope can pass in a quiet cell and fail in a noisy one."),
        dict(q=f"After removing ENSO and TNA effects, more cells show a decrease "
               f"({T.pct(summ['adj_joint_decreasing_area_share'])} vs {T.pct(summ['main_area_share_down'])} of the area). Why?",
             o=["The ocean causes the drying.", "The adjustment removes noise, so a small trend is easier to detect.",
                "New data were added.", "Rainfall fell faster after adjustment."], a=1,
             fb=f"Newly significant cells reflect reduced noise, not a new trend. And without 2022–2024, only "
                f"{T.pct(summ['drop2022_24_main_decreasing_kept_share_adj'])} of the original decreases survive even "
                "after adjustment."),
    ]
    score, answered = 0, 0
    for i, q in enumerate(quiz):
        pick = st.radio(f"**Q{i + 1}.** {q['q']}", q["o"], index=None, key=f"l_q_{i}")
        if pick is not None:
            answered += 1
            ok = q["o"].index(pick) == q["a"]
            score += ok
            (st.success if ok else st.error)(("Correct. " if ok else "Not quite. ") + q["fb"])
    st.markdown(f"**Score: {score} / {len(quiz)}**")
    done.append(answered == len(quiz))

counter.markdown(f"<div class='te-tile' style='text-align:center'><div class='l'>Challenges done</div>"
                 f"<div class='v'>{sum(done)} / 5</div></div>", unsafe_allow_html=True)
