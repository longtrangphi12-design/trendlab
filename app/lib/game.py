"""Discovery game layer: layers, missions, hot/cold, journal, claims, quiz.

Pure functions (no Streamlit, no file reads). Numbers used in texts are passed
in from data.py so every quoted figure comes from the result files.
"""
from __future__ import annotations

import numpy as np

from . import text as T

# ---------------------------------------------------------------- layers
# name -> (button label, [(colour, legend label), ...]) in code order
LAYERS = {
    "evidence": ("Evidence", [(T.C_GREY, "Not enough evidence (≠ no change)"), (T.C_INC, "Increase"),
                              (T.C_DEC[0], "Lowest third of decreases"), (T.C_DEC[1], "Middle third"),
                              (T.C_DEC[2], "Highest third"), (T.C_NODATA, "Not enough valid years")]),
    "drop": ("Without 2022–24", [(T.C_GREY, "Not enough evidence"), (T.C_KEEP, "Keeps evidence without 2022–24"),
                                 (T.C_LOSE, "Loses it"), (T.C_GAIN, "Evidence only without 2022–24"),
                                 (T.C_INC, "Increase")]),
    "slope": ("Trend size", [("#1E3A8A", "≤ −20 %/dec"), ("#3B82F6", "−20 to −10"), ("#BFDBFE", "−10 to −3"),
                             ("#F3F4F6", "±3"), ("#FED7AA", "+3 to +10"), ("#FB923C", "+10 to +20"),
                             ("#C2410C", "≥ +20")]),
    "q": ("q-value", [("#134E4A", "q ≤ 0.05 (evidence)"), ("#0D9488", "0.05–0.15 (close)"),
                      ("#99F6E4", "0.15–0.5"), ("#F0FDFA", "> 0.5")]),
    "detect": ("Detectability", [("#4C1D95", "≥ 1.5× detectable size"), ("#8B5CF6", "1–1.5×"),
                                 ("#C4B5FD", "0.5–1×"), ("#F5F3FF", "< 0.5×")]),
    "robust": ("Robustness", [("#1E3A8A", "Evidence kept in 4/4 tests"), ("#2563EB", "3/4"), ("#F472B6", "2/4"),
                              ("#FBCFE8", "0–1/4"), ("#F3F4F6", "No evidence in full period")]),
}
ANOMALY = [("#1E3A8A", "≥ 20% drier"), ("#3B82F6", "8–20% drier"), ("#BFDBFE", "3–8% drier"), ("#F3F4F6", "±3%"),
           ("#FED7AA", "3–8% wetter"), ("#FB923C", "8–20% wetter"), ("#C2410C", "≥ 20% wetter")]
SWIPE = [(T.C_GREY, "Not enough evidence"), ("#2563EB", "Decrease with evidence"), (T.C_INC, "Increase with evidence")]

MODES = {"Detective": ("Solve map missions", T.C_TEAL), "Grey Zone": ("Explore uncertainty", "#475569"),
         "Time Machine": ("Travel through 24 years", "#7C3AED"), "Evidence": ("Test a claim", "#B45309")}
TOOLS = ["Inspect", "Compare A/B", "Draw area", "Swipe 2024 | 2021"]

CL = {0: "Not tested", 1: "Not enough evidence", 2: "Increase with evidence", 3: "Decrease with evidence"}


def fq(q):
    return "—" if q is None else ("< 0.001" if q < 0.001 else f"{q:.3f}")


def legend_index(layer: str, c: dict) -> int | None:
    """Which legend item the selected cell falls in (to highlight it)."""
    if c is None or c.get("outside"):
        return None
    ec = c["ec"]
    if layer == "evidence":
        if ec == 3:
            return 1 + c["mag"] if c["mag"] in (1, 2, 3) else 3
        return {1: 0, 2: 1}.get(ec, 5)
    if layer == "drop":
        d = c["drops"].get("2022–24")
        if d == 2:
            return 4
        if ec in (2, 3):
            return 1 if d == ec else 2
        return 3 if d == 3 else 0
    if layer == "slope":
        return int(np.digitize(c["pct"], [-20, -10, -3, 3, 10, 20]))
    if layer == "q":
        q = c["q"]
        return 0 if q <= 0.05 else 1 if q <= 0.15 else 2 if q <= 0.5 else 3
    if layer == "detect":
        r = c["ratio"]
        return 0 if r >= 1.5 else 1 if r >= 1 else 2 if r >= 0.5 else 3
    if layer == "robust":
        if ec == 1:
            return 4
        k = c["kept"]
        return 0 if k == 4 else 1 if k == 3 else 2 if k == 2 else 3
    return None


# ---------------------------------------------------------------- journal
def add_journal(journal: list, jid: str, title: str, tag: str, body: str) -> list:
    if any(j["id"] == jid for j in journal):
        return journal
    return [{"id": jid, "title": title, "tag": tag, "text": body}] + journal


# ---------------------------------------------------------------- Detective
REWARD = {"strong": "Strong Decline Spotter", "fragile": "Fragile Signal Hunter",
          "robust": "Rock-Solid Finder", "inc": "Rare Increase Tracker"}


def missions(n: dict) -> list[dict]:
    """n: numbers from files – t2, n_cells, n_down, fragile, robust, n_up."""
    return [
        {"id": "strong", "title": "Find a strong decline",
         "task": f"Click a cell whose decrease is in the highest third of all decreases with evidence "
                 f"(more than {n['t2']:.1f}% per decade).",
         "hint": "Try the “Trend size” layer and look for the darkest blue patches in the south-central basin "
                 "and along the Andes.",
         "journal": ("A strong local decline",
                     f"Some cells lost more than {n['t2']:.1f}% of their rainfall per decade, with evidence – mostly "
                     "in the south-central basin and the Andes.")},
        {"id": "fragile", "title": "Find a fragile signal",
         "task": "Click a cell that shows a decrease with evidence, but loses that evidence when 2022–2024 are left out.",
         "hint": "Switch to “Robustness”: pink cells keep their result in 2 or fewer of the 4 drop-year tests. "
                 "Most decreases are like this.",
         "journal": ("A fragile rainfall signal",
                     f"Most decreasing cells ({n['fragile']:,} of {n['n_down']:,}) lose their evidence when the final "
                     "three years are excluded.")},
        {"id": "robust", "title": "Find a rock-solid signal",
         "task": "Click a decreasing cell that keeps its evidence in all four drop-year tests "
                 "(drought years, 2022–24, 2001–03, 2011–13).",
         "hint": f"On “Robustness”, the darkest blue cells passed all four tests. Only {n['robust']:,} cells do – "
                 "look in the south-central core.",
         "journal": ("A rock-solid signal",
                     f"Only {n['robust']:,} decreasing cells keep their evidence in all four drop-year tests.")},
        {"id": "inc", "title": f"Find one of the {n['n_up']} increases",
         "task": f"Only {n['n_up']} cells in the whole basin show an increase with evidence. "
                 "Click within 3 cells (0.3°) of one of them.",
         "hint": "They sit in the north-west, close to the equator. Look for the tiny orange dots.",
         "journal": ("The rare increases",
                     f"Only {n['n_up']} of {n['n_cells']:,} cells show an increase with evidence – isolated local "
                     "results, not a pattern.")},
    ]


def inc_distance(c: dict, inc_ij: np.ndarray) -> int:
    """Distance in cells (max of row/col steps) to the nearest increase cell."""
    return int(np.max(np.abs(inc_ij - np.array([c["i"], c["j"]])), axis=1).min())


def check_mission(mid: str, c: dict, inc_ij: np.ndarray) -> bool:
    ec = c["ec"]
    if mid == "strong":
        return ec == 3 and c["mag"] == 3
    if mid == "fragile":
        return ec == 3 and c["drops"].get("2022–24") != 3
    if mid == "robust":
        return ec == 3 and c["kept"] == 4
    if mid == "inc":
        return inc_distance(c, inc_ij) <= 3
    return False


def success_text(mid: str, c: dict, n: dict) -> str:
    if mid == "strong":
        return (f"This cell lost {abs(c['pct']):.1f}% of its rainfall per decade, and the test still passes after "
                f"correcting for {n['n_cells']:,} tests (q = {fq(c['q'])}).")
    if mid == "fragile":
        return (f"Without the three very dry final years this cell no longer passes. {n['fragile']:,} of the "
                f"{n['n_down']:,} decreasing cells behave like this.")
    if mid == "robust":
        return (f"One of only {n['robust']:,} cells whose decrease survives every drop-year test. These are the "
                "strongest local signals in the record.")
    return (f"Found it. With {n['n_up']} cells out of {n['n_cells']:,}, treat increases as isolated local results, "
            "not a pattern.")


HEAT_COL = {"HOT": "#DC2626", "WARM": "#F97316", "COLD": "#2563EB", "FOUND!": T.C_TEAL}


def heat(mid: str, c: dict, inc_ij: np.ndarray) -> tuple[str, str] | None:
    """Hot/Warm/Cold label + message. Increases: distance; others: statistical closeness (never geography)."""
    ec = c["ec"]
    if mid == "inc":
        d = inc_distance(c, inc_ij)
        if d <= 3:
            return None
        if d <= 15:
            return "HOT", f"You are very close – within {d / 10:.1f}° of an increase cell."
        if d <= 45:
            return "WARM", "Right part of the basin. Keep moving."
        return "COLD", "Far from any of the increases."
    if mid == "strong":
        if ec == 3:
            return ("HOT", "A decrease with evidence, but only in the middle third. Look for darker blue.") \
                if c["mag"] == 2 else ("WARM", "A decrease with evidence, but in the lowest third.")
        return "COLD", "No decrease with evidence here."
    if mid == "fragile":
        return ("WARM", "Not quite! This cell still has evidence without 2022–2024.") if ec == 3 \
            else ("COLD", "This cell has no decrease with evidence to begin with.")
    if mid == "robust":
        if ec == 3:
            return ("HOT", "Survives 3 of 4 drop-year tests – one short.") if c["kept"] == 3 \
                else ("WARM", f"A decrease with evidence, but it survives only {c['kept']} of 4 tests.")
        return "COLD", "No decrease with evidence here."
    return None


def miss_text(c: dict) -> str:
    base = f"This cell: {CL[c['ec']].lower()}, {T.fmt_signed(c['pct'])}% per decade, q = {fq(c['q'])}"
    return base + ("." if c["ec"] == 1 else f", keeps its result in {c['kept']} of 4 drop-year tests.")


# ---------------------------------------------------------------- Compare
def compare_text(a: dict | None, b: dict | None) -> str:
    if not (a and b):
        return "Pick two cells to compare. Tip: a grey cell and a blue cell with a similar slope make the best pair."
    if a["ec"] != b["ec"]:
        same = abs(a["pct"] - b["pct"]) <= max(3, 0.25 * max(abs(a["pct"]), abs(b["pct"])))
        ev, en = (a, "A") if a["ec"] in (2, 3) else (b, "B")
        gr, gn = (b, "B") if en == "A" else (a, "A")
        return ((f"The trends are similar in size, but only {en} has evidence. " if same else f"Only {en} has evidence. ")
                + f"Cell {en}’s slope is {ev['ratio']:.2f}× what its record could detect, cell {gn}’s is {gr['ratio']:.2f}×. "
                "Year-to-year noise differs between places, so the same slope can be clear in one cell and hidden in "
                "another. Size of change and strength of evidence are two different things.")
    return (f"Both cells are “{CL[a['ec']].lower()}”. Compare their detectable size ({a['ratio']:.2f}× vs "
            f"{b['ratio']:.2f}×) to see which signal stands out more from its own noise.")


def compare_rows(a: dict | None, b: dict | None) -> list[tuple[str, str, str]]:
    def v(x, f):
        return "—" if not x else f(x)
    spec = [("Evidence", lambda x: CL[x["ec"]]), ("Sen's slope", lambda x: f"{T.fmt_signed(x['slope'])} mm/yr"),
            ("Per decade", lambda x: f"{T.fmt_signed(x['pct'])}%"), ("p (Hamed–Rao)", lambda x: fq(x["p_hr"])),
            ("q (after FDR)", lambda x: fq(x["q"])), ("Detectable size", lambda x: f"{x['ratio']:.2f}×"),
            ("Drop-years tests", lambda x: "—" if x["ec"] == 1 else f"{x['kept']} / 4")]
    return [(k, v(a, f), v(b, f)) for k, f in spec]


# ---------------------------------------------------------------- Evidence challenge
def claims(n: dict) -> list[dict]:
    return [
        {"short": "Claim 1", "claim": "The whole Amazon basin became drier between 2001 and 2024.",
         "steps": [("layer_evidence", "Look at the Evidence layer: how much of the basin is blue?"),
                   ("layer_drop", "Switch to “Without 2022–24”: does the blue survive?"),
                   ("area", "Draw a box over the north of the basin and read its summary")],
         "opts": ["Supported", "Not supported – the change is regional and fragile", "Cannot be judged at all"], "ans": 1,
         "title": "The basin-wide trend",
         "fb": (f"Only {T.pct(n['area_down'])} of the basin area shows a decrease with evidence, the basin-wide slope "
                f"({T.fmt_signed(n['basin_sen'])} mm/yr) has p = {n['basin_p']:.2f}, and without 2022–24 only "
                f"{T.pct(n['kept_2224'])} of those cells keep their evidence. “The whole basin dried” is not supported.")},
        {"short": "Claim 2", "claim": "The south-central basin shows signs of drying.",
         "steps": [("area", "Draw a box over the south-central blue patch"),
                   ("layer_robust", "Check the Robustness layer"),
                   ("compare", "Compare a robust cell with a fragile one")],
         "opts": ["Supported, and robust", "Supported, but fragile", "Not supported"], "ans": 1,
         "title": "Drying in the south-central basin",
         "fb": ("The decreases cluster there, and the middle and southern bands have bootstrap intervals that exclude "
                "zero – but the signal depends heavily on 2022–2024"
                + (" and no region reaches p < 0.05 when the record stops in 2021" if n["none_2021"] else "")
                + ". Supported, but fragile.")},
        {"short": "Claim 3", "claim": "Grey cells mean rainfall did not change there.",
         "steps": [("layer_q", "Open the q-value layer"), ("grey", "Select a grey cell"),
                   ("layer_detect", "Open Detectability: is the change big enough to detect?")],
         "opts": ["True", "False – grey means “not enough evidence”", "True only in the north"], "ans": 1,
         "title": "The grey zone means uncertainty",
         "fb": (f"Grey means the test could not separate a change from year-to-year noise. The median detectable slope "
                f"is about {n['mds_median']:.1f} mm/yr, so smaller real changes usually stay grey. "
                "Not enough evidence ≠ no change.")},
    ]


# ---------------------------------------------------------------- Time Machine questions
def time_questions(ts) -> list[dict]:
    """ts: data.time_stats() dataframe. Answers are read from it, not hard-coded."""
    opts1 = [2005, 2010, 2016, 2024]
    sub = ts[ts["year"].isin(opts1)].set_index("year")["dry_share"]
    best = int(sub.idxmax())
    top2 = ts.nlargest(2, "dry_share")
    q1_fb = (f"{best}: {T.pct(sub[best], 0)} of the basin area was at least 3% drier than its own average"
             + (f" ({int(top2.iloc[1]['year'])}: {T.pct(top2.iloc[1]['dry_share'], 0)})" if int(top2.iloc[0]['year']) == best else "")
             + ". The record-dry years sit at the very end of the series – the reason the trend is fragile.")
    r15 = ts.set_index("year").loc[2015]
    e, w = r15["east"], r15["west"]
    ans2 = "Everywhere" if (e <= -3 and w <= -3) else ("Mostly the east" if e < w else "Mostly the west")
    q2_fb = (f"East of 62°W was {T.fmt_signed(e)}% vs its usual rainfall, the west {T.fmt_signed(w)}%, "
             "consistent with Jiménez-Muñoz et al. (2016) for the 2015–16 El Niño.")
    return [
        {"id": "q1", "q": "Which year left the largest share of the basin drier than usual?",
         "opts": [str(y) for y in opts1], "ans": str(best), "fb": q1_fb},
        {"id": "q2", "q": "In 2015, where was it dry?", "opts": ["Everywhere", "Mostly the east", "Mostly the west"],
         "ans": ans2, "fb": q2_fb},
    ]
