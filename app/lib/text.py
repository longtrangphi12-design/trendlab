"""All fixed wording in one place, so the rules are easy to check.

Rules: grey is never "no change"; direction and evidence are separate;
"associated with", never "caused by"; never "Validated";
exploratory layers are marked "Exploratory".
"""

APP_TITLE = "Trend Explorer"
APP_SUBTITLE = "Historical rainfall trends · Amazon River basin · 2001–2024 · not a forecast"

GREY_LONG = "Not enough evidence to detect a trend (does not mean no change)"
GREY_SHORT = "Not enough evidence"

CLASS_LABEL = {
    0: "Not enough valid years",
    1: GREY_SHORT,
    2: "Increase with evidence",
    3: "Decrease with evidence",
}

IN_SHORT = (
    "**In short:** the south-central basin and the Andean strip show signs of drying, "
    "but 24 years cannot separate the extreme 2022–2024 dry years from a persistent shift. "
    "This explorer shows that uncertainty instead of hiding it."
)

CENTRAL_MESSAGE = (
    "The Amazon shows signs of drying in the south-central basin and the Andean strip, "
    "but 24 years cannot separate the extreme 2022–2024 dry years from a persistent shift. "
    "This explorer shows that uncertainty instead of hiding it."
)

EVIDENCE_RULE = (
    "Evidence = q ≤ 0.05 after the Benjamini–Yekutieli correction, with the Hamed–Rao "
    "autocorrelation fix. The map and the details panel use the same class."
)

MAG_NOTE = "Relative scale among decreasing cells, not an absolute severity."
WINDOWS_NOTE = "Overlapping windows; descriptive only; p not corrected."
ASSOC_NOTE = "Statistical association, not causal attribution."
EXPLORATORY = "EXPLORATORY"

DROUGHT_SOURCES = (
    "Drought years marked in the charts are documented Amazon droughts: 2005 (Brando et al. 2010), "
    "2010 (Lewis et al. 2011), 2015–16 (Jiménez-Muñoz et al. 2016); 2023–24 (Marengo et al. 2024). "
    "Full references in About & methods."
)

DATA_BOX = (
    "NASA GPM IMERG Final V07, monthly (GPM_3IMERGM), NASA GES DISC  \n"
    "Basin: HydroBASINS level 3  \n"
    "0.1° grid · {n:,} cells · 2001–2024  \n"
    "Cells need ≥ {m} valid years."
)

# Colours (same as the UI v3 canvas)
C_GREY, C_GREY_BD = "#E5E7EB", "#D1D5DB"
C_INC = "#F97316"
C_DEC = ["#BFDBFE", "#3B82F6", "#1E3A8A"]          # lowest / middle / highest third
C_KEEP, C_LOSE, C_GAIN = "#1E3A8A", "#F9A8D4", "#3B82F6"
C_NODATA = "#FAFAFA"
C_TEAL = "#0F766E"
C_PINK_BAND = "#F9A8D4"
C_DROUGHT_BAND = "#94A3B8"
C_NINO, C_TNA = "#16A34A", "#7C3AED"


def fmt_signed(x: float | None, d: int = 1) -> str:
    if x is None:
        return "—"
    s = f"{abs(x):,.{d}f}"
    if round(x, d) == 0:
        return s
    return ("−" if x < 0 else "+") + s


def fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"
