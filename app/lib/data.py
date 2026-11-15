"""All file reads for the Trend Explorer app.

Rules (see handoff/ui-build-spec.md):
* The app only READS result files. It never recomputes trends, p, q or evidence.
  Counting cells by class or summing area weights is allowed (that is just
  describing the files), re-running a statistical test is not.
* Headline numbers come from summary_numbers.csv or file attributes.
* A missing optional file returns None, so the page can hide that card.
"""
from __future__ import annotations

import json
import warnings
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

# repo_root/app/lib/data.py -> repo_root/data/processed
DATA_DIR = Path(os.environ.get("TRENDLAB_DATA_DIR",
                               Path(__file__).resolve().parents[2] / "data" / "processed"))

YEARS = list(range(2001, 2025))
DROUGHT_YEARS = [2005, 2010, 2015, 2016]          # used in the drop-years check
LAST_YEARS = [2022, 2023, 2024]

# evidence_class codes in trend_cells.nc
OUTSIDE, NO_DATA, GREY, UP, DOWN = -1, 0, 1, 2, 3

# Drop-years variants: key -> (label on the switch, long label, file, variable)
VARIANTS = {
    "None": None,
    "Drought years": ("Drop drought years (2005, 2010, 2015, 2016)", "trend_cells.nc", "class_drop_drought", "drop_drought"),
    "2022–24": ("Drop 2022–2024", "trend_cells.nc", "class_drop_2022_24", "drop_2022_24"),
    "2001–03": ("Drop 2001–2003", "robustness_cells.nc", "class_drop_2001_03", "drop_2001_03"),
    "2011–13": ("Drop 2011–2013", "robustness_cells.nc", "class_drop_2011_13", "drop_2011_13"),
}

REGION_NAMES = {
    "toan_luu_vuc": ("Whole basin", "all cells"),
    "bac_lat>=-3": ("North", "lat ≥ 3°S"),
    "giua_-10<=lat<-3": ("Middle", "3–10°S"),
    "nam_lat<-10": ("South", "lat < 10°S"),
}


def path(name: str) -> Path:
    return DATA_DIR / name


def has(name: str) -> bool:
    return path(name).exists()


# ---------------------------------------------------------------- tables
@st.cache_data
def summary() -> dict:
    """summary_numbers.csv as {key: float}."""
    df = pd.read_csv(path("summary_numbers.csv"))
    return dict(zip(df["key"], df["value"].astype(float)))


@st.cache_data
def csv(name: str) -> pd.DataFrame | None:
    if not has(name):
        return None
    df = pd.read_csv(path(name))
    if "region" in df.columns:
        df["name"] = df["region"].map(lambda r: REGION_NAMES.get(r, (r, ""))[0])
        df["sub"] = df["region"].map(lambda r: REGION_NAMES.get(r, (r, ""))[1])
    return df


# ---------------------------------------------------------------- grids
@st.cache_data
def grid(name: str) -> dict | None:
    """Read one NetCDF grid into plain numpy arrays (picklable, cache-friendly).

    Returns {"lat", "lon", "vars": {name: 2-D array}, "attrs", "var_attrs"}.
    """
    if not has(name):
        return None
    with xr.open_dataset(path(name)) as ds:
        return {
            "lat": ds["lat"].values.astype(float),
            "lon": ds["lon"].values.astype(float),
            "vars": {v: ds[v].values for v in ds.data_vars},
            "attrs": dict(ds.attrs),
            "var_attrs": {v: dict(ds[v].attrs) for v in ds.data_vars},
        }


@st.cache_data
def annual() -> tuple[np.ndarray, np.ndarray] | None:
    """(years, array[year, lat, lon]) in mm/year."""
    if not has("annual.nc"):
        return None
    with xr.open_dataset(path("annual.nc")) as ds:
        return ds["year"].values.astype(int), ds["annual"].values.astype("float32")


def var(file: str, name: str) -> np.ndarray | None:
    g = grid(file)
    if g is None or name not in g["vars"]:
        return None
    return g["vars"][name]


def evidence() -> np.ndarray:
    return var("trend_cells.nc", "evidence_class")


def area_weight() -> np.ndarray:
    w = var("trend_cells.nc", "area_weight").astype(float)
    return np.where(evidence() >= GREY, np.nan_to_num(w), 0.0)


def magnitude_cutoffs() -> tuple[float, float]:
    a = grid("magnitude.nc")["attrs"]
    return (float(a["threshold_mild_moderate_pct_per_decade"]),
            float(a["threshold_moderate_strong_pct_per_decade"]))


def magnitude() -> np.ndarray:
    """magnitude_class, forced to 0 wherever evidence_class != 3 (safety net)."""
    m = var("magnitude.nc", "magnitude_class")
    return np.where(evidence() == DOWN, m, 0)


def json_attr(file: str, key: str) -> dict:
    g = grid(file)
    try:
        return json.loads(g["attrs"].get(key, "{}"))
    except (TypeError, ValueError):
        return {}


def dry_threshold_mm() -> str:
    g = grid("dry_months_cells.nc")
    m = re.search(r"<\s*([\d.]+)\s*mm", (g or {}).get("attrs", {}).get("note", ""))
    return f"{float(m.group(1)):.0f}" if m else "100"


# ---------------------------------------------------------------- drop years
def variant_class(key: str) -> np.ndarray | None:
    spec = VARIANTS.get(key)
    if spec is None:
        return None
    return var(spec[1], spec[2])


@st.cache_data
def variant_codes(key: str) -> np.ndarray:
    """Map category per cell for one drop-years variant.

    0 grey · 1 keeps evidence · 2 loses it · 3 evidence only without these years
    · 4 increase (in the variant) · NaN outside the basin.
    """
    main, alt = evidence(), variant_class(key)
    code = np.full(main.shape, np.nan)
    inside = main >= GREY
    code[inside] = 0
    main_ev = np.isin(main, [UP, DOWN])
    code[main_ev & (alt == main)] = 1
    code[main_ev & (alt != main)] = 2
    code[(main == GREY) & (alt == DOWN)] = 3
    code[inside & (alt == UP)] = 4
    return code


@st.cache_data
def variant_counts(key: str) -> dict:
    """Cell counts for the overview strip. kept share is read from file attrs."""
    main, alt = evidence(), variant_class(key)
    main_ev = np.isin(main, [UP, DOWN])
    spec = VARIANTS[key]
    attr_file, attr_key = ("trend_cells.nc", "sensitivity") if spec[1] == "trend_cells.nc" \
        else ("robustness_cells.nc", "cell_sensitivity")
    info = json_attr(attr_file, attr_key).get(spec[3], {})
    return {
        "long": spec[0],
        "n_ev": int(main_ev.sum()),
        "kept": int((main_ev & (alt == main)).sum()),
        "lost": int((main_ev & (alt != main)).sum()),
        "new": int(((main == GREY) & (alt == DOWN)).sum()),
        "inc": int((alt == UP).sum()),
        "kept_share": info.get("kept_significant_share"),
        "dropped_years": info.get("dropped_years", []),
    }


@st.cache_data
def main_codes() -> np.ndarray:
    """Main layer: 0 grey · 1 increase · 2/3/4 lowest/middle/highest third · 5 no data."""
    ec, mag = evidence(), magnitude()
    code = np.full(ec.shape, np.nan)
    code[ec == GREY] = 0
    code[ec == UP] = 1
    code[ec == DOWN] = 3                       # fallback if a third is missing
    for k in (1, 2, 3):
        code[(ec == DOWN) & (mag == k)] = 1 + k
    code[ec == NO_DATA] = 5
    return code


# ---------------------------------------------------------------- cells
def nearest_index(lat: float, lon: float) -> tuple[int, int] | None:
    g = grid("trend_cells.nc")
    la, lo = g["lat"], g["lon"]
    if not (la.min() - 0.05 <= lat <= la.max() + 0.05 and lo.min() - 0.05 <= lon <= lo.max() + 0.05):
        return None
    return int(np.abs(la - lat).argmin()), int(np.abs(lo - lon).argmin())


def _val(file: str, name: str, i: int, j: int):
    a = var(file, name)
    if a is None:
        return None
    v = a[i, j]
    return None if (isinstance(v, (float, np.floating)) and np.isnan(v)) else v.item()


def cell_record(lat: float, lon: float) -> dict | None:
    """Everything the details panel shows for one cell, read from the files.

    Returns None outside the grid; {"outside": True} outside the basin.
    """
    ij = nearest_index(lat, lon)
    if ij is None:
        return None
    i, j = ij
    g = grid("trend_cells.nc")
    rec = {"i": i, "j": j, "lat": float(g["lat"][i]), "lon": float(g["lon"][j])}
    ec = int(evidence()[i, j])
    if ec == OUTSIDE:
        rec["outside"] = True
        return rec
    T = "trend_cells.nc"
    rec.update(
        outside=False, ec=ec,
        slope=_val(T, "slope_mm_per_yr", i, j), pct=_val(T, "slope_pct_per_decade", i, j),
        mean=_val(T, "mean_mm_per_yr", i, j), p_hr=_val(T, "p_hr", i, j),
        q=_val(T, "q_by_hr", i, j), n_valid=_val(T, "n_valid_years", i, j),
        mag=int(magnitude()[i, j]),
        mds=_val("robustness_cells.nc", "mds_mm_per_yr", i, j),
        ratio=_val("robustness_cells.nc", "slope_over_mds", i, j),
        drops={k: (None if (a := variant_class(k)) is None else int(a[i, j]))
               for k in VARIANTS if VARIANTS[k] is not None},
    )
    rec["kept"] = sum(1 for d in rec["drops"].values() if d == ec)
    if has("drivers_cells.nc"):
        D = "drivers_cells.nc"
        rec["drivers"] = {
            "r2cv": _val(D, "r2_cv_detrended", i, j),
            "b_nino": _val(D, "beta_nino34_mm_per_sd", i, j),
            "b_tna": _val(D, "beta_tna_mm_per_sd", i, j),
            "slope_adj": _val(D, "slope_adj_mm_per_yr", i, j),
            "class_adj": _val(D, "class_adj_joint", i, j),
            "slope_adj_naive": _val(D, "slope_adj_naive_mm_per_yr", i, j),
            "class_adj_naive": _val(D, "class_adj_naive", i, j),
        }
    a = annual()
    rec["series"] = None if a is None else a[1][:, i, j].astype(float)
    return rec


@st.cache_data
def example_cells() -> dict:
    """Cells for the four 'Try' buttons, chosen from the files by a fixed rule."""
    g = grid("trend_cells.nc")
    ec = evidence()
    pct = var("trend_cells.nc", "slope_pct_per_decade").astype(float)
    slope = var("trend_cells.nc", "slope_mm_per_yr").astype(float)
    q = var("trend_cells.nc", "q_by_hr").astype(float)
    ratio = var("robustness_cells.nc", "slope_over_mds").astype(float)

    def at(mask, score, pick=np.nanargmin):
        s = np.where(mask, score, np.nan)
        k = pick(s)
        i, j = np.unravel_index(k, s.shape)
        return float(g["lat"][i]), float(g["lon"][j])

    return {
        "Steepest decrease": at(ec == DOWN, pct),
        "Grey, almost detectable": at((ec == GREY) & (slope < 0), q),
        "Grey, far from detectable": at(ec == GREY, ratio),
        "One of 5 increases": at(ec == UP, slope, np.nanargmax),
    }


@st.cache_data
def basin_points() -> dict:
    """Flattened basin cells for the hover/click layer (values rounded to keep the page light)."""
    g = grid("trend_cells.nc")
    ec = evidence()
    ii, jj = np.nonzero(ec >= GREY)
    T = "trend_cells.nc"
    pick = lambda n, d: np.round(var(T, n)[ii, jj].astype(float), d)
    return {
        "lat": np.round(g["lat"][ii], 2), "lon": np.round(g["lon"][jj], 2), "ii": ii, "jj": jj,
        "ec": ec[ii, jj].astype(int),
        "customdata": np.column_stack([pick("slope_mm_per_yr", 1), pick("slope_pct_per_decade", 1),
                                       pick("p_hr", 3), pick("q_by_hr", 3)]),
    }


@st.cache_data
def class_area_share(file: str, name: str, code: int) -> float:
    a = var(file, name)
    w = area_weight()
    return float(w[a == code].sum() / w.sum())


@st.cache_data
def class_count(file: str, name: str, code: int) -> int:
    return int((var(file, name) == code).sum())


@st.cache_data
def mds_median() -> float:
    m = var("robustness_cells.nc", "mds_mm_per_yr").astype(float)
    return float(np.nanmedian(m[evidence() >= GREY]))


def morans_i() -> float | None:
    v = grid("trend_cells.nc")["attrs"].get("morans_i_slope")
    return None if v is None else float(v)


def min_valid_years() -> int:
    return int(grid("trend_cells.nc")["attrs"].get("min_valid_years", 20))


def n_cells() -> int:
    return int((evidence() >= GREY).sum())


# ================================================================ Discovery helpers
# Everything below only counts, bins or averages values that are already in the files.
DROP_KEYS = ["Drought years", "2022–24", "2001–03", "2011–13"]


@st.cache_data
def kept_count() -> np.ndarray:
    """Number of the 4 drop-year tests in which a cell keeps its full-period class."""
    ec = evidence()
    return sum((variant_class(k) == ec).astype(int) for k in DROP_KEYS)


def _bin(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.digitize(values, edges).astype(float)


@st.cache_data
def layer_codes(name: str) -> np.ndarray:
    """Integer code per cell for one Discovery layer; NaN outside the basin.

    Code order matches game.LAYERS[name] (colours and labels)."""
    ec = evidence()
    inside = ec >= GREY
    out = np.full(ec.shape, np.nan)
    if name == "evidence":
        return main_codes()
    if name == "drop":
        return variant_codes("2022–24")
    if name == "slope":            # %/decade: ≤−20 | −20..−10 | −10..−3 | ±3 | 3..10 | 10..20 | ≥20
        v = var("trend_cells.nc", "slope_pct_per_decade").astype(float)
        out[inside] = _bin(v, [-20, -10, -3, 3, 10, 20])[inside]
    elif name == "q":              # ≤0.05 | 0.05–0.15 | 0.15–0.5 | >0.5
        v = var("trend_cells.nc", "q_by_hr").astype(float)
        out[inside] = (3 - _bin(-v, [-0.5, -0.15, -0.05]))[inside]
    elif name == "detect":         # ≥1.5 | 1–1.5 | 0.5–1 | <0.5
        v = var("robustness_cells.nc", "slope_over_mds").astype(float)
        out[inside] = (3 - _bin(v, [0.5, 1.0, 1.5]))[inside]
    elif name == "robust":         # kept 4 | 3 | 2 | 0–1 | no evidence
        k = kept_count()
        code = np.select([k == 4, k == 3, k == 2], [0, 1, 2], 3).astype(float)
        code[ec == GREY] = 4
        out[inside] = code[inside]
    return out


@st.cache_data
def swipe_codes(period: str) -> np.ndarray:
    """0 grey · 1 decrease · 2 increase, for the full record or the record ending 2021."""
    ec = evidence()
    cls = ec if period == "2024" else var("adjusted_drop_cells.nc", "class_raw_2001_2021")
    out = np.full(ec.shape, np.nan)
    inside = ec >= GREY
    out[inside] = 0
    out[inside & (cls == DOWN)] = 1
    out[inside & (cls == UP)] = 2
    return out


@st.cache_data
def whatif_codes(key: str) -> np.ndarray:
    """What-if map: key in raw_full, raw_2021, adj_full, adj_2021 (all precomputed in files)."""
    src = {"raw_full": ("trend_cells.nc", "evidence_class"),
           "raw_2021": ("adjusted_drop_cells.nc", "class_raw_2001_2021"),
           "adj_full": ("drivers_cells.nc", "class_adj_joint"),
           "adj_2021": ("adjusted_drop_cells.nc", "class_adj_2001_2021")}[key]
    cls = var(*src)
    ec = evidence()
    out = np.full(ec.shape, np.nan)
    inside = ec >= GREY
    out[inside] = 0
    out[inside & (cls == DOWN)] = 1
    out[inside & (cls == UP)] = 2
    return out


@st.cache_data
def anomaly_codes(year: int) -> np.ndarray:
    """Year's rainfall vs the cell's own 2001–2024 mean, in 7 bins (≥20% drier … ≥20% wetter)."""
    years, a = annual()
    k = int(np.where(years == year)[0][0])
    with warnings.catch_warnings():          # cells outside the basin are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(a, axis=0)
    pct = (a[k] - mean) / mean * 100
    out = np.full(pct.shape, np.nan)
    inside = evidence() >= GREY
    out[inside] = _bin(pct, [-20, -8, -3, 3, 8, 20])[inside]
    return out


@st.cache_data
def time_stats(split_lon: float = -62.0) -> pd.DataFrame:
    """Per year: basin mean, share of basin area ≥3% drier than its own mean, east/west anomaly (%)."""
    years, a = annual()
    w = area_weight()
    inside = evidence() >= GREY
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(a, axis=0)
    lon = grid("trend_cells.nc")["lon"][None, :] * np.ones(w.shape)
    rows = []
    for k, yr in enumerate(years):
        pct = (a[k] - mean) / mean * 100

        def region(mask):
            ww = w * mask
            return 100 * (np.nansum(a[k] * ww) / np.nansum(mean * ww) - 1)

        rows.append({"year": int(yr),
                     "basin_mm": float(np.nansum(a[k] * w) / w.sum()),
                     "dry_share": float((w * ((pct <= -3) & inside)).sum() / w.sum()),
                     "east": region(lon >= split_lon), "west": region(lon < split_lon)})
    return pd.DataFrame(rows)


@st.cache_data
def pools() -> dict:
    """Cell masks for the Detective missions and their sizes (quoted in the texts)."""
    ec, mag, k = evidence(), magnitude(), kept_count()
    q = var("trend_cells.nc", "q_by_hr").astype(float)
    s = var("trend_cells.nc", "slope_mm_per_yr").astype(float)
    m = {
        "strong": (ec == DOWN) & (mag == 3),
        "fragile": (ec == DOWN) & (variant_class("2022–24") != DOWN),
        "robust": (ec == DOWN) & (k == 4),
        "inc": ec == UP,
        "grey_near": (ec == GREY) & (s < 0) & (q > 0.05) & (q <= 0.15),
    }
    return {name: {"mask": mask, "n": int(mask.sum()), "ij": np.argwhere(mask)} for name, mask in m.items()}


def ring(name: str) -> tuple[float, float, float, float]:
    """(lat, lon, half-height, half-width) in degrees around a mission's target cells."""
    g = grid("trend_cells.nc")
    ij = pools()[name]["ij"]
    la, lo = g["lat"][ij[:, 0]], g["lon"][ij[:, 1]]
    return (float(np.median(la)), float(np.median(lo)),
            max(1.0, 1.5 * float(np.std(la))), max(1.0, 1.5 * float(np.std(lo))))


def area_summary(lat0, lat1, lon0, lon1) -> dict | None:
    """Counts per class inside a lat/lon box. Shares are of cells, nothing is re-tested."""
    g = grid("trend_cells.nc")
    rows = (g["lat"] >= min(lat0, lat1)) & (g["lat"] <= max(lat0, lat1))
    cols = (g["lon"] >= min(lon0, lon1)) & (g["lon"] <= max(lon0, lon1))
    box = rows[:, None] & cols[None, :]
    ec = evidence()
    cells = box & (ec >= GREY)
    n = int(cells.sum())
    if n == 0:
        return None
    pct = var("trend_cells.nc", "slope_pct_per_decade").astype(float)[cells]
    frag = cells & (ec == DOWN) & (variant_class("2022–24") != DOWN)
    years, a = annual()
    series = np.nanmean(a[:, cells], axis=1)
    return {"n": n, "dec": int((cells & (ec == DOWN)).sum()), "inc": int((cells & (ec == UP)).sum()),
            "grey": int((cells & (ec == GREY)).sum()), "median_pct": float(np.nanmedian(pct)),
            "fragile": int(frag.sum()), "series": series}
