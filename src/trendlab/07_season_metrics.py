"""Việc 7: chỉ số mùa vụ theo từng ô (PCI, biên độ và pha điều hòa bậc 1) và xu hướng của chúng.

Chạy (từ thư mục gốc repo):  python src/trendlab/07_season_metrics.py
Đọc : data/processed/monthly.nc, data/interim/amazon_mask.nc
Ghi : data/processed/season_cells.nc
"""
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

from trendlab.multitest import benjamini_hochberg
from trendlab.trend import sen_slope
from trendlab.trend_extra import benjamini_yekutieli, hamed_rao

ROOT = Path(__file__).resolve().parents[2]
INT, PROC = ROOT / "data/interim", ROOT / "data/processed"

# ---------------- CẤU HÌNH (chốt TRƯỚC khi xem kết quả) ----------------
ALPHA = 0.05
MAIN_Q = "q_by_hr"      # cùng quy tắc với script 06
R2_MIN = 0.5            # chỉ xét pha ở ô mà sóng năm giải thích >= 50% phương sai của chu kỳ mùa
SCENARIOS = {"drop_drought": [2005, 2010, 2015, 2016], "drop_2022_24": [2022, 2023, 2024]}
CLASS_LABEL = {-1: "ngoài lưu vực", 0: "không xét", 1: "chưa đủ bằng chứng",
               2: "tăng có bằng chứng", 3: "giảm có bằng chứng"}


def trend_layer(X, years):
    """Sen + Hamed-Rao cho từng cột của X (năm x ô). Trả về slope, q_bh, q_by."""
    t = years.astype(float)
    n = X.shape[1]
    slope, p = np.full(n, np.nan), np.full(n, np.nan)
    for c in range(n):
        slope[c] = sen_slope(X[:, c], t)
        p[c] = hamed_rao(X[:, c])["p"]
    return slope, benjamini_hochberg(p), benjamini_yekutieli(p)


def classify(slope, q):
    cls = np.ones(len(slope), dtype=np.int8)
    sig = q <= ALPHA
    cls[sig & (slope > 0)] = 2
    cls[sig & (slope < 0)] = 3
    return cls


# ---------------- nạp và kiểm tra ----------------
monthly = xr.open_dataset(PROC / "monthly.nc")["monthly"]            # mm/tháng
grid = xr.open_dataset(INT / "amazon_mask.nc")
mask, w = grid["mask"].astype(bool), grid["weight"]
assert monthly.dims == ("time", "lat", "lon"), monthly.dims
NY = 24
assert monthly.sizes["time"] == 12 * NY, "kỳ vọng 288 tháng"
assert (monthly["time"].dt.month.values == np.tile(np.arange(1, 13), NY)).all(), "tháng sai thứ tự"
years = np.arange(2001, 2001 + NY)
assert int(monthly["time"].dt.year[0]) == 2001
nlat, nlon = monthly.sizes["lat"], monthly.sizes["lon"]

P = monthly.values.reshape(NY, 12, -1).astype(float)                 # (năm, tháng, ô)
in_basin = mask.values.ravel()
idx = np.flatnonzero(in_basin & np.isfinite(P).all(axis=(0, 1)))
print(f"Ô trong lưu vực: {int(in_basin.sum())} | đủ 288 tháng: {len(idx)}")
P = P[:, :, idx]
n = P.shape[2]

# ---------------- chỉ số theo từng năm và từng ô ----------------
PCI = 100 * (P ** 2).sum(1) / P.sum(1) ** 2                           # Oliver: 100*sum(P^2)/sum(P)^2

ang = 2 * np.pi * np.arange(1, 13) / 12                               # tháng m đặt tại góc 2*pi*m/12
a = (2 / 12) * (P * np.cos(ang)[None, :, None]).sum(1)
b = (2 / 12) * (P * np.sin(ang)[None, :, None]).sum(1)
AMP = np.hypot(a, b)                                                  # mm/tháng

a0, b0 = a.mean(0), b.mean(0)                                         # sóng của chu kỳ mùa trung bình
phi0 = np.arctan2(b0, a0)
d = (np.arctan2(b, a) - phi0 + np.pi) % (2 * np.pi) - np.pi           # lệch pha so với khí hậu, gói về (-pi, pi]
PHASE = d * 12 / (2 * np.pi)                                          # đơn vị: tháng (dương = đỉnh đến muộn hơn)

clim = P.mean(0)                                                      # (12, ô)
r2 = (np.hypot(a0, b0) ** 2 / 2) / clim.var(0)                        # phần phương sai chu kỳ mùa do sóng năm giải thích
seasonal = r2 >= R2_MIN
peak = (phi0 % (2 * np.pi)) / (2 * np.pi) * 12
peak = np.where(peak < 0.5, peak + 12, peak)                          # tháng đạt đỉnh, tháng m có tâm tại số m
print(f"Ô có mùa rõ (R2 >= {R2_MIN}): {int(seasonal.sum())}/{n}")

# ---------------- xu hướng + độ nhạy cho từng lớp ----------------
layers = {"pci": (PCI, np.ones(n, bool)), "amp": (AMP, np.ones(n, bool)), "phase": (PHASE, seasonal)}
dims = ("lat", "lon")


def to_grid(vals, cols, dtype="float32", fill=np.nan):
    g = np.full(nlat * nlon, fill, dtype=dtype)
    g[idx[cols]] = vals
    return g.reshape(nlat, nlon)


def base_class(cls, cols):
    c = np.full(nlat * nlon, -1, dtype=np.int8)
    c[in_basin] = 0
    c[idx[cols]] = cls
    return c


data = {
    "pci_mean": (dims, to_grid(PCI.mean(0), np.arange(n))),
    "amp_mean_mm": (dims, to_grid(AMP.mean(0), np.arange(n))),
    "rel_amp": (dims, to_grid(np.hypot(a0, b0) / clim.mean(0), np.arange(n))),
    "season_r2": (dims, to_grid(r2, np.arange(n))),
    "peak_month_clim": (dims, to_grid(peak, np.arange(n))),
}
wv = w.values.ravel()
units = {"pci": ("PCI/thập kỷ", 10), "amp": ("mm/tháng/thập kỷ", 10), "phase": ("ngày/thập kỷ", 30.44 * 10)}
summary = {}
for name, (X, sel) in layers.items():
    t0 = time.time()
    cols = np.flatnonzero(sel)
    Xs = X[:, cols]
    slope, qbh, qby = trend_layer(Xs, years)
    q_main = qby if MAIN_Q == "q_by_hr" else qbh
    cls = classify(slope, q_main)
    cg = base_class(cls, cols)
    data[f"{name}_slope"] = (dims, to_grid(slope, cols))
    data[f"{name}_q_by_hr"] = (dims, to_grid(qby, cols))
    data[f"{name}_q_bh_hr"] = (dims, to_grid(qbh, cols))
    data[f"{name}_class"] = (dims, cg.reshape(nlat, nlon))
    print(f"\n=== {name.upper()} ({len(cols)} ô, {time.time() - t0:.0f}s) ===")
    for k in (2, 3, 1, 0):
        s = cg == k
        print(f"{CLASS_LABEL[k]:<20}: {int(s.sum()):>6} ô | {100 * wv[s].sum():5.1f}% diện tích lưu vực")
    lab, mul = units[name]
    sig = cls >= 2
    print(f"độ dốc trung vị (mọi ô xét): {np.nanmedian(slope) * mul:.3f} {lab}"
          + (f" | trung vị ở ô có bằng chứng: {np.nanmedian(slope[sig]) * mul:.3f}" if sig.any() else ""))
    summary[name] = {}
    for sc, drop in SCENARIOS.items():
        keep = ~np.isin(years, drop)
        s2, bh2, by2 = trend_layer(Xs[keep], years[keep])
        c2 = classify(s2, by2 if MAIN_Q == "q_by_hr" else bh2)
        kept = float(((c2 == cls) & sig).sum() / max(sig.sum(), 1))
        sign_ok = float((np.sign(s2) == np.sign(slope)).mean())
        data[f"{name}_class_{sc}"] = (dims, base_class(c2, cols).reshape(nlat, nlon))
        summary[name][sc] = {"kept_significant_share": kept, "sign_agreement": sign_ok}
        print(f"  độ nhạy [{sc}]: ô có bằng chứng giữ nguyên lớp {100 * kept:.1f}% | cùng dấu {100 * sign_ok:.1f}%")

# ---------------- ghi file ----------------
ds = xr.Dataset(data, coords={"lat": monthly["lat"], "lon": monthly["lon"]})
ds.attrs.update({
    "title": "Seasonality metrics per cell (PCI, annual harmonic amplitude and phase), 2001-2024",
    "pci": "Oliver PCI = 100*sum(P^2)/sum(P)^2 over 12 monthly totals of each calendar year; minimum 8.33",
    "harmonic": "a,b = 2/12*sum P cos/sin(2*pi*m/12), m=1..12; amplitude=hypot(a,b); phase anomaly = wrapped difference "
                "to the phase of the mean seasonal cycle, in months (positive = later peak)",
    "phase_rule": f"phase trend only where annual wave explains >= {R2_MIN} of seasonal-cycle variance",
    "evidence_class_codes": json.dumps(CLASS_LABEL, ensure_ascii=False),
    "sensitivity": json.dumps(summary),
    "note": "Exploratory layers: each layer is its own FDR family; none is combined with the annual-total map.",
})
enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars if ds[v].dtype.kind == "f"}
ds.to_netcdf(PROC / "season_cells.nc", encoding=enc)
print("\nĐã lưu:", PROC / "season_cells.nc")