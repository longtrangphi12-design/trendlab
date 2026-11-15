"""Việc 6: chạy toàn lưu vực, lưu kết quả từng ô thành file tĩnh cho giao diện.

Chạy (từ thư mục gốc repo):  python src/trendlab/06_run_basin.py
Đọc : data/processed/annual.nc, data/interim/amazon_mask.nc (mask + weight)
Ghi : data/processed/trend_cells.nc   (mỗi biến là một lớp lưới lat x lon)
      data/processed/basin_annual.csv (mưa năm trung bình lưu vực, có trọng số diện tích)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from trendlab.autocorr import lag1_autocorr
from trendlab.multitest import benjamini_hochberg
from trendlab.trend import mann_kendall, sen_slope
from trendlab.trend_extra import benjamini_yekutieli, hamed_rao, pettitt

ROOT = Path(__file__).resolve().parents[2]
INT, PROC = ROOT / "data/interim", ROOT / "data/processed"

# ---------------- CẤU HÌNH (đổi ở đây, không sửa chỗ khác) ----------------
ALPHA = 0.05
MIN_VALID = 20          # ô cần >= 20/24 năm hợp lệ (80%, cùng quy tắc với report.py)
MAIN_Q = "q_by_hr"      # "q_by_hr" (BY, đúng với mọi phụ thuộc) hoặc "q_bh_hr" (BH)
SCENARIOS = {           # kiểm tra độ nhạy: bỏ các năm này rồi chạy lại
    "drop_drought": [2005, 2010, 2015, 2016],   # danh sách hạn đã có trong bản prototype
    "drop_2022_24": [2022, 2023, 2024],         # THÊM SAU khi đã thấy khối thử: ghi rõ trong methods
}
CLASS_LABEL = {-1: "ngoài lưu vực", 0: "không đủ dữ liệu", 1: "chưa đủ bằng chứng",
               2: "tăng có bằng chứng", 3: "giảm có bằng chứng"}


# ---------------- hàm phụ ----------------
def run(A, years, idx, full):
    """Chạy các phép kiểm định cho từng cột (ô) idx của A (thời gian x ô)."""
    keys = ["mean_mm", "slope", "p_mk", "p_hr", "hr_factor", "hr_fallback"]
    if full:
        keys += ["r1", "pettitt_p", "pettitt_year"]
    out = {k: np.full(len(idx), np.nan) for k in keys}
    t = years.astype(float)
    t0 = time.time()
    for c, col in enumerate(idx):
        x = A[:, col].astype(float)
        out["mean_mm"][c] = np.nanmean(x)
        out["slope"][c] = sen_slope(x, t)
        out["p_mk"][c] = mann_kendall(x)["p"]
        hr = hamed_rao(x)
        out["p_hr"][c], out["hr_factor"][c] = hr["p"], hr["factor"]
        out["hr_fallback"][c] = float(hr["fallback"])
        if full:
            out["r1"][c] = lag1_autocorr(x)
            pt = pettitt(x)
            out["pettitt_p"][c] = pt["p"]
            if pt["change_index"] >= 0:      # năm ĐẦU của giai đoạn sau điểm gãy
                out["pettitt_year"][c] = t[np.isfinite(x)][pt["change_index"] + 1]
        if (c + 1) % 10000 == 0:
            print(f"  {c + 1}/{len(idx)} ô ({time.time() - t0:.0f}s)")
    out["q_bh_mk"] = benjamini_hochberg(out["p_mk"])
    out["q_bh_hr"] = benjamini_hochberg(out["p_hr"])
    out["q_by_hr"] = benjamini_yekutieli(out["p_hr"])
    return out


def classify(slope, q):
    """1 = chưa đủ bằng chứng, 2 = tăng có bằng chứng, 3 = giảm có bằng chứng."""
    cls = np.ones(len(slope), dtype=np.int8)
    sig = q <= ALPHA
    cls[sig & (slope > 0)] = 2
    cls[sig & (slope < 0)] = 3
    return cls


def morans_i(field):
    """Moran's I toàn cục, láng giềng Queen (8 ô quanh), bỏ ô NaN. Chỉ để chẩn đoán."""
    ok = np.isfinite(field)
    z = np.where(ok, field - np.nanmean(field), 0.0)
    zp, op = np.pad(z, 1), np.pad(ok.astype(float), 1)
    H, W = z.shape
    num, wsum = 0.0, 0.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            nb = zp[1 + di:1 + di + H, 1 + dj:1 + dj + W]
            both = ok & (op[1 + di:1 + di + H, 1 + dj:1 + dj + W] > 0)
            num += (z * nb * both).sum()
            wsum += both.sum()
    return float(ok.sum() / wsum * num / (z ** 2).sum())


# ---------------- nạp dữ liệu và kiểm tra ----------------
annual = xr.open_dataset(PROC / "annual.nc")["annual"]
grid_ds = xr.open_dataset(INT / "amazon_mask.nc")
mask, w = grid_ds["mask"].astype(bool), grid_ds["weight"]
assert annual.dims == ("year", "lat", "lon"), annual.dims
assert np.allclose(annual["lat"], mask["lat"]) and np.allclose(annual["lon"], mask["lon"]), \
    "lưới annual.nc và mặt nạ không khớp"
years = annual["year"].values
assert len(years) == 24, f"kỳ vọng 24 năm, có {len(years)}"
ny, nlat, nlon = annual.shape

A = annual.values.reshape(ny, -1)
in_basin = mask.values.ravel()
n_valid = np.isfinite(A).sum(0)
use = in_basin & (n_valid >= MIN_VALID)
idx = np.flatnonzero(use)
print(f"Ô trong lưu vực: {int(in_basin.sum())} | đủ dữ liệu để kiểm định: {len(idx)}")
assert len(idx) > 0

# ---------------- chạy chính ----------------
t0 = time.time()
print("Chạy chính (MK, Sen, Hamed-Rao, Pettitt, lag-1) ...")
res = run(A, years, idx, full=True)
cls = classify(res["slope"], res[MAIN_Q])
res["slope_pct_decade"] = 100 * 10 * res["slope"] / res["mean_mm"]
print(f"Xong chạy chính sau {time.time() - t0:.0f}s")


def to_grid(vals, dtype="float32", fill=np.nan):
    g = np.full(nlat * nlon, fill, dtype=dtype)
    g[idx] = vals
    return g.reshape(nlat, nlon)


cls_full = np.full(nlat * nlon, -1, dtype=np.int8)
cls_full[in_basin] = 0
cls_full[idx] = cls
cls_grid = cls_full.reshape(nlat, nlon)

dims = ("lat", "lon")
ds = xr.Dataset(
    {
        "evidence_class": (dims, cls_grid),
        "slope_mm_per_yr": (dims, to_grid(res["slope"])),
        "slope_pct_per_decade": (dims, to_grid(res["slope_pct_decade"])),
        "mean_mm_per_yr": (dims, to_grid(res["mean_mm"])),
        "p_mk": (dims, to_grid(res["p_mk"])),
        "p_hr": (dims, to_grid(res["p_hr"])),
        "q_bh_mk": (dims, to_grid(res["q_bh_mk"])),
        "q_bh_hr": (dims, to_grid(res["q_bh_hr"])),
        "q_by_hr": (dims, to_grid(res["q_by_hr"])),
        "hr_factor": (dims, to_grid(res["hr_factor"])),
        "hr_fallback": (dims, to_grid(res["hr_fallback"], dtype="float32")),
        "lag1_r1": (dims, to_grid(res["r1"])),
        "pettitt_p": (dims, to_grid(res["pettitt_p"])),
        "pettitt_first_year_after": (dims, to_grid(res["pettitt_year"])),
        "n_valid_years": (dims, n_valid.reshape(nlat, nlon).astype("int16")),
        "area_weight": (dims, w.values.astype("float32")),
    },
    coords={"lat": annual["lat"], "lon": annual["lon"]},
)

# ---------------- kiểm tra độ nhạy (đã định sẵn) ----------------
basin_series = (annual * w).sum(("lat", "lon"))        # trung bình lưu vực có trọng số
bs = basin_series.values
print("\n=== ĐỘ NHẠY: bỏ năm rồi chạy lại ===")
sens_summary = {}
for name, drop in SCENARIOS.items():
    keep = ~np.isin(years, drop)
    r = run(A[keep], years[keep], idx, full=False)
    c2 = classify(r["slope"], r[MAIN_Q])
    same = float((c2 == cls).mean())
    sig_main = cls >= 2
    kept = float(((c2 == cls) & sig_main).sum() / max(sig_main.sum(), 1))
    sign_ok = float((np.sign(r["slope"]) == np.sign(res["slope"])).mean())
    bs_slope = sen_slope(bs[keep], years[keep].astype(float))
    print(f"[{name}] bỏ {drop}: cùng lớp {100 * same:.1f}% ô | "
          f"ô có bằng chứng ở chạy chính vẫn giữ nguyên lớp: {100 * kept:.1f}% | "
          f"cùng dấu Sen: {100 * sign_ok:.1f}% | tăng/giảm có bằng chứng: "
          f"{int((c2 == 2).sum())}/{int((c2 == 3).sum())} | Sen trung bình lưu vực: {bs_slope:.1f} mm/năm")
    ds[f"class_{name}"] = (dims, np.where(cls_grid >= 1, to_grid(c2, dtype="int8", fill=0), cls_grid).astype("int8"))
    sens_summary[name] = {"dropped_years": drop, "same_class_share": same,
                          "kept_significant_share": kept, "sign_agreement": sign_ok}

# ---------------- tóm tắt toàn lưu vực ----------------
wv = w.values.ravel()
print("\n=== TÓM TẮT TOÀN LƯU VỰC (chạy chính) ===")
print(f"Quy tắc bằng chứng: {MAIN_Q} <= {ALPHA}, m = {len(idx)} ô")
for k in (2, 3, 1, 0):
    sel = cls_full == k
    print(f"{CLASS_LABEL[k]:<22}: {int(sel.sum()):>6} ô | {100 * wv[sel].sum():5.1f}% diện tích lưu vực")
print(f"MK thô p<=0,05: {int((res['p_mk'] <= ALPHA).sum())} ô | HR thô: {int((res['p_hr'] <= ALPHA).sum())} ô | "
      f"BH(HR): {int((res['q_bh_hr'] <= ALPHA).sum())} | BY(HR): {int((res['q_by_hr'] <= ALPHA).sum())}")
print(f"Hệ số HR (n/n*): nhỏ nhất {np.nanmin(res['hr_factor']):.2f}, trung vị {np.nanmedian(res['hr_factor']):.2f}, "
      f"lớn nhất {np.nanmax(res['hr_factor']):.2f} | ô quay về MK thường: {int(np.nansum(res['hr_fallback']))}")
print(f"Lag-1 r1: trung vị {np.nanmedian(res['r1']):.2f}")
print(f"Sen slope: trung vị {np.nanmedian(res['slope']):.1f} mm/năm "
      f"(5%..95%: {np.nanpercentile(res['slope'], 5):.1f} .. {np.nanpercentile(res['slope'], 95):.1f})")
mi = morans_i(ds["slope_mm_per_yr"].values)
print(f"Moran's I của độ dốc Sen (Queen): {mi:.2f}  (gần 1 = các ô lân cận rất giống nhau, "
      "nên KHÔNG coi mỗi ô là một bằng chứng độc lập)")

hr_b = hamed_rao(bs)
print(f"\nTrung bình lưu vực (một chuỗi): Sen {sen_slope(bs, years.astype(float)):.1f} mm/năm, "
      f"MK p={mann_kendall(bs)['p']:.3f}, Hamed-Rao p={hr_b['p']:.3f}")
print(f"Mưa năm trung bình lưu vực: {bs.mean():.0f} mm/năm (kỳ vọng cỡ 2000-2500)")

# ---------------- ghi file tĩnh ----------------
ds.attrs.update({
    "title": "Amazon basin annual rainfall trend per grid cell, 2001-2024",
    "source": "GPM IMERG Final V07 monthly (GPM_3IMERGM), NASA GES DISC; basin: HydroBASINS lev03",
    "units_annual": "mm/year (monthly mm/hr x real hours per month, summed over calendar year)",
    "method": "Sen slope; Mann-Kendall with Hamed-Rao variance correction; "
              f"FDR adjustment over all valid cells; main evidence rule {MAIN_Q} <= {ALPHA}",
    "evidence_class_codes": json.dumps(CLASS_LABEL, ensure_ascii=False),
    "min_valid_years": MIN_VALID,
    "morans_i_slope": mi,
    "sensitivity": json.dumps(sens_summary),
    "note": "Magnitude classes (mild/moderate/strong) are NOT set: thresholds still undecided.",
})
enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars if ds[v].dtype.kind == "f"}
ds.to_netcdf(PROC / "trend_cells.nc", encoding=enc)
pd.DataFrame({"year": years, "basin_mean_mm": bs}).to_csv(PROC / "basin_annual.csv", index=False)
print("\nĐã lưu:", PROC / "trend_cells.nc", "và", PROC / "basin_annual.csv")