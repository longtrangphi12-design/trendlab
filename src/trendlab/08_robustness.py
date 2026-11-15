"""Việc 8: kiểm tra độ bền của kết quả chính + xu hướng nhỏ nhất mà chuỗi phát hiện được.

Chạy (từ thư mục gốc repo):  python src/trendlab/08_robustness.py
Đọc : data/processed/annual.nc, monthly.nc, trend_cells.nc (từ script 06), data/interim/amazon_mask.nc
Ghi : data/processed/robustness_cells.nc, data/processed/regions_robustness.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from trendlab.multitest import benjamini_hochberg
from trendlab.trend import mann_kendall, sen_slope
from trendlab.trend_extra import benjamini_yekutieli, hamed_rao

ROOT = Path(__file__).resolve().parents[2]
INT, PROC = ROOT / "data/interim", ROOT / "data/processed"

# ---------------- CẤU HÌNH (chốt TRƯỚC khi xem kết quả) ----------------
ALPHA = 0.05
CELL_SCENARIOS = {                       # đối chứng mất sức: bỏ 3 năm ở ĐẦU và 3 năm Ở GIỮA chuỗi
    "drop_2001_03": [2001, 2002, 2003],
    "drop_2011_13": [2011, 2012, 2013],
}
# tiểu vùng theo dải vĩ độ (lo <= lat < hi). CHỌN SAU KHI ĐÃ XEM bản đồ nên ghi rõ trong methods;
# luôn đối chiếu với "toàn lưu vực".
REGIONS = {"toan_luu_vuc": (-90, 90), "bac_lat>=-3": (-3, 90),
           "giua_-10<=lat<-3": (-10, -3), "nam_lat<-10": (-90, -10)}
BOOT_B, BOOT_L = 2000, 3                 # bootstrap khối: 2000 mẫu, khối dài 3 năm
Z_POWER = 1.96 + 0.84                    # 5% hai phía, công suất 80%


# ---------------- hàm phụ ----------------
def classify(slope, q):
    cls = np.ones(len(slope), dtype=np.int8)
    sig = q <= ALPHA
    cls[sig & (slope > 0)] = 2
    cls[sig & (slope < 0)] = 3
    return cls


def sen_many(X, t):
    """Sen cho nhiều chuỗi cùng lúc: X (B, n) -> (B,)."""
    i, j = np.triu_indices(X.shape[1], 1)
    return np.median((X[:, j] - X[:, i]) / (t[j] - t[i]), axis=1)


def block_boot_ci(x, t, L=BOOT_L, B=BOOT_B, seed=0):
    """Khoảng 95% của độ dốc Sen: bootstrap theo khối trên phần dư quanh đường Sen (giữ tự tương quan ngắn)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    beta = sen_slope(x, t)
    fit = np.median(x - beta * t) + beta * t
    e = x - fit
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, nb))
    ix = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n]
    sb = sen_many(fit[None, :] + e[ix], t)
    return np.percentile(sb, [2.5, 97.5])


def window_scan(x, years, w=3):
    """Bỏ lần lượt mọi cửa sổ w năm liên tiếp, trả về (năm đầu cửa sổ, độ dốc Sen còn lại)."""
    out = []
    for s in range(len(x) - w + 1):
        keep = np.ones(len(x), bool)
        keep[s:s + w] = False
        out.append((int(years[s]), sen_slope(x[keep], years[keep].astype(float))))
    return out


def run_cells(A, years, cols):
    slope, p = np.full(len(cols), np.nan), np.full(len(cols), np.nan)
    t = years.astype(float)
    t0 = time.time()
    for c, col in enumerate(cols):
        x = A[:, col]
        slope[c] = sen_slope(x, t)
        p[c] = hamed_rao(x)["p"]
        if (c + 1) % 10000 == 0:
            print(f"  {c + 1}/{len(cols)} ô ({time.time() - t0:.0f}s)")
    return slope, benjamini_yekutieli(p)


# ---------------- nạp ----------------
annual = xr.open_dataset(PROC / "annual.nc")["annual"]
monthly = xr.open_dataset(PROC / "monthly.nc")["monthly"]
tc = xr.open_dataset(PROC / "trend_cells.nc")
grid = xr.open_dataset(INT / "amazon_mask.nc")
w = grid["weight"].values
years = annual["year"].values
ny, nlat, nlon = annual.shape
assert ny == 24 and monthly.sizes["time"] == 288
A = annual.values.reshape(ny, -1).astype(float)
cls_main = tc["evidence_class"].values.ravel()
slope_main = tc["slope_mm_per_yr"].values.ravel()
cols = np.flatnonzero(cls_main >= 1)                  # ô đã được kiểm định ở script 06
print(f"Ô đã kiểm định: {len(cols)}")
t = years.astype(float)

# ---------------- 1) xu hướng nhỏ nhất phát hiện được, từng ô ----------------
X = A[:, cols]
beta = slope_main[cols]
res = X - beta[None, :] * t[:, None]
res = res - np.median(res, axis=0)[None, :]
sd = np.sqrt((res ** 2).sum(0) / (ny - 2))            # độ lệch chuẩn phần dư quanh đường Sen
mds = Z_POWER * sd / np.sqrt(ny * (ny ** 2 - 1) / 12)  # mm/năm
mean_mm = X.mean(0)
mds_pct = 100 * 10 * mds / mean_mm
print("\n=== XU HƯỚNG NHỎ NHẤT PHÁT HIỆN ĐƯỢC (n = 24, 5%, công suất 80%, CHƯA tính FDR/tự tương quan) ===")
print(f"mm/năm        : 5% {np.percentile(mds, 5):.1f} | trung vị {np.median(mds):.1f} | 95% {np.percentile(mds, 95):.1f}")
print(f"% mỗi thập kỷ : 5% {np.percentile(mds_pct, 5):.1f} | trung vị {np.median(mds_pct):.1f} | 95% {np.percentile(mds_pct, 95):.1f}")
gray = cls_main[cols] == 1
print(f"Ô 'chưa đủ bằng chứng': MDS trung vị {np.median(mds[gray]):.1f} mm/năm; "
      f"{100 * (np.abs(beta[gray]) >= mds[gray]).mean():.1f}% trong số đó có |độ dốc ước lượng| >= MDS")
print(f"Độ dốc ước lượng trung vị |Sen|: {np.median(np.abs(beta)):.1f} mm/năm (so với MDS trung vị {np.median(mds):.1f})")

# ---------------- 2) đối chứng mất sức ở từng ô ----------------
print("\n=== ĐỐI CHỨNG MẤT SỨC (bỏ 3 năm ở đầu / ở giữa chuỗi) ===")
out_classes, cell_summary = {}, {}
sig_main = cls_main[cols] >= 2
for name, drop in CELL_SCENARIOS.items():
    keep = ~np.isin(years, drop)
    s2, q2 = run_cells(A[keep], years[keep], cols)
    c2 = classify(s2, q2)
    kept = float(((c2 == cls_main[cols]) & sig_main).sum() / max(sig_main.sum(), 1))
    sign_ok = float((np.sign(s2) == np.sign(beta)).mean())
    g = np.full(nlat * nlon, -1, dtype=np.int8)
    g[grid["mask"].values.ravel().astype(bool)] = 0
    g[cols] = c2
    out_classes[f"class_{name}"] = (("lat", "lon"), g.reshape(nlat, nlon))
    cell_summary[name] = {"dropped_years": drop, "kept_significant_share": kept, "sign_agreement": sign_ok}
    print(f"[{name}] bỏ {drop}: ô có bằng chứng giữ nguyên lớp {100 * kept:.1f}% | cùng dấu {100 * sign_ok:.1f}% | "
          f"tăng/giảm có bằng chứng: {int((c2 == 2).sum())}/{int((c2 == 3).sum())}")

# ---------------- 3) tiểu vùng: Sen, bootstrap khối, quét cửa sổ, năm thủy văn ----------------
lat2d = np.repeat(annual["lat"].values[:, None], nlon, axis=1).ravel()
ok_cells = np.isfinite(A).all(0) & (w.ravel() > 0)
M = monthly.values.reshape(288, -1).astype(float)
rows = []
print("\n=== TIỂU VÙNG (chuỗi trung bình có trọng số diện tích) ===")
for name, (lo, hi) in REGIONS.items():
    sel = ok_cells & (lat2d >= lo) & (lat2d < hi)
    if sel.sum() == 0:
        continue
    wr = w.ravel()[sel] / w.ravel()[sel].sum()
    x = A[:, sel] @ wr                                     # 24 giá trị mm/năm
    xm = M[:, sel] @ wr                                    # 288 giá trị mm/tháng
    b = sen_slope(x, t)
    mk, hr = mann_kendall(x), hamed_rao(x)
    ci = block_boot_ci(x, t)
    sc = window_scan(x, years)
    sl = np.array([s for _, s in sc])
    flip = [y for y, s in sc if np.sign(s) != np.sign(b)]
    # năm thủy văn: bắt đầu từ tháng khô nhất của khí hậu vùng đó
    m0 = int(np.argmin(xm.reshape(24, 12).mean(0)))
    nblk = (288 - m0) // 12
    hy = xm[m0:m0 + 12 * nblk].reshape(nblk, 12).sum(1)
    hy_years = (2001 + np.arange(nblk)).astype(float)
    hb, hp = sen_slope(hy, hy_years), hamed_rao(hy)["p"]
    sd_r = np.std(x - (np.median(x - b * t) + b * t), ddof=2)
    r = dict(region=name, n_cells=int(sel.sum()), mean_mm=x.mean(), sen_mm_yr=b, p_mk=mk["p"], p_hr=hr["p"],
             boot_lo=ci[0], boot_hi=ci[1], boot_excludes_zero=bool(ci[0] > 0 or ci[1] < 0),
             drop3_min=sl.min(), drop3_max=sl.max(), drop3_sign_flips=len(flip),
             drop3_flip_windows=";".join(f"{y}-{y + 2}" for y in flip),
             hydro_start_month=m0 + 1, hydro_n=nblk, hydro_sen=hb, hydro_p_hr=hp,
             mds_mm_yr=Z_POWER * sd_r / np.sqrt(ny * (ny ** 2 - 1) / 12))
    rows.append(r)
    print(f"\n[{name}] {r['n_cells']} ô, mưa TB {r['mean_mm']:.0f} mm/năm")
    print(f"  Sen {b:.1f} mm/năm | MK p={mk['p']:.3f} | Hamed-Rao p={hr['p']:.3f}")
    print(f"  Bootstrap khối 95%: [{ci[0]:.1f}, {ci[1]:.1f}] -> {'loại trừ 0' if r['boot_excludes_zero'] else 'KHÔNG loại trừ 0'}")
    print(f"  Bỏ 3 năm liên tiếp (quét {len(sc)} cửa sổ): Sen từ {sl.min():.1f} đến {sl.max():.1f}; "
          f"đổi dấu ở {len(flip)} cửa sổ" + (f" ({r['drop3_flip_windows']})" if flip else ""))
    print(f"  Năm thủy văn (bắt đầu tháng {m0 + 1}, n={nblk}): Sen {hb:.1f} mm/năm, Hamed-Rao p={hp:.3f}")
    print(f"  Xu hướng nhỏ nhất phát hiện được ở chuỗi này: ~{r['mds_mm_yr']:.1f} mm/năm")
pd.DataFrame(rows).to_csv(PROC / "regions_robustness.csv", index=False)


# ---------------- ghi file ----------------
def to_grid(vals):
    g = np.full(nlat * nlon, np.nan, dtype="float32")
    g[cols] = vals
    return g.reshape(nlat, nlon)


ds = xr.Dataset({"resid_sd_mm": (("lat", "lon"), to_grid(sd)),
                 "mds_mm_per_yr": (("lat", "lon"), to_grid(mds)),
                 "mds_pct_per_decade": (("lat", "lon"), to_grid(mds_pct)),
                 "slope_over_mds": (("lat", "lon"), to_grid(np.abs(beta) / mds)),
                 **out_classes},
                coords={"lat": annual["lat"], "lon": annual["lon"]})
ds.attrs.update({
    "mds": "minimum detectable slope ~ 2.8 * residual SD / sqrt(n(n^2-1)/12); optimistic lower bound "
           "(no FDR, no autocorrelation); describes what a 24-year series could detect, not a forecast",
    "cell_sensitivity": json.dumps(cell_summary),
    "regions": "latitude bands chosen after viewing the maps; always compare with the whole basin",
})
enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars if ds[v].dtype.kind == "f"}
ds.to_netcdf(PROC / "robustness_cells.nc", encoding=enc)
print("\nĐã lưu:", PROC / "robustness_cells.nc", "và", PROC / "regions_robustness.csv")