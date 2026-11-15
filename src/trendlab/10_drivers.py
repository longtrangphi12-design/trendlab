"""Việc 10: nhánh GIẢI THÍCH. Biến động mưa năm theo ENSO (Niño 3.4) và Đại Tây Dương nhiệt đới bắc (TNA).

Hai câu hỏi, hai phép tính (chốt trước khi chạy, không chọn lại sau khi thấy kết quả):
 A) Biến động giữa các năm: hồi quy OLS bội trên chuỗi ĐÃ LOẠI XU HƯỚNG tuyến tính (y, x1, x2 đều loại).
    Báo R2, R2 hiệu chỉnh, R2 kiểm chứng chéo (bỏ từng khối 3 năm liên tiếp).
 B) Xu hướng còn lại: y_adj = y - b1*x1 - b2*x2, với b từ mô hình y ~ năm + x1 + x2 ("joint", bản CHÍNH).
    Bản nhạy: b từ y ~ x1 + x2 (không có năm, "naive"). Sau đó chạy lại Sen + MK-Hamed-Rao + BY như bản chính.
Chỉ số: dị thường TS (so với khí hậu từng tháng 2001-2024), trung bình năm dương lịch, cùng năm với mưa (không trễ), chuẩn hóa (z).
Kết quả mô tả thống kê, KHÔNG chứng minh nhân quả.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from trendlab.trend_extra import hamed_rao   # giống dòng import trong 06_run_basin.py

ROOT = Path(__file__).resolve().parents[2]
INT, PROC = ROOT / "data/interim", ROOT / "data/processed"
ALPHA, BLOCK = 0.05, 3
REGIONS = {"toan_luu_vuc": lambda lat: lat > -99, "bac_lat>=-3": lambda lat: lat >= -3,
           "giua_-10<=lat<-3": lambda lat: (lat >= -10) & (lat < -3), "nam_lat<-10": lambda lat: lat < -10}


def index_annual(path):
    """Dị thường TS theo tháng (trung bình có trọng số cos(lat) trong hộp) -> trung bình năm."""
    ts = xr.open_dataset(path)["TS"]
    w = np.cos(np.deg2rad(ts["lat"]))
    s = ts.weighted(w).mean(("lat", "lon"))
    anom = s.groupby("time.month") - s.groupby("time.month").mean("time")
    return anom.groupby("time.year").mean("time").to_series()


def lstsq(X, Y):
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def detrend(A, t):
    T = np.column_stack([np.ones_like(t), t])
    return A - T @ lstsq(T, A)


def sen_hr(x, yrs):
    d = hamed_rao(x, ALPHA)
    return stats.theilslopes(x, yrs)[0], d["p"]


def by_q(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = p[o] * m * np.sum(1.0 / np.arange(1, m + 1)) / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out


def classify(Yadj, yrs, ok_idx):
    """Sen + HR + BY trên từng cột; trả (slope, q, class) cho các cột ok_idx."""
    n = len(ok_idx)
    slope, p = np.empty(n), np.empty(n)
    for k, j in enumerate(ok_idx):
        slope[k], p[k] = sen_hr(Yadj[:, j], yrs)
    q = by_q(p)
    cls = np.where(q <= ALPHA, np.where(slope > 0, 2, 3), 1).astype(np.int8)
    return slope, q, cls


# ---------- dữ liệu ----------
tc = xr.open_dataset(PROC / "trend_cells.nc")
ann = xr.open_dataset(PROC / "annual.nc")["annual"]
cls0 = tc["evidence_class"].values
lat2d = np.repeat(tc["lat"].values[:, None], tc.sizes["lon"], 1)
W = tc["area_weight"].values
yrs = ann["year"].values.astype(float)
n = len(yrs)
t = yrs - yrs.mean()

x1y, x2y = index_annual(INT / "ts_nino34.nc"), index_annual(INT / "ts_tna.nc")
idx = pd.concat([x1y.rename("nino34"), x2y.rename("tna")], axis=1).loc[yrs.astype(int)]
assert len(idx) == n and not idx.isna().any().any(), "chỉ số biển thiếu năm"
Z = ((idx - idx.mean()) / idx.std(ddof=1)).values            # (n,2), chuẩn hóa
Zd = detrend(Z, t)
r = np.corrcoef(Z.T)[0, 1]
rd = np.corrcoef(Zd.T)[0, 1]
print(f"Tương quan Niño3.4–TNA: {r:.2f} (đã loại xu hướng: {rd:.2f}) | VIF = {1 / (1 - rd**2):.2f}")
for j, nm in enumerate(("nino34", "tna")):
    sl, p = sen_hr(Z[:, j], yrs)
    print(f"Xu hướng của chỉ số {nm}: Sen {sl:+.3f} SD/năm, p = {p:.3f}")
idx.to_csv(PROC / "drivers_indices.csv", index_label="year")

Y2 = ann.transpose("year", "lat", "lon").values.reshape(n, -1)
tested = (cls0 >= 1).reshape(-1)
full = np.isfinite(Y2).all(0)
ok = np.where(tested & full)[0]
print(f"Ô đã kiểm định ở 06: {tested.sum()} | đủ {n} năm để chạy nhánh này: {len(ok)}")
Y = Y2[:, ok].astype(float)

# ---------- A) biến động giữa các năm (chuỗi đã loại xu hướng) ----------
Yd = detrend(Y, t)
D = np.column_stack([np.ones(n), Zd])
res = Yd - D @ lstsq(D, Yd)
sst = (Yd**2).sum(0)
r2 = 1 - (res**2).sum(0) / sst
r2adj = 1 - (1 - r2) * (n - 1) / (n - 3)
beta = lstsq(D, Yd)                                           # (3, N): const, nino, tna (mm / 1 SD)
pred = np.empty_like(Yd)
for s in range(0, n, BLOCK):
    te = np.arange(s, min(s + BLOCK, n))
    tr = np.setdiff1d(np.arange(n), te)
    pred[te] = D[te] @ lstsq(D[tr], Yd[tr])
r2cv = 1 - ((Yd - pred) ** 2).sum(0) / sst

# ---------- B) xu hướng còn lại ----------
Xj = np.column_stack([np.ones(n), t, Z])
bj = lstsq(Xj, Y)
Yadj_joint = Y - Z @ bj[2:]
Xn = np.column_stack([np.ones(n), Z])
bn = lstsq(Xn, Y)
Yadj_naive = Y - Z @ bn[1:]
sl_j, q_j, c_j = classify(Yadj_joint, yrs, range(len(ok)))
sl_n, q_n, c_n = classify(Yadj_naive, yrs, range(len(ok)))

# ---------- ghi file ----------
def put(vals, fill=np.nan, dtype=float):  # trả (dims, mảng) để xr.Dataset nhận
    a = np.full(tested.shape, fill, dtype=dtype)
    a[ok] = vals
    return ("lat", "lon"), a.reshape(cls0.shape)

out = xr.Dataset(
    {"r2_detrended": put(r2), "r2_adj_detrended": put(r2adj), "r2_cv_detrended": put(r2cv),
     "beta_nino34_mm_per_sd": put(beta[1]), "beta_tna_mm_per_sd": put(beta[2]),
     "slope_adj_mm_per_yr": put(sl_j), "q_by_adj": put(q_j), "class_adj_joint": put(c_j, 0, np.int8),
     "class_adj_naive": put(c_n, 0, np.int8), "slope_adj_naive_mm_per_yr": put(sl_n)},
    coords={"lat": tc["lat"], "lon": tc["lon"]})
out.attrs["note"] = ("class codes as evidence_class (1 not enough, 2 up, 3 down; 0 = not run). "
                     "Descriptive statistics, not causal attribution. Indices: Nino3.4 and TNA from MERRA-2 TS, same-year, standardised.")
out.to_netcdf(PROC / "drivers_cells.nc")

# ---------- tóm tắt ----------
c_main = cls0.reshape(-1)[ok]
wok = W.reshape(-1)[ok]
print("\nR2 (đã loại xu hướng) trung vị: %.2f | R2 hiệu chỉnh: %.2f | R2 kiểm chứng chéo: %.2f"
      % (np.median(r2), np.median(r2adj), np.median(r2cv)))
print("Ô có R2 kiểm chứng chéo > 0: %.1f%%" % (100 * (r2cv > 0).mean()))
dec = c_main == 3
for nm, c in (("joint (chính)", c_j), ("naive (nhạy)", c_n)):
    print(f"\n[{nm}] ô GIẢM có bằng chứng ban đầu: {dec.sum()} -> vẫn giảm: {(c[dec] == 3).sum()} "
          f"({100 * (c[dec] == 3).mean():.1f}%) | mới giảm: {((c == 3) & ~dec).sum()} | mới tăng: {(c == 2).sum()}")
    print(f"   tổng giảm sau hiệu chỉnh: {(c == 3).sum()} ({100 * wok[c == 3].sum() / W.reshape(-1)[tested].sum():.1f}% diện tích)")

# ---------- bảng theo vùng ----------
rows = []
Wt, Yf = W.reshape(-1)[ok], Y
lat_ok = lat2d.reshape(-1)[ok]
for nm, f in REGIONS.items():
    m = f(lat_ok)
    w = Wt[m] / Wt[m].sum()
    y = (Yf[:, m] * w).sum(1)
    sl0, p0 = sen_hr(y, yrs)
    yd = detrend(y[:, None], t)[:, 0]
    b, *_ = np.linalg.lstsq(D, yd, rcond=None)
    rs = yd - D @ b
    s2 = (rs**2).sum() / (n - 3)
    se = np.sqrt(s2 * np.diag(np.linalg.inv(D.T @ D)))
    pv = 2 * stats.t.sf(np.abs(b / se), n - 3)
    bj_r = lstsq(Xj, y)
    ya = y - Z @ bj_r[2:]
    sl1, p1 = sen_hr(ya, yrs)
    r2r = 1 - (rs**2).sum() / (yd**2).sum()
    rows.append(dict(region=nm, n_cells=int(m.sum()), r2_detrended=r2r,
                     beta_nino34_mm_per_sd=b[1], p_nino34=pv[1], beta_tna_mm_per_sd=b[2], p_tna=pv[2],
                     sen_raw=sl0, p_hr_raw=p0, sen_adj=sl1, p_hr_adj=p1))
tab = pd.DataFrame(rows)
tab.to_csv(PROC / "regions_drivers.csv", index=False)
pd.set_option("display.width", 200, "display.float_format", "{:.3g}".format)
print("\n", tab.to_string(index=False))
print("\nĐã lưu: drivers_cells.nc, regions_drivers.csv, drivers_indices.csv")