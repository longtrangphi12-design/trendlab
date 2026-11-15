"""Việc 12: xu hướng đã hiệu chỉnh ENSO + TNA có còn nếu bỏ 2022–2024?

Cùng mô hình "joint" của 10_drivers.py (y ~ năm + Niño3.4 + TNA, trừ phần chỉ số), nhưng khớp và kiểm định chỉ trên 2001–2021.
So với bản chưa hiệu chỉnh cũng bỏ 2022–2024 (class_drop_2022_24 trong trend_cells.nc).
Mô tả thống kê, không chứng minh nhân quả. Đây là kiểm tra độ nhạy đặt ra SAU khi thấy kết quả của 10, nên báo cáo như một kiểm tra phụ.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from trendlab.trend_extra import hamed_rao   # giống dòng import trong 06_run_basin.py

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
ALPHA, LAST = 0.05, 2021
REGIONS = {"toan_luu_vuc": lambda lat: lat > -99, "bac_lat>=-3": lambda lat: lat >= -3,
           "giua_-10<=lat<-3": lambda lat: (lat >= -10) & (lat < -3), "nam_lat<-10": lambda lat: lat < -10}


def lstsq(X, Y):
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def sen_hr(x, yrs):
    x = np.asarray(x, float)
    if np.ptp(x) == 0:
        return 0.0, 1.0
    p = hamed_rao(x, ALPHA)["p"]
    return stats.theilslopes(x, yrs)[0], (1.0 if not np.isfinite(p) else p)


def by_q(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = p[o] * m * np.sum(1.0 / np.arange(1, m + 1)) / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out


tc = xr.open_dataset(PROC / "trend_cells.nc")
ann = xr.open_dataset(PROC / "annual.nc")["annual"]
idx = pd.read_csv(PROC / "drivers_indices.csv", index_col="year")
cls0 = tc["evidence_class"].values
W = tc["area_weight"].values.reshape(-1)
lat = np.repeat(tc["lat"].values[:, None], tc.sizes["lon"], 1).reshape(-1)
yrs_all = ann["year"].values.astype(int)
keep = yrs_all <= LAST
yrs = yrs_all[keep].astype(float)
n = len(yrs)
t = yrs - yrs.mean()
I = idx.loc[yrs_all[keep]].values
Z = (I - idx.values.mean(0)) / idx.values.std(0, ddof=1)       # chuẩn hóa theo cả 24 năm, như bản 10
Y2 = ann.transpose("year", "lat", "lon").values.reshape(len(yrs_all), -1)
tested = (cls0 >= 1).reshape(-1)
ok = np.where(tested & np.isfinite(Y2).all(0))[0]
Y = Y2[keep][:, ok].astype(float)
X = np.column_stack([np.ones(n), t, Z])
Yadj = Y - Z @ lstsq(X, Y)[2:]


def classify(M):
    sl, p = np.empty(M.shape[1]), np.empty(M.shape[1])
    for k in range(M.shape[1]):
        sl[k], p[k] = sen_hr(M[:, k], yrs)
    q = by_q(p)
    return np.where(q <= ALPHA, np.where(sl > 0, 2, 3), 1).astype(np.int8)


c_raw, c_adj = classify(Y), classify(Yadj)
c_main = cls0.reshape(-1)[ok]
c_drop_saved = tc["class_drop_2022_24"].values.reshape(-1)[ok]
dec = c_main == 3
print(f"Chỉ dùng 2001–{LAST} (n = {n})")
print(f"Ô giảm có bằng chứng ở bản chính (24 năm): {dec.sum()}")
print(f"  bỏ 2022–24, KHÔNG hiệu chỉnh : còn giảm {(c_raw[dec] == 3).sum()} ({100 * (c_raw[dec] == 3).mean():.1f}%) | "
      f"(bản đã lưu class_drop_2022_24: {(c_drop_saved[dec] == 3).sum()} = {100 * (c_drop_saved[dec] == 3).mean():.1f}%)")
print(f"  bỏ 2022–24, CÓ hiệu chỉnh    : còn giảm {(c_adj[dec] == 3).sum()} ({100 * (c_adj[dec] == 3).mean():.1f}%)")
tot = W[tested].sum()
for nm, c in (("không hiệu chỉnh", c_raw), ("có hiệu chỉnh", c_adj)):
    print(f"  tổng ô giảm ({nm}): {(c == 3).sum()} = {100 * W[ok][c == 3].sum() / tot:.1f}% diện tích | tổng ô tăng: {(c == 2).sum()}")

rows = []
for nm, f in REGIONS.items():
    m = f(lat[ok])
    w = W[ok][m] / W[ok][m].sum()
    y = (Y[:, m] * w).sum(1)
    ya = y - Z @ lstsq(X, y)[2:]
    s0, p0 = sen_hr(y, yrs)
    s1, p1 = sen_hr(ya, yrs)
    rows.append(dict(region=nm, sen_raw_2001_2021=s0, p_raw=p0, sen_adj_2001_2021=s1, p_adj=p1))
tab = pd.DataFrame(rows)
tab.to_csv(PROC / "adjusted_robustness.csv", index=False)
pd.set_option("display.width", 200, "display.float_format", "{:.3g}".format)
print("\n", tab.to_string(index=False))
print("\nĐã lưu: adjusted_robustness.csv")


# ---- lưu phân loại 2001–2021 để các con số trên giao diện đọc từ file ----
def put_cls(v):
    a = np.zeros(tested.shape, np.int8)
    a[ok] = v
    return ("lat", "lon"), a.reshape(cls0.shape)


xr.Dataset({"class_raw_2001_2021": put_cls(c_raw), "class_adj_2001_2021": put_cls(c_adj)},
           coords={"lat": tc["lat"], "lon": tc["lon"]},
           attrs={"kept_share_raw": float((c_raw[dec] == 3).mean()), "kept_share_adj": float((c_adj[dec] == 3).mean()),
                  "n_main_decreasing": int(dec.sum()), "note": "1/2/3 as evidence_class; years 2001-2021 only"}
           ).to_netcdf(PROC / "adjusted_drop_cells.nc")
print("Đã lưu: adjusted_drop_cells.nc")