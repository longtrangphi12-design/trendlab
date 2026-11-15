"""Việc 11 (khám phá): (A) số tháng khô mỗi năm, (B) độ dốc Sen mở rộng theo năm kết thúc, (C) bản đồ ENSO hay TNA chi phối.

Quy ước (chốt trước khi chạy):
 A) Tháng khô = mưa < DRY_MM (100 mm/tháng) trong năm dương lịch; số tháng khô mỗi ô mỗi năm (0..12).
    Sen + MK-Hamed-Rao + BY như bản chính. Mã lớp: 1 chưa đủ bằng chứng, 2 TĂNG số tháng khô, 3 GIẢM số tháng khô.
    Phân tích KHÁM PHÁ: không thay thế bản đồ chính. Ngưỡng 100 mm cần dẫn nguồn trước khi dùng trên slide.
 B) Độ dốc Sen của 2001..Y cho Y = 2015..2024 (toàn lưu vực và 3 vùng). Các cửa sổ chồng lấn nhiều nên KHÔNG độc lập;
    p chưa hiệu chỉnh đa kiểm định, chỉ để mô tả độ nhạy với năm cuối.
 C) Từ drivers_cells.nc: ô nào |beta Niño3.4| > |beta TNA| (1) hay ngược lại (2), chỉ ở ô có R2 kiểm chứng chéo > 0; còn lại 0.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from trendlab.trend_extra import hamed_rao   # giống dòng import trong 06_run_basin.py

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
ALPHA, DRY_MM = 0.05, 100.0
REGIONS = {"toan_luu_vuc": lambda lat: lat > -99, "bac_lat>=-3": lambda lat: lat >= -3,
           "giua_-10<=lat<-3": lambda lat: (lat >= -10) & (lat < -3), "nam_lat<-10": lambda lat: lat < -10}


def sen_hr(x, yrs):
    x = np.asarray(x, float)
    if np.ptp(x) == 0:                       # chuỗi hằng (vd. luôn 0 tháng khô): không có xu hướng
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
cls0 = tc["evidence_class"].values
W = tc["area_weight"].values
lat2d = np.repeat(tc["lat"].values[:, None], tc.sizes["lon"], 1)
yrs = ann["year"].values.astype(float)
n = len(yrs)
Y2 = ann.transpose("year", "lat", "lon").values.reshape(n, -1)
tested = (cls0 >= 1).reshape(-1)
ok = np.where(tested & np.isfinite(Y2).all(0))[0]
Wf, latf = W.reshape(-1), lat2d.reshape(-1)

# ================= A) số tháng khô =================
mo = xr.open_dataset(PROC / "monthly.nc")["monthly"]
dry = (mo < DRY_MM).groupby("time.year").sum("time").where(ann.notnull())
D2 = dry.transpose("year", "lat", "lon").values.reshape(n, -1)[:, ok].astype(float)
slope, p = np.empty(len(ok)), np.empty(len(ok))
for k in range(len(ok)):
    slope[k], p[k] = sen_hr(D2[:, k], yrs)
q = by_q(p)
cls = np.where(q <= ALPHA, np.where(slope > 0, 2, 3), 1).astype(np.int8)


def put(v, fill=np.nan, dtype=float):
    a = np.full(tested.shape, fill, dtype=dtype)
    a[ok] = v
    return ("lat", "lon"), a.reshape(cls0.shape)


xr.Dataset({"mean_dry_months": put(D2.mean(0)), "slope_dry_months_per_decade": put(slope * 10),
            "q_by": put(q), "class_dry": put(cls, 0, np.int8)},
           coords={"lat": tc["lat"], "lon": tc["lon"]},
           attrs={"note": f"dry month = < {DRY_MM} mm/month; class 2 = MORE dry months, 3 = FEWER; exploratory."}
           ).to_netcdf(PROC / "dry_months_cells.nc")

aw = Wf[ok]
tot = Wf[tested].sum()
print(f"[A] số tháng khô/năm trung bình lưu vực (trọng số diện tích): {np.average(D2.mean(0), weights=aw):.2f}")
print(f"    TĂNG có bằng chứng: {(cls == 2).sum()} ô ({100 * aw[cls == 2].sum() / tot:.1f}% diện tích) | "
      f"GIẢM có bằng chứng: {(cls == 3).sum()} ô ({100 * aw[cls == 3].sum() / tot:.1f}%)")
main_dec = cls0.reshape(-1)[ok] == 3
if main_dec.sum():
    print(f"    Trong {main_dec.sum()} ô mưa-năm giảm có bằng chứng: {(cls[main_dec] == 2).sum()} ô cũng có thêm tháng khô "
          f"({100 * (cls[main_dec] == 2).mean():.1f}%)")
rows = []
for nm, f in REGIONS.items():
    m = f(latf[ok])
    s = np.average(D2[:, m], axis=1, weights=aw[m])
    sl, pp = sen_hr(s, yrs)
    rows.append(dict(region=nm, mean_dry_months=s.mean(), sen_dry_months_per_decade=sl * 10, p_hr=pp))
tabA = pd.DataFrame(rows)
tabA.to_csv(PROC / "regions_dry_months.csv", index=False)
pd.set_option("display.width", 200, "display.float_format", "{:.3g}".format)
print(tabA.to_string(index=False))

# ================= B) độ dốc mở rộng theo năm kết thúc =================
rows = []
for nm, f in REGIONS.items():
    m = f(latf[ok])
    s = np.average(Y2[:, ok][:, m], axis=1, weights=aw[m])
    for end in range(2015, 2025):
        k = int(end - yrs[0] + 1)
        sl, pp = sen_hr(s[:k], yrs[:k])
        rows.append(dict(region=nm, end_year=end, n_years=k, sen_mm_yr=sl, p_hr=pp))
tabB = pd.DataFrame(rows)
tabB.to_csv(PROC / "expanding_window.csv", index=False)
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for ax, (nm, g) in zip(axs, tabB.groupby("region", sort=False)):
    ax.axhline(0, color="0.5", lw=0.8)
    ax.plot(g["end_year"], g["sen_mm_yr"], color="0.3", lw=1)
    sig = g["p_hr"] < ALPHA
    ax.scatter(g["end_year"][sig], g["sen_mm_yr"][sig], color="#1f4e9c", zorder=3, label="p < 0.05 (uncorrected)")
    ax.scatter(g["end_year"][~sig], g["sen_mm_yr"][~sig], facecolor="white", edgecolor="0.3", zorder=3)
    ax.set_title(nm, fontsize=10)
    ax.set_xlabel("End year (series starts 2001)")
axs[0].set_ylabel("Sen slope (mm/yr)")
axs[0].legend(fontsize=7, loc="lower left")
fig.suptitle("Sen slope as the series is extended: sensitivity to the final years (overlapping windows, descriptive only)", fontsize=10)
fig.tight_layout()
fig.savefig(PROC / "expanding_slope.png", dpi=160)
print("\n[B] độ dốc Sen mở rộng (mm/năm), toàn lưu vực:")
print(tabB[tabB.region == "toan_luu_vuc"][["end_year", "sen_mm_yr", "p_hr"]].to_string(index=False))

# ================= C) ENSO hay TNA chi phối =================
dp = PROC / "drivers_cells.nc"
if dp.exists():
    dr = xr.open_dataset(dp)
    bn, bt, r2cv = (dr[k].values for k in ("beta_nino34_mm_per_sd", "beta_tna_mm_per_sd", "r2_cv_detrended"))
    dom = np.zeros(bn.shape, np.int8)
    good = np.isfinite(r2cv) & (r2cv > 0)
    dom[good & (np.abs(bn) >= np.abs(bt))] = 1
    dom[good & (np.abs(bn) < np.abs(bt))] = 2
    xr.Dataset({"dominant_driver": (("lat", "lon"), dom)}, coords={"lat": dr["lat"], "lon": dr["lon"]},
               attrs={"note": "1 = Nino3.4 larger |beta|, 2 = TNA larger |beta|, 0 = none/not predictive (CV R2 <= 0). Descriptive."}
               ).to_netcdf(PROC / "dominant_driver.nc")
    t = Wf[tested].sum()
    print("\n[C] diện tích (trong lưu vực): ENSO chi phối %.1f%% | TNA chi phối %.1f%% | không dự báo được (R2cv<=0) %.1f%%" % tuple(
        100 * Wf[tested & (dom.reshape(-1) == v)].sum() / t for v in (1, 2, 0)))
else:
    print("\n[C] bỏ qua: chưa có drivers_cells.nc (chạy 10_drivers.py trước).")
print("\nĐã lưu: dry_months_cells.nc, regions_dry_months.csv, expanding_window.csv, expanding_slope.png, dominant_driver.nc")