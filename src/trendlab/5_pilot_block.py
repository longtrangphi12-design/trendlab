"""Việc 3: chạy thử cả quy trình trên khối 10x10 ô (IMERG 0,1° -> khoảng 1°x1°).

Chạy:  python src/trendlab/05_pilot_block.py
Tùy chọn: python src/trendlab/05_pilot_block.py <chỉ_số_vĩ_độ> <chỉ_số_kinh_độ>
  (góc dưới-trái của khối, tính theo chỉ số ô; mặc định: khối nằm trọn trong
   lưu vực và gần tâm lưu vực nhất)
Đầu ra: data/processed/pilot_block.csv và pilot_block.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from trendlab.autocorr import lag1_autocorr
from trendlab.multitest import benjamini_hochberg
from trendlab.trend import mann_kendall, sen_slope

ROOT = Path(__file__).resolve().parents[2]
INT, PROC = ROOT / "data/interim", ROOT / "data/processed"
B = 10        # cạnh khối (ô)
ALPHA = 0.05


# ====================== 3 phép kiểm định bổ sung ======================
def hamed_rao(x, alpha=0.05):
    """Mann-Kendall có hiệu chỉnh phương sai Hamed-Rao (1998).
    1) trừ xu hướng Sen, lấy HẠNG của phần dư; 2) tính tự tương quan rho_k của hạng;
    3) chỉ giữ độ trễ có |rho_k| > 1,96/sqrt(n); 4) nhân Var(S) với
       1 + 2/(n(n-1)(n-2)) * sum (n-k)(n-k-1)(n-k-2) rho_k.
    factor > 1: MK thường quá tự tin. Nếu factor <= 0 (tự tương quan âm rất mạnh)
    thì quay về MK thường và fallback=True để ô không bị rơi khỏi phép đếm m của BY."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    base = mann_kendall(x)
    if n < 4 or not np.isfinite(base["var_s"]) or base["var_s"] == 0:
        return {**base, "factor": np.nan, "fallback": True}

    t = np.arange(1, n + 1, dtype=float)
    ranks = stats.rankdata(x - sen_slope(x, t) * t)
    d = ranks - ranks.mean()
    denom = (d * d).sum()
    bound = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)

    total = 0.0
    for k in range(1, n):
        rho = (d[:-k] * d[k:]).sum() / denom
        if abs(rho) > bound:
            total += (n - k) * (n - k - 1) * (n - k - 2) * rho
    factor = 1 + 2 / (n * (n - 1) * (n - 2)) * total
    if factor <= 0:
        return {**base, "factor": float(factor), "fallback": True}

    var_s = base["var_s"] * factor
    s = base["s"]
    z = (s - 1) / np.sqrt(var_s) if s > 0 else ((s + 1) / np.sqrt(var_s) if s < 0 else 0.0)
    return {"s": s, "var_s": float(var_s), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z))), "n": n,
            "factor": float(factor), "fallback": False}


def pettitt(x):
    """Pettitt (1979): một điểm gãy đột ngột. U_t = sum_{i<=t} sum_{j>t} sign(x_j - x_i),
    K = max|U_t|, p ~ 2 exp(-6K^2/(n^3+n^2)) (xấp xỉ, tối đa 1).
    change_index = vị trí (từ 0) của điểm CUỐI CÙNG trước khi gãy."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return {"k": np.nan, "change_index": -1, "p": np.nan, "n": n}
    sgn = np.sign(x[None, :] - x[:, None])
    u = np.array([sgn[: t + 1, t + 1:].sum() for t in range(n - 1)])
    k = float(np.abs(u).max())
    p = min(1.0, float(2 * np.exp(-6 * k ** 2 / (n ** 3 + n ** 2))))
    return {"k": k, "change_index": int(np.abs(u).argmax()), "p": p, "n": n}


def benjamini_yekutieli(p):
    """BY = BH nhân c(m) = sum_{i=1..m} 1/i, tối đa 1. Đúng với mọi kiểu phụ thuộc
    giữa các ô (BH cần độc lập hoặc phụ thuộc dương), đổi lại thận trọng hơn.
    p = NaN bị bỏ qua, không tính vào m."""
    p = np.asarray(p, dtype=float)
    m = int(np.isfinite(p).sum())
    if m == 0:
        return np.full(p.shape, np.nan)
    c_m = (1.0 / np.arange(1, m + 1)).sum()
    return np.minimum(benjamini_hochberg(p) * c_m, 1.0)


# ====================== nạp dữ liệu ======================
annual = xr.open_dataset(PROC / "annual.nc")["annual"]      # mm/năm, đã áp mặt nạ
monthly = xr.open_dataset(PROC / "monthly.nc")["monthly"]   # mm/tháng
mask = xr.open_dataset(INT / "amazon_mask.nc")["mask"].astype(bool)

# ---------- 1) chọn khối 10x10 nằm trọn trong lưu vực ----------
if len(sys.argv) == 3:
    i0, j0 = int(sys.argv[1]), int(sys.argv[2])
else:
    inside = np.lib.stride_tricks.sliding_window_view(mask.values, (B, B)).all(axis=(-1, -2))
    cand = np.argwhere(inside)
    if len(cand) == 0:
        raise SystemExit("Không có khối 10x10 nào nằm trọn trong lưu vực")
    ii, jj = np.nonzero(mask.values)
    ci, cj = ii.mean(), jj.mean()                            # tâm lưu vực (theo chỉ số)
    d = (cand[:, 0] + B / 2 - ci) ** 2 + (cand[:, 1] + B / 2 - cj) ** 2
    i0, j0 = cand[d.argmin()]
blk = dict(lat=slice(i0, i0 + B), lon=slice(j0, j0 + B))
a = annual.isel(blk)
m = monthly.isel(blk)
print(f"Khối: lat {float(a.lat[0]):.2f}..{float(a.lat[-1]):.2f}, "
      f"lon {float(a.lon[0]):.2f}..{float(a.lon[-1]):.2f} (chỉ số {i0},{j0})")
assert a.shape[1:] == (B, B), "khối không đủ 10x10"

# ---------- 2) kiểm tra hợp lý TRƯỚC khi chạy thống kê ----------
n_year = a.sizes["year"]
print(f"Số năm: {n_year} ({int(a.year[0])}..{int(a.year[-1])})")
assert n_year == 24, "kỳ vọng 24 năm 2001-2024"
assert bool(mask.isel(blk).all()), "khối có ô ngoài lưu vực"
print("Số giá trị NaN trong khối (năm x ô):", int(a.isnull().sum()))

# (a) tính lại mm/năm từ mm/tháng bằng đường khác: phải khớp với annual.nc
re_annual = m.groupby("time.year").sum("time", skipna=False)
assert np.allclose(re_annual.values, a.values, equal_nan=True), "annual.nc không khớp monthly.nc"

# (b) mm/năm phải cỡ nghìn; tránh nhầm mm/giờ (~0,2) hoặc mm/tháng (~180)
mean_mm = float(a.mean())
print(f"Mưa năm trung bình của khối: {mean_mm:.0f} mm/năm")
assert 300 < mean_mm < 6000, "ngoài khoảng hợp lý: kiểm tra đơn vị"

# (c) chu kỳ mùa của khối (12 giá trị, mm/tháng); cộng 12 tháng ≈ mưa năm
season = m.mean(("lat", "lon")).groupby("time.month").mean()
print("Chu kỳ mùa (mm/tháng):", " ".join(f"{v:.0f}" for v in season.values))
assert abs(float(season.sum()) - mean_mm) < 0.02 * mean_mm, "tổng 12 tháng lệch mưa năm"

# ---------- 3) thống kê từng ô ----------
t = a["year"].values
rows = []
for i in range(B):
    for j in range(B):
        x = a.values[:, i, j]
        mk, hr, pt = mann_kendall(x), hamed_rao(x), pettitt(x)
        rows.append(dict(
            lat=float(a.lat[i]), lon=float(a.lon[j]), i=i, j=j,
            mean_mm=np.nanmean(x),
            sen_mm_per_yr=sen_slope(x, t),
            sen_pct_per_decade=100 * 10 * sen_slope(x, t) / np.nanmean(x),
            p_mk=mk["p"], p_hr=hr["p"], hr_factor=hr["factor"], hr_fallback=hr["fallback"],
            r1=lag1_autocorr(x),
            pettitt_p=pt["p"],
            pettitt_year=int(t[pt["change_index"] + 1]) if pt["change_index"] >= 0 else np.nan,
        ))
df = pd.DataFrame(rows)

# điều chỉnh đa kiểm định trên 100 ô của khối (m = 100, KHÔNG phải cả lưu vực)
df["q_bh_mk"] = benjamini_hochberg(df["p_mk"].values)
df["q_by_hr"] = benjamini_yekutieli(df["p_hr"].values)
df["sig_raw_mk"] = df["p_mk"] <= ALPHA
df["sig_by_hr"] = df["q_by_hr"] <= ALPHA
df.to_csv(PROC / "pilot_block.csv", index=False)

# ---------- 4) tóm tắt ----------
print("\n=== TÓM TẮT 100 Ô ===")
print(f"MK thô p<=0,05          : {int(df.sig_raw_mk.sum())} ô")
print(f"MK + BH   q<=0,05       : {int((df.q_bh_mk <= ALPHA).sum())} ô")
print(f"Hamed-Rao thô p<=0,05   : {int((df.p_hr <= ALPHA).sum())} ô")
print(f"Hamed-Rao + BY q<=0,05  : {int(df.sig_by_hr.sum())} ô   <- chuẩn dự kiến cho bản đồ")
print(f"Hệ số hiệu chỉnh HR (n/n*): nhỏ nhất {df.hr_factor.min():.2f}, "
      f"trung vị {df.hr_factor.median():.2f}, lớn nhất {df.hr_factor.max():.2f} "
      "(>1: MK thường quá tự tin; <1: quá thận trọng)")
print(f"Ô HR phải quay về MK thường (hệ số <= 0): {int(df.hr_fallback.sum())}")
print(f"Tự tương quan lag-1 r1  : trung bình {df.r1.mean():.2f}, trung vị {df.r1.median():.2f}")
print(f"Sen slope               : trung vị {df.sen_mm_per_yr.median():.1f} mm/năm "
      f"(khoảng {df.sen_mm_per_yr.min():.1f} .. {df.sen_mm_per_yr.max():.1f})")
print(f"Pettitt p<=0,05         : {int((df.pettitt_p <= ALPHA).sum())} ô; "
      f"năm gãy phổ biến nhất: {df.pettitt_year.mode().iloc[0]:.0f}")
same = 100 * max((df.sen_mm_per_yr > 0).mean(), (df.sen_mm_per_yr < 0).mean())
print(f"Ô có cùng dấu Sen       : {same:.0f}% (ô gần nhau cùng dấu là bình thường: "
      "các ô KHÔNG độc lập về không gian)")

# ---------- 5) hình kiểm tra ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
sl = df.pivot(index="lat", columns="lon", values="sen_mm_per_yr")
lim = np.nanmax(np.abs(sl.values))
im = ax[0].pcolormesh(sl.columns, sl.index, sl.values, cmap="RdBu", vmin=-lim, vmax=lim, shading="auto")
sg = df[df.sig_by_hr]
ax[0].scatter(sg.lon, sg.lat, s=14, c="k", label="HR + BY q<=0,05")
ax[0].legend(loc="upper right", fontsize=8)
fig.colorbar(im, ax=ax[0], label="Sen's slope (mm/năm)")
ax[0].set_title("Độ dốc Sen theo ô")
ax[1].plot(t, a.mean(("lat", "lon")), "o-")
ax[1].set_title(f"Mưa năm trung bình khối ({mean_mm:.0f} mm/năm)")
ax[1].set_xlabel("năm"); ax[1].set_ylabel("mm/năm")
ax[2].bar(range(1, 13), season.values)
ax[2].set_title("Chu kỳ mùa của khối"); ax[2].set_xlabel("tháng"); ax[2].set_ylabel("mm/tháng")
fig.tight_layout()
fig.savefig(PROC / "pilot_block.png", dpi=120)
print("\nĐã lưu:", PROC / "pilot_block.csv", "và", PROC / "pilot_block.png")
