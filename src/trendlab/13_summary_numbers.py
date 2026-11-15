"""Việc 13: gom các con số tóm tắt vào MỘT file (data/processed/summary_numbers.csv: key,value) để giao diện và slide đọc từ file.
Chỉ ĐỌC các file đã có; không tính lại phân loại. Cần chạy sau 06, 10, 11 và 12."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from trendlab.trend_extra import hamed_rao   # giống dòng import trong 06_run_basin.py

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
out = {}

# --- chỉ số biển ---
idx = pd.read_csv(PROC / "drivers_indices.csv", index_col="year")
yrs = idx.index.values.astype(float)
t = yrs - yrs.mean()
Z = ((idx - idx.mean()) / idx.std(ddof=1)).values
T = np.column_stack([np.ones_like(t), t])
Zd = Z - T @ np.linalg.lstsq(T, Z, rcond=None)[0]
r, rd = np.corrcoef(Z.T)[0, 1], np.corrcoef(Zd.T)[0, 1]
out.update(corr_nino34_tna=r, corr_nino34_tna_detrended=rd, vif_detrended=1 / (1 - rd**2), vif_raw=1 / (1 - r**2))
for j, nm in enumerate(("nino34", "tna")):
    out[f"index_{nm}_sen_sd_per_yr"] = stats.theilslopes(Z[:, j], yrs)[0]
    out[f"index_{nm}_p_hr"] = hamed_rao(Z[:, j], 0.05)["p"]

# --- bản chính ---
tc = xr.open_dataset(PROC / "trend_cells.nc")
cls = tc["evidence_class"].values
W = tc["area_weight"].values
tested = cls >= 1
for nm, v in (("up", 2), ("down", 3), ("not_enough", 1)):
    out[f"main_cells_{nm}"] = int((cls == v).sum())
    out[f"main_area_share_{nm}"] = float(W[cls == v].sum() / W[tested].sum())

# --- nhánh hiệu chỉnh (10) ---
dr = xr.open_dataset(PROC / "drivers_cells.nc")
dec = cls == 3
for nm in ("joint", "naive"):
    c = dr[f"class_adj_{nm}"].values
    out[f"adj_{nm}_main_decreasing_kept"] = int(((c == 3) & dec).sum())
    out[f"adj_{nm}_main_decreasing_kept_share"] = float(((c == 3) & dec).sum() / dec.sum())
    out[f"adj_{nm}_new_decreasing"] = int(((c == 3) & ~dec).sum())
    out[f"adj_{nm}_new_increasing"] = int((c == 2).sum())
    out[f"adj_{nm}_decreasing_area_share"] = float(W[c == 3].sum() / W[tested].sum())
r2cv = dr["r2_cv_detrended"].values
out["r2_detrended_median"] = float(np.nanmedian(dr["r2_detrended"].values))
out["r2_adj_detrended_median"] = float(np.nanmedian(dr["r2_adj_detrended"].values))
out["r2_cv_median"] = float(np.nanmedian(r2cv))
out["r2_cv_positive_share_of_cells"] = float((r2cv[np.isfinite(r2cv)] > 0).mean())

# --- yếu tố chi phối (11) ---
dom = xr.open_dataset(PROC / "dominant_driver.nc")["dominant_driver"].values
for nm, v in (("none", 0), ("nino34", 1), ("tna", 2)):
    out[f"dominant_{nm}_area_share"] = float(W[tested & (dom == v)].sum() / W[tested].sum())

# --- bỏ 2022–2024 (12) ---
ad = xr.open_dataset(PROC / "adjusted_drop_cells.nc")
out["drop2022_24_main_decreasing_kept_share_raw"] = float(ad.attrs["kept_share_raw"])
out["drop2022_24_main_decreasing_kept_share_adj"] = float(ad.attrs["kept_share_adj"])
out["drop2022_24_main_decreasing_n"] = int(ad.attrs["n_main_decreasing"])
for nm in ("raw", "adj"):
    c = ad[f"class_{nm}_2001_2021"].values
    out[f"drop2022_24_{nm}_decreasing_area_share"] = float(W[c == 3].sum() / W[tested].sum())

pd.DataFrame({"key": list(out), "value": list(out.values())}).to_csv(PROC / "summary_numbers.csv", index=False)
for k, v in out.items():
    print(f"{k:55s} {v:.4g}" if isinstance(v, float) else f"{k:55s} {v}")
print("\nĐã lưu: summary_numbers.csv")