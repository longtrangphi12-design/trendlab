"""Việc 9: chia mức độ nhẹ/vừa/mạnh cho các ô CÓ bằng chứng -> data/processed/magnitude.nc

Ngưỡng = tam phân vị của |% thay đổi mỗi thập kỷ| trong các ô GIẢM có bằng chứng,
áp dụng cùng một thang cho cả hai chiều. Đây là mức TƯƠNG ĐỐI trong lưu vực này,
không phải ngưỡng tuyệt đối.
"""
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"

tc = xr.open_dataset(PROC / "trend_cells.nc")
cls = tc["evidence_class"].values
pct = np.abs(tc["slope_pct_per_decade"].values)
dec, inc = cls == 3, cls == 2
assert dec.sum() >= 30, "quá ít ô giảm để chia tam phân vị"
t1, t2 = np.nanpercentile(pct[dec], [100 / 3, 200 / 3])

mag = np.zeros(cls.shape, dtype=np.int8)  # 0 = không áp dụng (ô không có bằng chứng)
for lo, hi, k in ((-np.inf, t1, 1), (t1, t2, 2), (t2, np.inf, 3)):
        mag[dec & (pct >= lo) & (pct < hi)] = k    # chỉ ô GIẢM có bằng chứng; ô tăng giữ 0

print(f"Ngưỡng: nhẹ < {t1:.2f}%/thập kỷ <= vừa < {t2:.2f}%/thập kỷ <= mạnh")
for name, m in (("giảm", dec), ("tăng", inc)):
    print(name, {k: int(((mag == k) & m).sum()) for k in (1, 2, 3)})

out = xr.Dataset({"magnitude_class": (("lat", "lon"), mag)},
                 coords={"lat": tc["lat"], "lon": tc["lon"]})
out.attrs.update({
    "threshold_mild_moderate_pct_per_decade": float(t1),
    "threshold_moderate_strong_pct_per_decade": float(t2),
    "note": "Relative scale from tertiles of decreasing cells with evidence; 0 = not applicable (includes the few increasing cells, shown as one class from evidence_class == 2).",
})
out.to_netcdf(PROC / "magnitude.nc")
print("Đã lưu:", PROC / "magnitude.nc")