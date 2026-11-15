"""Kiểm tra đơn vị, độ phủ thời gian, dữ liệu thiếu, độ hợp lý của mưa."""
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
p = xr.open_dataset(ROOT / "data/interim/imerg_amazon.nc")["precipitation"]
m = xr.open_dataset(ROOT / "data/interim/amazon_mask.nc")
mask, w = m["mask"].astype(bool), m["weight"]

# 1) độ phủ thời gian
t = p["time"].to_index()
n = len(t)
print("Số tháng:", n, "| từ", t[0], "đến", t[-1])
assert t.is_unique and t.is_monotonic_increasing, "thời gian trùng hoặc sai thứ tự"
full = (n == 288)
print("Bản ĐỦ 24 năm" if full else "Bản THỬ (chưa đủ 288 tháng), bỏ qua kiểm tra theo năm")

# 2) đơn vị gốc
print("Đơn vị gốc:", p.attrs.get("units_original"))   # kỳ vọng mm/hr

# 3) mm/giờ -> mm/tháng (nhân số giờ thực của từng tháng)
hours = p["time"].dt.days_in_month * 24
monthly_mm = p * hours

# 4) dữ liệu thiếu trong lưu vực
nan_in_basin = p.isnull().where(mask).mean()   # chỉ tính các ô TRONG lưu vực
print("Tỷ lệ NaN trong lưu vực: %.4f%%" % (100 * float(nan_in_basin)))

# 5) trung bình lưu vực theo tháng (có trọng số diện tích)
basin_month = (monthly_mm * w).sum(("lat", "lon"))
print(basin_month.to_series().round(0).to_string())
assert 50 < float(basin_month.mean()) < 500, "mưa tháng ngoài khoảng hợp lý: kiểm tra đơn vị / trọng số"

# 6) kiểm tra theo năm và chu kỳ mùa: chỉ khi đủ dữ liệu
if full:
    annual_mm = monthly_mm.groupby("time.year").sum("time", skipna=False)
    basin_annual = (annual_mm * w).sum(("lat", "lon"))
    print(basin_annual.to_series().round(0).to_string())
    print("Trung bình: %.0f mm/năm (kỳ vọng ~2000-2500)" % float(basin_annual.mean()))
    assert 1500 < float(basin_annual.mean()) < 3500, "kiểm tra lại đơn vị / trọng số"
    season = basin_month.groupby("time.month").mean()
    print(season.to_series().round(0).to_string())
print("XONG: các kiểm tra đạt")