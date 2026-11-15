"""Tạo annual.nc (mm/năm) và monthly.nc (mm/tháng), đã áp mặt nạ, cho giao diện."""
from pathlib import Path
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
INT, PROC = ROOT / "data/interim", ROOT / "data/processed"
PROC.mkdir(parents=True, exist_ok=True)

p = xr.open_dataset(INT / "imerg_amazon.nc")["precipitation"]
mask = xr.open_dataset(INT / "amazon_mask.nc")["mask"].astype(bool)

hours = p["time"].dt.days_in_month * 24            # số giờ thực của từng tháng
monthly = (p * hours).where(mask).astype("float32")
monthly.name, monthly.attrs["units"] = "monthly", "mm/month"

annual = monthly.groupby("time.year").sum("time", skipna=False).astype("float32")
annual.name, annual.attrs["units"] = "annual", "mm/year"

monthly.to_netcdf(PROC / "monthly.nc")
annual.to_netcdf(PROC / "annual.nc")
print("monthly:", dict(monthly.sizes), "| annual:", dict(annual.sizes))
print("Trung bình lưu vực: %.0f mm/năm" % float(annual.mean()))   # kỳ vọng ~2173
