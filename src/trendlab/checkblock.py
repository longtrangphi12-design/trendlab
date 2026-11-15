import numpy as np, xarray as xr
from trendlab.trend import sen_slope

a = xr.open_dataset("data/processed/annual.nc")["annual"]
m = xr.open_dataset("data/processed/monthly.nc")["monthly"]
w = xr.open_dataset("data/interim/amazon_mask.nc")["weight"]

blk = dict(lat=slice(137, 147), lon=slice(169, 179))
b_ann = a.isel(blk).mean(("lat", "lon"))
basin = (a * w).sum(("lat", "lon"))          # trung bình có trọng số diện tích cả lưu vực

print("năm   khối   lưu vực")
for y in a.year.values:
    print(y, f"{float(b_ann.sel(year=y)):7.0f} {float(basin.sel(year=y)):7.0f}")
print("Sen khối   :", round(sen_slope(b_ann.values), 1), "mm/năm")
print("Sen lưu vực:", round(sen_slope(basin.values), 1), "mm/năm")

mb = m.isel(blk).mean(("lat", "lon"))
early = mb.sel(time=slice("2001", "2012")).groupby("time.month").mean()
late = mb.sel(time=slice("2013", "2024")).groupby("time.month").mean()
print("\nChênh lệch (2013-2024) trừ (2001-2012), mm/tháng, theo tháng 1..12:")
print(np.round((late - early).values, 0))
print("Số tháng:", m.sizes["time"], "| tháng trùng:", int(m.time.to_index().duplicated().sum()))