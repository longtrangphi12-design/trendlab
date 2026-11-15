"""Tạo mặt nạ lưu vực Amazon + trọng số diện tích trên lưới IMERG.
Bước 1: python src/trendlab/02_build_mask.py            -> in bảng lưu vực lớn nhất
Bước 2: python src/trendlab/02_build_mask.py <MAIN_BAS> -> tạo mặt nạ + ảnh kiểm tra
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import regionmask
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
SHP = ROOT / "data" / "external" / "hybas_sa_lev03_v1c.shp"
GRID = ROOT / "data" / "interim" / "imerg_amazon.nc"
OUT = ROOT / "data" / "interim" / "amazon_mask.nc"
PNG = ROOT / "data" / "interim" / "mask_check.png"

gdf = gpd.read_file(SHP)

if len(sys.argv) < 2:
    big = gdf.dissolve(by="MAIN_BAS")
    big["area_km2"] = big.to_crs("EPSG:6933").area / 1e6
    top = big.sort_values("area_km2", ascending=False)[["area_km2"]].head(10)
    print(top.round(0))
    raise SystemExit("\nChọn MAIN_BAS có diện tích ~6 triệu km² (Amazon), rồi chạy lại:\n"
                     "  python src/trendlab/02_build_mask.py <MAIN_BAS>")

main_bas = int(sys.argv[1])
basin = gdf[gdf["MAIN_BAS"] == main_bas].dissolve(by="MAIN_BAS")

da = xr.open_dataset(GRID)["precipitation"].isel(time=0)
inside = regionmask.mask_geopandas(basin, da["lon"], da["lat"]).notnull()

# trọng số diện tích cos(vĩ độ), chỉ trong lưu vực, tổng = 1
w = np.cos(np.deg2rad(da["lat"])) * xr.ones_like(da["lon"])
w = w.where(inside, 0.0)
w = w / w.sum()

# kiểm tra diện tích: Amazon cỡ 6.0-6.9 triệu km²
R = 6371.0
d = np.deg2rad(abs(float(da["lat"][1] - da["lat"][0])))
cell_km2 = (R * d) ** 2 * np.cos(np.deg2rad(da["lat"])) * xr.ones_like(da["lon"])
print("Diện tích mặt nạ: %.2f triệu km²" % (float(cell_km2.where(inside).sum()) / 1e6))
print("Số ô trong lưu vực:", int(inside.sum()))

xr.Dataset({"mask": inside.astype("int8"), "weight": w}).to_netcdf(OUT)

# ảnh kiểm tra bằng mắt
fig, ax = plt.subplots(figsize=(7, 6))
inside.astype(int).plot(ax=ax, cmap="Greens", add_colorbar=False)
basin.boundary.plot(ax=ax, color="k", linewidth=0.8)
ax.set_title("Amazon mask (green) + HydroBASINS boundary")
fig.savefig(PNG, dpi=120, bbox_inches="tight")
print("Đã lưu:", OUT, "và", PNG)