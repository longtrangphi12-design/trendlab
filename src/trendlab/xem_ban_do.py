import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xarray as xr

d = xr.open_dataset("data/processed/trend_cells.nc")
print(d.attrs["sensitivity"])

# mã lớp: -1 ngoài lưu vực, 0 không đủ dữ liệu, 1 chưa đủ bằng chứng, 2 tăng (đỏ), 3 giảm (xanh)
cmap = ListedColormap(["white", "#C3C8CE", "#E5E7EB", "#D9302C", "#3182BD"])
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
for a, name in zip(ax, ["evidence_class", "class_drop_drought", "class_drop_2022_24"]):
    d[name].plot(ax=a, cmap=cmap, vmin=-1.5, vmax=3.5, add_colorbar=False)
    a.set_title(name)
fig.tight_layout()
fig.savefig("data/processed/evidence_maps.png", dpi=110)
print("đã lưu data/processed/evidence_maps.png")