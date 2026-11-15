"""Tải IMERG tháng + MERRA-2 TS, cắt theo khung, lưu NetCDF gọn.
Chạy lại được: các tháng đã tải sẽ được bỏ qua."""
import os
from pathlib import Path

import earthaccess
import xarray as xr

# ---- CẤU HÌNH (đổi vùng ở đây, không sửa chỗ khác) ----
START, END = "2001-01-01", "2024-12-31"   # bản thử 3 tháng; chạy đủ thì đổi thành "2001-01-01", "2024-12-31"
AMAZON_BOX = dict(west=-82.0, south=-21.0, east=-43.0, north=7.0)
BOXES = {
    "nino34": dict(west=-170.0, south=-5.0, east=-120.0, north=5.0),
    "tna": dict(west=-57.5, south=5.5, east=-15.0, north=23.5),   # Đại Tây Dương nhiệt đới bắc
}

ROOT = Path(__file__).resolve().parents[2]   # thư mục gốc repo (script nằm ở src/trendlab/)
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim"
PARTS = OUT / "parts"
for d in (RAW, OUT, PARTS):
    d.mkdir(parents=True, exist_ok=True)


def login():
    """Dùng biến môi trường nếu có, không thì hỏi (mật khẩu HIỆN khi gõ)."""
    if not (os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD")):
        os.environ["EARTHDATA_USERNAME"] = input("Earthdata username: ").strip()
        os.environ["EARTHDATA_PASSWORD"] = input("Earthdata password (hiện khi gõ): ").strip()
    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise SystemExit("Đăng nhập thất bại: kiểm tra lại tên đăng nhập / mật khẩu")
    print("Đăng nhập OK")


def fetch(short_name, version, box, var, out_name, group=None):
    granules = earthaccess.search_data(
        short_name=short_name, version=version,
        temporal=(START, END),
        bounding_box=(box["west"], box["south"], box["east"], box["north"]),
    )
    print(f"{short_name}: {len(granules)} tệp")
    part_dir = PARTS / out_name
    part_dir.mkdir(exist_ok=True)

    pieces = []
    for i, g in enumerate(granules):
        part = part_dir / f"{i:03d}.nc"
        if part.exists():                      # đã tải rồi -> bỏ qua
            pieces.append(xr.open_dataarray(part).load())
            continue
        path = earthaccess.download([g], str(RAW))[0]
        ds = xr.open_dataset(path, group=group) if group else xr.open_dataset(path)
        da = ds[var]
        if "lat" in da.dims and "lon" in da.dims:
            da = da.transpose("time", "lat", "lon")   # IMERG V07: (time, lon, lat) -> (time, lat, lon)
        da = da.sel(lon=slice(box["west"], box["east"]),
                    lat=slice(box["south"], box["north"])).load()
        da.attrs["units_original"] = da.attrs.get("units", "")
        da.to_netcdf(part)
        pieces.append(da)
        ds.close()
        os.remove(str(path))
        print(f"  [{i + 1}/{len(granules)}] xong")

    out = xr.concat(pieces, dim="time").sortby("time")
    out.attrs["units_original"] = pieces[0].attrs.get("units_original", "")
    out.to_dataset(name=var).to_netcdf(OUT / f"{out_name}.nc")
    print("đã lưu", out_name, dict(out.sizes), "| đơn vị gốc:", out.attrs["units_original"])


if __name__ == "__main__":
    import sys
    login()
    only = sys.argv[1:]          # ví dụ: python src/trendlab/01_download.py tna
    if not only:
        fetch("GPM_3IMERGM", "07", AMAZON_BOX, "precipitation", "imerg_amazon", group="Grid")
    for name, box in BOXES.items():
        if only and name not in only:
            continue
        fetch("M2TMNXSLV", "5.12.4", box, "TS", f"ts_{name}")