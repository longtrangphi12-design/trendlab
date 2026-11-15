# Sources

## Data

| Role | Product | Version | Provider / access | Notes |
|---|---|---|---|---|
| Rainfall (main data) | GPM IMERG Final Run, Monthly, `GPM_3IMERGM`, variable `precipitation`, group `Grid`, 0.1° | V07 | NASA GES DISC, downloaded with `earthaccess` (`src/trendlab/01_download.py`) | NASA open data. Period used: 2001-01 to 2024-12 (288 months). Original unit mm/hr, converted to mm/month by multiplying by the real number of hours of each month, then summed per calendar year. Satellite-gauge merged estimate. |
| Basin boundary (additional data) | HydroBASINS, South America, level 3, file `hybas_sa_lev03_v1c.shp`, main basin selected with `MAIN_BAS` | v1c | HydroSHEDS (https://www.hydrosheds.org) | Not a NASA product, added only to build the Amazon mask. Check the HydroSHEDS licence terms at the link above before the final submission. Not committed to Git (`data/` is ignored). |
| Niño 3.4 skin temperature | MERRA-2 `M2TMNXSLV`, variable `TS`, box 170°W-120°W, 5°S-5°N | 5.12.4 | NASA GES DISC, `earthaccess` | Downloaded by `01_download.py` but NOT used in any reported result (the explanatory regression branch was not implemented). [ĐIỀN: xoá dòng này nếu không có file data/interim/ts_nino34.nc] |
MERRA-2 monthly single-level diagnostics (M2TMNXSLV), variable TS (skin temperature)	5.12.4	NASA GMAO, via GES DISC	Niño 3.4 and TNA indices

Challenge-provided datasets: [ĐIỀN: tên bộ dữ liệu nếu cuộc thi có phát, hoặc "none; all rainfall data are NASA open data from GES DISC"].

## Citations

- Huffman, G.J., Stocker, E.F., Bolvin, D.T., Nelkin, E.J., Tan, J. (2023). GPM IMERG Final Precipitation L3 1 month 0.1 degree x 0.1 degree V07. NASA GES DISC. doi:10.5067/GPM/IMERGM/3B-MONTH/07 (check the DOI on the product page).
- Lehner, B., Grill, G. (2013). Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems. Hydrological Processes, 27(15), 2171-2186. doi:10.1002/hyp.9740

## Software

| Library | Version | Link |
|---|---|---|
| Python | 3.14.2 | https://www.python.org |
| numpy | 2.5.4 | https://numpy.org |
| scipy | 1.18.1 | https://scipy.org |
| pandas | 3.1.1 | https://pandas.pydata.org |
| xarray | 2026.9.0 | https://xarray.dev |
| netCDF4 | 1.7.5.1.post1 | https://unidata.github.io/netcdf4-python |
| matplotlib | 3.11.2 | https://matplotlib.org |
| geopandas | 1.2.0 | https://geopandas.org |
| regionmask | 0.13.0 | https://regionmask.readthedocs.io |
| earthaccess | 0.19.0 | https://github.com/nsidc/earthaccess |
| Streamlit (interface) | [ĐIỀN sau khi cài] | https://streamlit.io |