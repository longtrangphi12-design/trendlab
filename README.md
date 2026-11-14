# trendlab

Historical trend detection and robustness checks for gridded environmental time series.
First application: annual rainfall over the **Amazon basin, 2001-2024**, built for the
NASA Space Apps Challenge 2026 "Earth System Trend Detective".

This is a **historical analysis**, not a forecasting system.

## What it does

- Per-cell Sen's slope and Mann-Kendall test, with the Hamed-Rao autocorrelation correction.
- False-discovery-rate control across tens of thousands of cells (Benjamini-Yekutieli as the main rule, Benjamini-Hochberg for comparison).
- Three-tier evidence labels: increase with evidence / decrease with evidence / not enough evidence.
- Robustness checks fixed in advance: drop-years scenarios, block bootstrap, 3-year window scans, hydrological year, and the minimum trend a 24-year series could detect.
- Exploratory seasonality indices (Oliver's PCI, annual harmonic amplitude and phase).

## Main findings (2001-2024, GPM IMERG Final V07 monthly)

- No evidence of a basin-wide annual trend: basin-mean Sen slope -7.0 mm/yr, p = 0.22, block-bootstrap 95% interval [-14.3, 1.3].
- 10.7% of basin area (south-central Amazon and a strip near the Andes) shows a decreasing signal under the main rule, but most of it disappears when 2022-2024 are dropped (14.8% of significant cells keep their class), while dropping early or middle years keeps about 60-64%.
- South of 3°S the area-mean series shows weak decreases (about -10 to -13 mm/yr). After correcting for testing three sub-regions they do not reach 5%.
- No robust change in seasonality (PCI, amplitude, phase).
- With 24 years, the median minimum detectable slope is about 22 mm/yr per cell, so "not enough evidence" does not mean "no change".

Full details and limits: `docs/methods.md`. Data and citations: `docs/sources.md`. AI use: `AI_USAGE.md`.

## Repository layout
src/trendlab/ package modules (trend, multitest, autocorr, cleaning, report, simulate, ...)
trend_extra.py: Hamed-Rao, Pettitt, Benjamini-Yekutieli
01_download.py ... 08_robustness.py: Amazon pipeline scripts
tests/ unit tests (pytest)
docs/ methods.md, data_format.md, sources.md
data/ NOT in Git (raw, interim, processed, external)


## Reproduce

Python 3.14.2 was used (versions in `docs/sources.md`).

```bash
pip install -r requirements.txt
pip install -e .
pytest
```

**Data you need**

1. A free NASA Earthdata account. Script 01 asks for the username and password when it runs (the password is visible while typing; never commit credentials).
2. HydroBASINS South America level 3 (`hybas_sa_lev03_v1c.shp` with its companion files) from https://www.hydrosheds.org, saved by hand in `data/external/`. Check its licence terms.

**Run from the repository root, in this order**

```bash
python src/trendlab/01_download.py          # IMERG monthly 2001-2024 (earthaccess), cropped to the Amazon box
python src/trendlab/02_build_mask.py        # prints the largest basins; pick the Amazon MAIN_BAS (about 6 million km2)
python src/trendlab/02_build_mask.py <MAIN_BAS>   # builds the mask and area weights
python src/trendlab/03_check_data.py        # units, time coverage, missing data, plausibility
python src/trendlab/04_make_annual.py       # data/processed/annual.nc and monthly.nc
python src/trendlab/06_run_basin.py         # per-cell trend and evidence classes -> trend_cells.nc
python src/trendlab/07_season_metrics.py    # exploratory seasonality layers -> season_cells.nc
python src/trendlab/08_robustness.py        # robustness and detectability -> robustness_cells.nc
```

`5_pilot_block.py` is an optional debugging run on a 10x10 block of cells.

Processed result files: [ĐIỀN: link tới GitHub Release / Drive nếu bạn chia sẻ file kết quả, hoặc "regenerate with the commands above"].

## Interface

[ĐIỀN: ghi trạng thái giao diện Streamlit, ví dụ "Streamlit explorer in `app/` reads the files in `data/processed/`", hoặc "planned, not included in this submission"].

## Limitations

24 years of data give low statistical power, and the main decreasing signal is sensitive to the last three years. Neighbouring cells are strongly dependent, so counts of significant cells overstate independent evidence. IMERG is a satellite estimate calibrated with gauges, and no station validation was done. No causes are attributed. See `docs/methods.md`, sections 9 and 10.

## Licence

Code: MIT (see `LICENSE`). Data keep the licences of their providers (see `docs/sources.md`).