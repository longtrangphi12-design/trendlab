# MarqLens – Amazon rainfall trend explorer (Streamlit app)

Historical rainfall trends, Amazon River basin, 2001–2024 (not a forecast).

## Run
```
pip install -r requirements.txt
streamlit run app/streamlit_app.py      # run from the repo root
```
Result files go in `data/processed/` (trend_cells.nc, magnitude.nc, robustness_cells.nc,
annual.nc, drivers_cells.nc, dominant_driver.nc, dry_months_cells.nc, adjusted_drop_cells.nc,
summary_numbers.csv, basin_annual.csv, regions_*.csv, expanding_window.csv,
adjusted_robustness.csv, drivers_indices.csv). Another folder: set `TRENDLAB_DATA_DIR`.

## Layout
- `app/streamlit_app.py` – page config, theme CSS, navigation (4 pages)
- `app/lib/data.py` – every file read (`@st.cache_data`), drop-years codes, `cell_record()`
- `app/lib/charts.py` – Plotly figures (pure functions)
- `app/lib/text.py` – all fixed wording and colours
- `app/lib/game.py` – Discovery missions, hot/cold hints, journal, claims (pure functions)
- `app/views/` – map_explorer, discovery, lab, regions, exploratory, about_methods
  (named `views`, not `pages`: a folder called `pages/` makes Streamlit ignore st.navigation on direct links)

The app only reads result files; it never recomputes trends, p, q or evidence.
