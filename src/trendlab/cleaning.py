"""Time selection, gap handling and yearly aggregation of gridded data."""


def clip_time(da, start, end):
    """Keep time steps between start and end (inclusive), e.g. "2002", "2005"."""
    return da.sel(time=slice(start, end))


def select_months(da, months):
    """Keep only the given calendar months (e.g. [6, 7, 8]).

    Seasons that cross the new year (e.g. 12, 1, 2) are NOT supported by
    `yearly`, which groups by calendar year.
    """
    return da.sel(time=da["time"].dt.month.isin(list(months)))


def mask_sparse_cells(da, min_valid_frac=0.8):
    """Set a cell to NaN everywhere if too many of its time steps are NaN."""
    frac = da.notnull().mean("time")
    return da.where(frac >= min_valid_frac)


def yearly(da, steps_per_year, how="mean", min_valid_frac=0.8):
    """Aggregate to one value per calendar year.

    steps_per_year: how many time steps a complete year contains in `da`
        (12 for monthly data; 3 if you kept three months per year).
    how="mean": a year needs at least min_valid_frac * steps_per_year valid
        steps, otherwise NaN.
    how="sum": a year needs ALL steps valid, otherwise NaN, because summing
        with gaps biases the total downward.
    """
    if how not in ("mean", "sum"):
        raise ValueError("how must be 'mean' or 'sum'")
    groups = da.resample(time="YS")
    values = groups.mean() if how == "mean" else groups.sum()
    n_valid = da.notnull().resample(time="YS").sum()
    need = steps_per_year if how == "sum" else min_valid_frac * steps_per_year

    out = values.where(n_valid >= need).transpose(*da.dims)
    out.attrs = dict(da.attrs)
    out.attrs["aggregation"] = (
        f"yearly {how}, needs >= {need:g} valid steps of {steps_per_year}"
    )
    return out
