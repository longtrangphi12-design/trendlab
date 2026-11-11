# Standard data format

Every gridded dataset is an `xarray.DataArray` with:

- dims in this order: `("time", "lat", "lon")`
- `time`: datetime64, sorted
- `lat`: degrees north, ascending, within -90..90
- `lon`: degrees east, ascending, within -180..180
- `attrs["units"]`: required, any string (the library never assumes a unit)
- missing values: `NaN`
