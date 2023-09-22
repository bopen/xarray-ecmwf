import xarray as xr


def test_open_dataset() -> None:
    request = {
        "dataset": "reanalysis-era5-land",
        "variable": ["2m_temperature"],
        "date": ["2000-01-01/2000-01-02"],
    }

    res = xr.open_dataset(request, engine="ecmwf")  # type: ignore

    assert isinstance(res, xr.Dataset)
    assert set(res.dims) == {"time", "lat", "lon"}
