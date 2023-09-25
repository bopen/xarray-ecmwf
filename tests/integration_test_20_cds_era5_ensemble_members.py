import xarray as xr


def test_cds_era5() -> None:
    request = {
        "dataset": "reanalysis-era5-single-levels",
        "product_type": ["ensemble_members"],
        "number": ["1", "2"],
        "variable": ["2m_temperature"],
        "year": ["2022"],
        "month": ["01", "07"],
        "day": ["01", "16"],
        "time": ["00:00", "12:00"],
    }

    res = xr.open_dataset(request, engine="ecmwf")  # type: ignore

    assert isinstance(res, xr.Dataset)
    assert set(res.dims) == {"number", "time", "lat", "lon"}


def test_cds_era5_single_time() -> None:
    request = {
        "dataset": "reanalysis-era5-single-levels",
        "product_type": ["ensemble_members"],
        "number": ["1", "2"],
        "variable": ["2m_temperature"],
        "year": ["2022"],
        "month": ["01", "07"],
        "day": ["01", "16"],
        "time": ["00:00", "12:00"],
    }
    ds = xr.open_dataset(request, engine="ecmwf")  # type: ignore
    da = ds.data_vars["2m_temperature"]

    res = da.sel(time="2022-07-16T00:00").mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1


def test_cds_era5_small_slice_time() -> None:
    request = {
        "dataset": "reanalysis-era5-single-levels",
        "product_type": ["ensemble_members"],
        "number": ["1", "2"],
        "variable": ["2m_temperature"],
        "year": ["2022"],
        "month": ["01", "07"],
        "day": ["01", "16"],
        "time": ["00:00", "12:00"],
    }
    ds = xr.open_dataset(request, engine="ecmwf")  # type: ignore
    da = ds.data_vars["2m_temperature"]

    res = da.sel(time="2022-07-02").mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1


def test_cds_era5_big_slice_time() -> None:
    request = {
        "dataset": "reanalysis-era5-single-levels",
        "product_type": ["ensemble_members"],
        "number": ["1", "2"],
        "variable": ["2m_temperature"],
        "year": ["2022"],
        "month": ["01", "07"],
        "day": ["01", "16"],
        "time": ["00:00", "12:00"],
    }
    ds = xr.open_dataset(request, engine="ecmwf")  # type: ignore
    da = ds.data_vars["2m_temperature"]

    res = da.sel(time=slice("2022-07-02", "2022-07-03")).mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1
