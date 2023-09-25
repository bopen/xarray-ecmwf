import xarray as xr

REQUEST = {
    "source": "ecmwf",
    "type": "pf",
    "param": ["msl"],
    "date": 0,
    "number": [1, 2],
    "step": [12, 24],
}


def test_open_dataset() -> None:
    res = xr.open_dataset(REQUEST, engine="ecmwf", client="ecmwf-opendata")  # type: ignore

    assert isinstance(res, xr.Dataset)
    assert set(res.dims) == {"realization", "leadtime", "lat", "lon"}


# def test_cds_era5_single_time() -> None:
#     ds = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore
#     da = ds.data_vars["2m_temperature"]

#     res = da.sel(time="2022-07-16T00:00").mean().compute()

#     assert isinstance(res, xr.DataArray)
#     assert res.size == 1


# def test_cds_era5_small_slice_time() -> None:
#     ds = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore
#     da = ds.data_vars["2m_temperature"]

#     res = da.sel(time="2022-07-02").mean().compute()

#     assert isinstance(res, xr.DataArray)
#     assert res.size == 1


# def test_cds_era5_big_slice_time() -> None:
#     ds = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore
#     da = ds.data_vars["2m_temperature"]

#     res = da.sel(time=slice("2022-07-02", "2022-07-03")).mean().compute()

#     assert isinstance(res, xr.DataArray)
#     assert res.size == 1
