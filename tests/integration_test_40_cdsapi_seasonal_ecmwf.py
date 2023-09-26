import xarray as xr

REQUEST = {
    "dataset": "seasonal-original-single-levels",
    "originating_centre": "ecmwf",
    "system": "51",
    "variable": ["2m_temperature"],
    "year": ["2023"],
    "month": ["08", "09"],
    "day": ["01"],
    "time": ["00:00"],
    "leadtime_hour": ["36", "72"],
}


def test_open_dataset() -> None:
    res = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore

    assert isinstance(res, xr.Dataset)
    assert set(res.dims) == {
        "time",
        "number",
        "step",
        "latitude",
        "longitude",
    }


# def test_cds_seasonal_single_time() -> None:
#     ds = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore
#     da = ds.data_vars["2m_temperature"]

#     res = da.sel(time="2022-07-16T00:00").mean().compute()

#     assert isinstance(res, xr.DataArray)
#     assert res.size == 1


# def test_cds_seasonal_small_slice_time() -> None:
#     ds = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore
#     da = ds.data_vars["2m_temperature"]

#     res = da.sel(time="2022-07-02").mean().compute()

#     assert isinstance(res, xr.DataArray)
#     assert res.size == 1


# def test_cds_seasonal_big_slice_time() -> None:
#     ds = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore
#     da = ds.data_vars["2m_temperature"]

#     res = da.sel(time=slice("2022-07-02", "2022-07-03")).mean().compute()

#     assert isinstance(res, xr.DataArray)
#     assert res.size == 1