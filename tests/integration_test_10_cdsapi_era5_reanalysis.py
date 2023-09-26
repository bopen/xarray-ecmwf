import logging

import xarray as xr

LOGGER = logging.getLogger(__name__)

REQUEST = {
    "dataset": "reanalysis-era5-single-levels",
    "product_type": ["reanalysis"],
    "variable": ["2m_temperature"],
    "year": ["2022"],
    "month": ["01", "07"],
    "day": ["01", "16"],
    "time": ["00:00", "12:00"],
}


def test_open_dataset() -> None:
    res = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore

    assert isinstance(res, xr.Dataset)
    assert set(res.dims) == {"time", "latitude", "longitude"}
    LOGGER.info(res)


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
