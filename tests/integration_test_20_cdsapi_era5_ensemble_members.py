import logging

import xarray as xr

LOGGER = logging.getLogger(__name__)

REQUEST = {
    "dataset": "reanalysis-era5-single-levels",
    "product_type": ["ensemble_members"],
    "variable": ["2m_temperature"],
    "year": ["2022"],
    "month": ["01", "07"],
    "day": ["01", "16"],
    "time": ["00:00", "12:00"],
}


def test_open_dataset() -> None:
    res = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore

    assert isinstance(res, xr.Dataset)
    assert set(res.dims) == {"number", "time", "latitude", "longitude"}
    LOGGER.info(res)


def test_compare_chunked_no_chunked() -> None:
    ds1 = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore

    assert ds1.chunks["time"] == (2, 2, 2, 2)

    res1 = ds1.data_vars["2m_temperature"].load()

    ds2 = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"month": 1}, chunks={})  # type: ignore

    assert ds2.chunks["time"] == (4, 4)

    res2 = ds2.data_vars["2m_temperature"].load()

    ds0 = xr.open_dataset(REQUEST, engine="ecmwf", chunks={})  # type: ignore
    res0 = ds0.data_vars["2m_temperature"].load()

    assert (res0 - res1).shape == res0.shape
    assert (res0 == res1).all()

    assert (res0 - res2).shape == res0.shape
    assert (res0 == res2).all()


def test_cds_era5_single_time() -> None:
    ds1 = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    res1 = ds1.data_vars["2m_temperature"].sel(time="2022-07-16T00:00").load()
    assert isinstance(res1, xr.DataArray)

    ds0 = xr.open_dataset(REQUEST, engine="ecmwf", chunks={})  # type: ignore
    res0 = ds0.data_vars["2m_temperature"].sel(time="2022-07-16T00:00").load()

    assert (res0 - res1).shape == res0.shape
    assert (res0 == res1).all()


def test_cds_era5_small_slice_time() -> None:
    ds1 = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da1 = ds1.data_vars["2m_temperature"]
    res1 = da1.sel(time="2022-07-01").load()

    assert isinstance(res1, xr.DataArray)

    ds0 = xr.open_dataset(REQUEST, engine="ecmwf", chunks={})  # type: ignore
    res0 = ds0.data_vars["2m_temperature"].sel(time="2022-07-01").load()

    assert (res0 - res1).shape == res0.shape
    assert (res0 == res1).all()


def test_cds_era5_big_slice_time() -> None:
    ds = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da = ds.data_vars["2m_temperature"]

    res = da.sel(time=slice("2022-07-01", "2022-07-18")).mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1
