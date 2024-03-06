import logging

import xarray as xr

LOGGER = logging.getLogger(__name__)

REQUEST = {
    "dataset": "reanalysis-era5-single-levels",
    "product_type": ["reanalysis"],
    "variable": ["2m_temperature"],
    "date": ["2022-01-01/2022-01-05"],
    "time": ["00:00", "12:00"],
}


def test_open_dataset() -> None:
    res = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore

    assert isinstance(res, xr.Dataset)
    assert set(res.dims) == {"time", "latitude", "longitude"}
    LOGGER.info(res)


def test_compare_chunked_no_chunked() -> None:
    ds1 = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da1 = ds1.data_vars["t2m"]
    res1 = da1.load()

    ds2 = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 2}, chunks={})  # type: ignore
    da2 = ds2.data_vars["t2m"]
    res2 = da2.load()

    ds3 = xr.open_dataset(REQUEST, engine="ecmwf", chunks={})  # type: ignore
    da3 = ds3.data_vars["t2m"]
    res3 = da3.load()

    assert (res3 - res2).shape == res3.shape
    assert (res3 == res2).all()

    assert (res3 - res1).shape == res3.shape
    assert (res3 == res1).all()


def test_cds_era5_single_time() -> None:
    ds1 = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da1 = ds1.data_vars["t2m"]
    res1 = da1.sel(time="2022-01-02T12:00").load()

    assert isinstance(res1, xr.DataArray)

    ds2 = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da2 = ds2.data_vars["t2m"]
    res2 = da2.sel(time="2022-01-02T12:00").load()

    assert isinstance(res2, xr.DataArray)

    ds3 = xr.open_dataset(REQUEST, engine="ecmwf", chunks={})  # type: ignore
    da3 = ds3.data_vars["t2m"]
    res3 = da3.sel(time="2022-01-02T12:00").load()

    assert (res3 - res1).shape == res3.shape
    assert (res3 == res1).all()

    assert (res3 - res2).shape == res3.shape
    assert (res3 == res2).all()


def test_cds_era5_small_slice_time() -> None:
    ds1 = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da1 = ds1.data_vars["t2m"]
    res1 = da1.sel(time="2022-01-02").load()

    assert isinstance(res1, xr.DataArray)

    ds2 = xr.open_dataset(REQUEST, engine="ecmwf", chunks={})  # type: ignore
    da2 = ds2.data_vars["t2m"]
    res2 = da2.sel(time="2022-01-02").load()

    assert (res2 - res1).shape == res2.shape
    assert (res2 == res1).all()


def test_cds_era5_empty_slice_time() -> None:
    ds = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da = ds.data_vars["t2m"]

    res = da.sel(time=slice("2022-01-07", "2022-01-08")).compute()
    assert isinstance(res, xr.DataArray)
    assert res["time"].size == 0


def test_cds_era5_big_slice_time() -> None:
    ds = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da = ds.data_vars["t2m"]

    res = da.sel(time=slice("2022-01-02", "2022-01-03")).mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1


def test_cds_era5_small_slice_time_longitute() -> None:
    ds = xr.open_dataset(REQUEST, engine="ecmwf", chunks={})  # type: ignore
    da = ds.data_vars["t2m"]

    res = da.sel(time="2022-01-02", longitude=0.25).mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1
