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


def test_compare_chunked_no_chunked() -> None:
    ds1 = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da1 = ds1.data_vars["2m_temperature"]

    ds2 = xr.open_dataset(REQUEST, engine="ecmwf", chunks={})  # type: ignore
    da2 = ds2.data_vars["2m_temperature"]

    assert da1.equals(da2)


def test_cds_era5_single_time() -> None:
    ds = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da = ds.data_vars["2m_temperature"]

    res = da.sel(time="2022-07-16T00:00").mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1


def test_cds_era5_small_slice_time() -> None:
    ds = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da = ds.data_vars["2m_temperature"]

    res = da.sel(time="2022-07-01").mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1


def test_cds_era5_empty_slice_time() -> None:
    ds = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da = ds.data_vars["2m_temperature"]

    res = da.sel(time="2022-07-02").compute()
    assert isinstance(res, xr.DataArray)
    assert res["time"].size == 0


def test_cds_era5_big_slice_time() -> None:
    ds = xr.open_dataset(REQUEST, engine="ecmwf", request_chunks={"day": 1}, chunks={})  # type: ignore
    da = ds.data_vars["2m_temperature"]

    res = da.sel(time=slice("2022-07-01", "2022-07-16")).mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1


def test_cds_era5_small_slice_time_longitute() -> None:
    ds = xr.open_dataset(REQUEST, engine="ecmwf", chunks={})  # type: ignore
    da = ds.data_vars["2m_temperature"]

    res = da.sel(time="2022-07-01", longitude=0.25).mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1
