import logging

import xarray as xr

LOGGER = logging.getLogger(__name__)

REQUEST = {
    "dataset": "reanalysis-era5-pressure-levels",
    "product_type": ["ensemble_members"],
    "variable": ["temperature"],
    "year": ["2022"],
    "month": ["01", "07"],
    "day": ["01", "16"],
    "time": ["00:00", "12:00"],
    "pressure_level": ["1000", "500"],
}


def test_open_dataset() -> None:
    res = xr.open_dataset(REQUEST, engine="ecmwf")  # type: ignore

    assert isinstance(res, xr.Dataset)
    assert set(res.dims) == {
        "number",
        "isobaricInhPa",
        "time",
        "latitude",
        "longitude",
    }
    LOGGER.info(res)


def test_compare_chunked_no_chunked() -> None:
    ds1 = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"day": 1, "pressure_level": 1},
        chunks={},
    )
    res1 = ds1.data_vars["temperature"].load()

    ds2 = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        chunks={},
    )
    res2 = ds2.data_vars["temperature"].load()

    assert res1.equals(res2)


def test_cds_era5_single_time() -> None:
    ds = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"day": 1, "pressure_level": 1},
        chunks={},
    )
    da = ds.data_vars["temperature"]

    res = da.sel(time="2022-07-16T00:00")

    assert isinstance(res, xr.DataArray)


def test_cds_era5_small_slice_time() -> None:
    ds = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"day": 1, "pressure_level": 1},
        chunks={},
    )
    da = ds.data_vars["temperature"]

    res = da.sel(time="2022-07-02").mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1


def test_cds_era5_big_slice_time() -> None:
    ds = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"day": 1, "pressure_level": 1},
        chunks={},
    )
    da = ds.data_vars["temperature"]

    res = da.sel(time=slice("2022-07-02", "2022-07-03")).mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1
