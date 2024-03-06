import logging

import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)

REQUEST = {
    "dataset": "seasonal-original-single-levels",
    "originating_centre": "ecmwf",
    "system": "51",
    "variable": ["2m_temperature"],
    "year": ["2022", "2023"],
    "month": ["08", "09"],
    "day": ["01"],
    "time": ["00:00"],
    "leadtime_hour": ["36", "72"],
}


def test_open_dataset() -> None:
    res = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"leadtime_hour": 1, "day": 1},
        chunks={},
    )

    assert isinstance(res, xr.Dataset)
    assert set(res.dims) == {
        "time",
        "step",
        "number",
        "latitude",
        "longitude",
    }
    assert res.time.size == 4
    assert res.step.size == 2

    LOGGER.info(res)


def test_compare_chunked_no_chunked() -> None:
    ds1 = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"leadtime_hour": 1, "day": 1},
        chunks={},
    )

    assert ds1.chunks["time"] == (1, 1, 1, 1)

    res1 = ds1.data_vars["t2m"].load()

    ds2 = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"leadtime_hour": 1, "month": 1},
        chunks={},
    )

    assert ds2.chunks["time"] == (1, 1, 1, 1)

    res2 = ds2.data_vars["t2m"].load()

    ds0 = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        chunks={},
    )
    res0 = ds0.data_vars["t2m"].load()

    assert (res0 - res1).shape == res0.shape
    assert (res0 == res1).all()

    assert (res0 - res2).shape == res0.shape
    assert (res0 == res2).all()


def test_cds_seasonal_single_time() -> None:
    ds1 = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"leadtime_hour": 1, "day": 1},
        chunks={},
    )
    da1 = ds1.data_vars["t2m"]
    res1 = da1.sel(time="2023-08-01T00:00").load()

    assert isinstance(res1, xr.DataArray)

    ds0 = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        chunks={},
    )
    res0 = ds0.data_vars["t2m"].sel(time="2023-08-01T00:00").load()

    assert (res0 - res1).shape == res0.shape
    assert (res0 == res1).all()


def test_cds_seasonal_small_slice_time() -> None:
    ds1 = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"leadtime_hour": 1, "day": 1},
        chunks={},
    )
    da1 = ds1.data_vars["t2m"]
    res1 = da1.sel(time="2023-08-01").load()

    assert isinstance(res1, xr.DataArray)

    ds0 = xr.open_dataset(REQUEST, engine="ecmwf", chunks={})  # type: ignore
    res0 = ds0.data_vars["t2m"].sel(time="2023-08-01").load()

    assert (res0 - res1).shape == res0.shape
    assert (res0 == res1).all()


def test_cds_seasonal_small_slice_time_and_step() -> None:
    ds = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"leadtime_hour": 1, "day": 1},
        chunks={},
    )
    da = ds.data_vars["t2m"]
    res = da.sel(time="2023-08-01", step=np.timedelta64(36, "h")).mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1


def test_cds_seasonal_big_slice_time() -> None:
    ds = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_chunks={"leadtime_hour": 1, "day": 1},
        chunks={},
    )
    da = ds.data_vars["t2m"]

    res = da.sel(time=slice("2022-07-02", "2022-07-03")).mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1
