import datetime
import logging

import xarray as xr

LOGGER = logging.getLogger(__name__)

REQUEST = {
    "source": "ecmwf",
    "type": "pf",
    "param": ["msl"],
    "date": -1,
    "step": ["12", "24", "48"],
    "time": ["0", "12"],
    "number": [1, 2],
}


def test_open_dataset() -> None:
    res = xr.open_dataset(REQUEST, engine="ecmwf", client="ecmwf-opendata")  # type: ignore

    assert isinstance(res, xr.Dataset)
    assert set(res.dims) == {"time", "step", "latitude", "longitude", "number"}
    assert res.time.size == 2
    assert res.step.size == 3

    LOGGER.info(res)


def test_compare_chunked_no_chunked() -> None:
    ds1 = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        client="ecmwf-opendata",
        request_chunks={"step": 1, "number": 1},
        chunks={},
    )
    res1 = ds1.data_vars["msl"].load()

    ds2 = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        client="ecmwf-opendata",
        chunks={},
    )
    res2 = ds2.data_vars["msl"].load()

    assert res1.equals(res2)


def test_cds_era5_single_time() -> None:
    ds = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        client="ecmwf-opendata",
        chunks={},
        request_chunks={"step": 1, "number": 1},
    )
    da = ds.data_vars["msl"]

    time = datetime.date.today() - datetime.timedelta(days=1)
    res = da.sel(time=f"{time}T00:00").compute()

    assert isinstance(res, xr.DataArray)
    assert set(res.dims) == {"step", "latitude", "longitude", "number"}

    res = da.sel(time=f"{time}T00:00").mean().compute()

    assert res.size == 1


def test_cds_era5_small_slice_time() -> None:
    ds = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        client="ecmwf-opendata",
        chunks={},
        request_chunks={"step": 1, "number": 1},
    )
    da = ds.data_vars["msl"]

    time = datetime.date.today() - datetime.timedelta(days=1)
    res = da.sel(time=f"{time}").compute()

    assert isinstance(res, xr.DataArray)
    assert res.time.size == 2

    res = da.sel(time=f"{time}").mean().compute()

    assert isinstance(res, xr.DataArray)
    assert res.size == 1


def test_cds_era5_small_step() -> None:
    ds = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        client="ecmwf-opendata",
        chunks={},
        request_chunks={"step": 1, "number": 1},
    )
    da = ds.data_vars["msl"]

    datetime.date.today() - datetime.timedelta(days=1)
    res = da.sel(step=datetime.timedelta(hours=12)).compute()

    assert isinstance(res, xr.DataArray)
    assert set(res.dims) == {"time", "latitude", "longitude", "number"}

    res = da.sel(step=datetime.timedelta(hours=12)).mean().compute()

    assert res.size == 1


def test_cds_era5_small_slice_step() -> None:
    ds = xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        client="ecmwf-opendata",
        chunks={},
        request_chunks={"step": 1, "number": 1},
    )
    da = ds.data_vars["msl"]

    datetime.date.today() - datetime.timedelta(days=1)

    step_slice = slice(datetime.timedelta(hours=12), datetime.timedelta(hours=24))

    res = da.sel(step=step_slice).compute()

    assert isinstance(res, xr.DataArray)
    assert set(res.dims) == {"time", "step", "latitude", "longitude", "number"}
    assert res.step.size == 2
