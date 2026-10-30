import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# CDS can return fewer time steps than requested for the first request of a
# dataset: ERA5-Land hourly has no 1950-01-01 00:00. xarray-ecmwf must return
# NaN for the missing step and the right value for every other step.
DATE_REQUEST = {
    "dataset": "dummy-dataset",
    "variable": ["dummy_variable"],
    "date": ["1950-01-01/1950-01-04"],
    "time": ["00:00", "01:00", "02:00"],
}
YMD_REQUEST = {
    "dataset": "dummy-dataset",
    "variable": ["dummy_variable"],
    "year": ["1950"],
    "month": ["01", "02"],
    "day": ["01", "02"],
    "time": ["00:00", "01:00", "02:00"],
}
# the first step of the fake dataset, like ERA5-Land
DATASET_START = pd.Timestamp("1950-01-01 01:00")
# the value of the first requested hour is missing
EXPECTED = np.array([np.nan, 1, 2, 24, 25, 26, 48, 49, 50, 72, 73, 74])


# the request itself is the result, downloaded as JSON
class MissingFirstStepClient:
    def __init__(self, client_kwargs: dict[str, Any]) -> None:
        pass

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        return request

    def get_filename(self, result: Any) -> str:
        return hashlib.md5(str(result).encode("utf-8")).hexdigest() + ".json"

    def download(self, result: Any, target: str | None = None) -> str:
        assert target is not None
        Path(target).write_text(json.dumps(result))
        return target


def as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


# one field per hour, the value is the hours since 1950-01-01 00:00
def open_missing_first_step(path: str, **kwargs: Any) -> xr.Dataset:
    request = json.loads(Path(path).read_text())
    if "date" in request:
        start, stop = as_list(request["date"])[0].split("/")
        days = pd.date_range(start, stop, freq="D")
    else:
        days = pd.DatetimeIndex(
            [
                f"{year}-{month}-{day}"
                for year in as_list(request["year"])
                for month in as_list(request["month"])
                for day in as_list(request["day"])
            ]
        )
    time = pd.DatetimeIndex(
        [f"{day:%Y-%m-%d} {hour}" for day in days for hour in request["time"]]
    )
    time = time[time >= DATASET_START]
    hours = (time - pd.Timestamp("1950-01-01")) / pd.Timedelta("1h")
    values = np.broadcast_to(hours.to_numpy()[:, None, None], (len(time), 2, 3))
    return xr.Dataset(
        {"dummy": (("time", "latitude", "longitude"), values)},
        coords={"time": time, "latitude": [10.0, 0.0], "longitude": [0.0, 1.0, 2.0]},
    )


def open_dummy_dataset(
    request: dict[str, Any], request_chunks: dict[str, int], cache_folder: Path
) -> xr.Dataset:
    return xr.open_dataset(
        request,  # type: ignore
        engine="ecmwf",
        request_chunks=request_chunks,
        request_client_class=MissingFirstStepClient,
        open_dataset=open_missing_first_step,
        cache_kwargs={"cache_folder": str(cache_folder)},
        chunks={},
    )


@pytest.mark.parametrize(
    "time_key",
    [
        slice(0, 6),  # the whole first chunk
        slice(1, 6),  # first chunk, starting after the missing step
        slice(0, 4),  # first chunk, stopping before its end
        4,  # a single step
    ],
)
def test_first_chunk_read_returns_requested_steps(
    tmp_path: Path, time_key: int | slice
) -> None:
    ds = open_dummy_dataset(DATE_REQUEST, {"day": 2}, tmp_path)

    res = ds["dummy"].isel(time=time_key, latitude=0, longitude=0).values

    assert np.array_equal(res, EXPECTED[time_key], equal_nan=True)


def test_shifted_read_returns_next_step_values(tmp_path: Path) -> None:
    ds = open_dummy_dataset(DATE_REQUEST, {"day": 2}, tmp_path)
    # dask turns the lazy shift into a read of the first chunk starting at index 1
    da = ds["dummy"].shift(time=-1).isel(time=slice(None, -1))

    res = da.isel(latitude=0, longitude=0).values

    assert np.array_equal(res, EXPECTED[1:], equal_nan=True)


def test_month_chunked_read_returns_requested_steps(tmp_path: Path) -> None:
    # month chunks have a tuple of chunk sizes
    ds = open_dummy_dataset(YMD_REQUEST, {"month": 1}, tmp_path)

    res = ds["dummy"].isel(latitude=0, longitude=0).values

    # 1950-02-01 is 744 hours after 1950-01-01
    expected = [np.nan, 1, 2, 24, 25, 26, 744, 745, 746, 768, 769, 770]
    assert np.array_equal(res, expected, equal_nan=True)
