import calendar
from typing import Any, ContextManager, Protocol

import numpy as np
import pandas as pd
import xarray as xr


class RequestClientProtocol(Protocol):
    def __init__(self, client_kwargs: dict[str, Any]) -> None:
        ...

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        ...

    def get_filename(self, result: Any) -> str:
        ...

    def download(self, result: Any, target: str | None = None) -> str:
        ...


class DatasetCacherProtocol(Protocol):
    def retrieve(
        self,
        request: dict[str, Any],
        override_cache_file: bool | None = None,
        force_valid_time_as_time: bool = False,
    ) -> ContextManager[xr.Dataset]:
        ...


class RequestChunkerProtocol(Protocol):
    def __init__(self, request: dict[str, Any], request_chunks: dict[str, Any]) -> None:
        ...

    def get_request_dimensions(self) -> dict[str, list[Any]]:
        ...

    def get_coords_attrs_and_dtype(
        self, dataset_cacher: DatasetCacherProtocol
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
        ...

    def get_variables(self) -> list[str]:
        ...

    def get_chunks(self) -> dict[str, Any]:
        ...

    def get_chunk_values(
        self, key: tuple[int | slice, ...], dataset_cacher: DatasetCacherProtocol
    ) -> np.typing.ArrayLike:
        ...


def build_chunks_header_requests(
    dim: str,
    request: dict[str, Any],
    request_chunks: dict[str, int],
    dtype: str = "int32",
) -> tuple[np.typing.NDArray[Any], int, list[tuple[int, dict[str, Any]]]]:
    chunk_requests = []
    request_chunks_dim = request_chunks.get(dim, len(request[dim]))
    istart = 0
    while istart < len(request[dim]):
        istop = min(istart + request_chunks_dim, len(request[dim]))
        indices = range(istart, istop)
        values = [request[dim][k] for k in indices]
        chunk_requests.append((istart, {dim: values}))
        istart += request_chunks_dim
    coord = np.array(request[dim], dtype=dtype)
    return coord, request_chunks_dim, chunk_requests


def build_chunk_date_requests(
    request: dict[str, Any], request_chunks: dict[str, int]
) -> tuple[np.typing.NDArray[np.datetime64], int, list[tuple[int, dict[str, Any]]]]:
    assert set(request_chunks).intersection(["month", "day", "year"]) <= {"day"}

    date_start_str, date_stop_str = request["date"][0].split("/")
    date_stop = pd.to_datetime(date_stop_str)
    chunk_days = request_chunks.get("day", 1)
    timedelta_days = pd.Timedelta(f"{chunk_days}D")

    times: list[np.datetime64] = []
    chunk_requests: list[tuple[int, dict[str, Any]]] = []
    start = None

    for date in pd.date_range(date_start_str, date_stop_str):
        if "day" in request_chunks and (
            start is None or date - timedelta_days == start
        ):
            start, stop = date, min(
                date + timedelta_days - pd.Timedelta(1, "d"), date_stop
            )
            chunk_requests.append(
                (
                    len(times),
                    {"date": f"{start.date()}/{stop.date()}"},
                )
            )

        for time in request["time"]:
            assert len(time) == 5
            try:
                datetime = np.datetime64(f"{date.date()}T{time}", "ns")
            except ValueError as ex:
                print(ex)
                break
            times.append(datetime)

    if len(chunk_requests) == 0:
        chunk_requests = [(0, {})]

    return np.array(times), len(request["time"]) * chunk_days, chunk_requests


def generate_ymd_coordinates(
    request: dict[str, Any]
) -> np.typing.NDArray[np.datetime64]:
    times: list[np.datetime64] = []
    for year in request["year"]:
        assert len(year) == 4
        for month in request["month"]:
            assert len(month) == 2
            ndays = calendar.monthrange(int(year), int(month))[1]
            for day in request["day"]:
                assert len(day) == 2
                if int(day) > ndays:
                    break
                for time in request["time"]:
                    assert len(time) == 5
                    datetime = np.datetime64(f"{year}-{month}-{day}T{time}", "ns")
                    times.append(datetime)
    return np.array(times)


def build_chunk_ymd_month_requests(
    request: dict[str, Any], request_chunks: dict[str, int]
) -> tuple[
    np.typing.NDArray[np.datetime64],
    int | tuple[int, ...],
    list[tuple[int, dict[str, Any]]],
]:
    if request_chunks["month"] != 1:
        raise ValueError("split on day values != 1 not supported")

    datetimes: list[np.datetime64] = []
    chunk_requests = []
    chunks = []
    for year in request["year"]:
        assert len(year) == 4
        for month in request["month"]:
            chunk = 0
            assert len(month) == 2
            ndays = calendar.monthrange(int(year), int(month))[1]
            start = len(datetimes)
            days = []
            for day in request["day"]:
                assert len(day) == 2
                if int(day) > ndays:
                    break
                days.append(day)
                for time in request["time"]:
                    assert len(time) == 5
                    chunk += 1
                    datetime = np.datetime64(f"{year}-{month}-{day}T{time}", "ns")
                    datetimes.append(datetime)
            chunks.append(chunk)

            chunk_requests.append((start, {"year": year, "month": month, "day": days}))
    return np.array(datetimes), tuple(chunks), chunk_requests


def build_chunk_ymd_requests(
    request: dict[str, Any], request_chunks: dict[str, int]
) -> tuple[
    np.typing.NDArray[np.datetime64],
    int | tuple[int, ...],
    list[tuple[int, dict[str, Any]]],
]:
    time_chunk_keys = list(set(request_chunks).intersection(["month", "day", "year"]))
    if len(time_chunk_keys) == 0:
        return generate_ymd_coordinates(request), len(request), [(0, {})]

    assert len(time_chunk_keys) == 1
    time_chunk_key = time_chunk_keys[0]
    assert time_chunk_key in set(["day", "month"])

    if time_chunk_key == "month":
        out = build_chunk_ymd_month_requests(request, request_chunks)
    elif time_chunk_key == "day":
        out = build_chunk_ymd_day_requests(request, request_chunks)
    return out


def build_chunk_ymd_day_requests(
    request: dict[str, Any], request_chunks: dict[str, int]
) -> tuple[np.typing.NDArray[np.datetime64], int, list[tuple[int, dict[str, Any]]]]:
    if request_chunks["day"] != 1:
        raise ValueError("split on day values != 1 not supported")

    times: list[np.datetime64] = []
    chunk_requests = []
    for year in request["year"]:
        assert len(year) == 4
        for month in request["month"]:
            assert len(month) == 2
            for day in request["day"]:
                start = len(times)
                assert len(day) == 2
                for time in request["time"]:
                    assert len(time) == 5
                    try:
                        datetime = np.datetime64(f"{year}-{month}-{day}T{time}", "ns")
                    except ValueError:
                        break
                    times.append(datetime)
                else:
                    if request_chunks["day"] != 1:
                        raise ValueError("split on day values != 1 not supported")
                    chunk_requests.append(
                        (start, {"year": year, "month": month, "day": day})
                    )

    return np.array(times), len(request["time"]), chunk_requests


def build_time_chunk_requests(
    request: dict[str, Any], request_chunks: dict[str, int]
) -> tuple[
    np.typing.NDArray[np.datetime64],
    int | tuple[int, ...],
    list[tuple[int, dict[str, Any]]],
]:
    if "year" in request:
        time, time_chunk, time_chunk_requests = build_chunk_ymd_requests(
            request, request_chunks
        )
    elif "date" in request:
        time, time_chunk, time_chunk_requests = build_chunk_date_requests(
            request, request_chunks
        )
    else:
        raise ValueError("request must contain either 'year' or 'date'")

    return time, time_chunk, time_chunk_requests
