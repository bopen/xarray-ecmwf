import logging
from typing import Any

import attrs
import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from . import client_protocol

LOGGER = logging.getLogger(__name__)

SUPPORTED_DATASETS = {"reanalysis-era5-single-levels", "reanalysis-era5-land"}


def build_chunks_header_requests(
    dim: str,
    request: dict[str, Any],
    request_chunks: dict[str, int],
    dtype: str = "int32",
) -> tuple[np.typing.NDArray[np.datetime64], int, list[tuple[int, dict[str, Any]]]]:
    request_chunks[dim]
    chunk_requests = []
    istart = 0
    while istart < len(request[dim]):
        indices = range(istart, istart + request_chunks[dim])
        values = [request[dim][k] for k in indices]
        chunk_requests.append((istart, {dim: values}))
        istart += request_chunks[dim]
    coord = np.array(request[dim], dtype=dtype)
    return coord, request_chunks[dim], chunk_requests


def build_chunk_date_requests(
    request: dict[str, Any], request_chunks: dict[str, int]
) -> tuple[np.typing.NDArray[np.datetime64], int, list[tuple[int, dict[str, Any]]]]:
    assert len(request_chunks) <= 1, "split on more than one param not supported"
    assert set(request_chunks) <= {"day"}

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
            start, stop = date, min(date + timedelta_days, date_stop)
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
            except ValueError:
                break
            times.append(datetime)

    if len(chunk_requests) == 0:
        chunk_requests = [(0, {})]

    return np.array(times), len(request["time"]) * chunk_days, chunk_requests


def build_chunk_ymd_requests(
    request: dict[str, Any], request_chunks: dict[str, int]
) -> tuple[np.typing.NDArray[np.datetime64], int, list[tuple[int, dict[str, Any]]]]:
    assert len(request_chunks) <= 1, "split on more than one param not supported"
    assert set(request_chunks) < {"month", "day"}

    times: list[np.datetime64] = []
    chunk_requests = []
    for year in request["year"]:
        assert len(year) == 4
        for month in request["month"]:
            assert len(month) == 2
            if "month" in request_chunks:
                if request_chunks["month"] != 1:
                    raise ValueError("split on month values != 1 not supported")
                chunk_requests.append((len(times), {"year": year, "month": month}))
            for day in request["day"]:
                assert len(day) == 2
                for time in request["time"]:
                    assert len(time) == 5
                    try:
                        datetime = np.datetime64(f"{year}-{month}-{day}T{time}", "ns")
                    except ValueError:
                        break
                    times.append(datetime)
                else:
                    if "day" in request_chunks:
                        if request_chunks["day"] != 1:
                            raise ValueError("split on day values != 1 not supported")
                        chunk_requests.append(
                            (len(times), {"year": year, "month": month, "day": day})
                        )

    if len(chunk_requests) == 0:
        chunk_requests = [(0, {})]

    return np.array(times), len(request["time"]), chunk_requests


def build_chunk_requests(
    request: dict[str, Any], request_chunks: dict[str, int]
) -> tuple[np.typing.NDArray[np.datetime64], int, list[tuple[int, dict[str, Any]]]]:
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


@attrs.define
class CdsapiRequestClient:
    client_kwargs: dict[str, Any] = {"quiet": True, "retry_max": 1}

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        request = request.copy()
        dataset = request.pop("dataset")
        if dataset not in SUPPORTED_DATASETS:
            LOGGER.warning(f"{dataset=} not supported")
        client = cdsapi.Client(**self.client_kwargs)
        return client.retrieve(dataset, request | {"format": "grib"})

    def get_filename(self, result: Any) -> str:
        return result.location.split("/")[-1]  # type: ignore

    def download(self, result: Any, target: str | None = None) -> str:
        return result.download(target)  # type: ignore


@attrs.define(slots=False)
class CdsapiRequestChunker:
    request: dict[str, Any]
    request_chunks: dict[str, Any]

    def compute_request_coords(self) -> dict[str, Any]:
        # XXX:
        #   add compute of time coordinate
        #   save chunk requests in instance with reference to index
        #   take from build_chunk_requests
        #   only then:
        #   - extend to number
        #   - extend to step (more tricky because it is a Timedelta)
        #   - low priority: extend to levelInHPa

        self.chunks = {}
        self.chunk_requests = {}
        coords = {}

        time, time_chunk, time_chunk_requests = build_chunk_requests(
            self.request, self.request_chunks
        )
        self.chunks["time"] = time_chunk
        self.chunk_requests["time"] = time_chunk_requests
        coords["time"] =  xr.IndexVariable("time", time, {}),  # type: ignore

        if "number" in self.request:
            number, number_chunk, number_chunk_request = build_chunks_header_requests(
                "number",
                self.request,
                self.request_chunks,
                dtype="int32"
            )
            self.chunks["number"] = number_chunk
            self.chunk_requests["number"] = number_chunk_request
            coords["number"] = xr.IndexVariable("number", number, {})

        if "step" in self.request:
            step, step_chunk, step_chunk_request = build_chunks_header_requests(
                "step",
                self.request,
                self.request_chunks,
                dtype="int32"
            )
            self.chunks["step"] = step_chunk
            self.chunk_requests["step"] = step_chunk_request
            coords["step"] = xr.IndexVariable("step", step * np.timedelta64(1, "h"), {})

        return coords

    def get_coords_attrs_and_dtype(
        self, dataset_cacher: client_protocol.DatasetCacherProtocol
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
        coords = self.compute_request_coords()
        with dataset_cacher.retrieve(self.request) as sample_ds:
            da = list(sample_ds.data_vars.values())[0]
            coords["lat"] = da.lat
            coords["lon"] = da.lon
            return coords, sample_ds.attrs, da.attrs, da.dtype

    def get_variables(self) -> list[str]:
        if "variable" in self.request:
            return list(self.request["variable"])
        elif "param" in self.request:
            return list(self.request["param"])
        raise ValueError(f"'variable' parameter not found in {list(self.request)}")
