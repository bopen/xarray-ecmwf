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


def build_chunk_date_requests(request: dict[str, Any], request_split: dict[str, int]):
    assert len(request_split) <= 1, "split on more than one param not supported"
    assert set(request_split) <= {"day"}

    date_start_str, date_stop_str = request["date"][0].split("/")
    date_stop = pd.to_datetime(date_stop_str)
    chunk_days = request_split.get("day", 1)
    timedelta_days = pd.Timedelta(f"{chunk_days}D")

    times: list[np.datetime64] = []
    chunk_requests = []
    start = None

    for date in pd.date_range(date_start_str, date_stop_str):
        if "day" in request_split and (start is None or date - timedelta_days == start):
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


def build_chunk_ymd_requests(request: dict[str, Any], request_split: dict[str, int]):
    assert len(request_split) <= 1, "split on more than one param not supported"
    assert set(request_split) < {"month", "day"}

    times: list[np.datetime64] = []
    chunk_requests = []
    for year in request["year"]:
        assert len(year) == 4
        for month in request["month"]:
            assert len(month) == 2
            if "month" in request_split:
                if request_split["month"] != 1:
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
                    if "day" in request_split:
                        if request_split["day"] != 1:
                            raise ValueError("split on day values != 1 not supported")
                        chunk_requests.append(
                            (len(times), {"year": year, "month": month, "day": day})
                        )

    if len(chunk_requests) == 0:
        chunk_requests = [(0, {})]

    return np.array(times), len(request["time"]), chunk_requests


def build_chunk_requests(request: dict[str, Any], request_split: dict[str, int]):
    if "year" in request:
        time, time_chunk, time_chunk_requests = build_chunk_ymd_requests(
            request, request_split
        )
    elif "date" in request:
        time, time_chunk, time_chunk_requests = build_chunk_date_requests(
            request, request_split
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

        time, time_chunk, time_chunk_requests = build_chunk_requests(
            self.request, self.request_chunks
        )
        self.time_chunks = time_chunk
        self.time_chunk_requests = time_chunk_requests

        coords = {
            "time": xr.IndexVariable("time", time, {}),
        }

        return coords

    def get_coords_attrs_and_dtype(
        self, dataset_cacher=client_protocol.DatasetCacherProtocol
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
