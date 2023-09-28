import bisect
import logging
from typing import Any

import attrs
import cdsapi
import numpy as np
import xarray as xr

from . import client_common

LOGGER = logging.getLogger(__name__)


@attrs.define
class CdsapiRequestClient:
    client_kwargs: dict[str, Any] = {"quiet": True, "retry_max": 1}

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        request = request.copy()
        dataset = request.pop("dataset")
        client = cdsapi.Client(**self.client_kwargs)
        return client.retrieve(dataset, request | {"format": "grib"})

    def get_filename(self, result: Any) -> str:
        return result.location.split("/")[-1]  # type: ignore

    def download(self, result: Any, target: str | None = None) -> str:
        return result.download(target)  # type: ignore


SUPPORTED_REQUEST_DIMENSIONS = [
    "date",
    "year",
    "month",
    "day",
    "time",
    "number",
    "leadtime_hour",
    "pressure_level",
]


@attrs.define(slots=False)
class CdsapiRequestChunker:
    request: dict[str, Any]
    request_chunks: dict[str, Any]

    def get_request_dimensions(self) -> dict[str, list[Any]]:
        request_dimensions: dict[str, list[Any]] = {}
        for dim in SUPPORTED_REQUEST_DIMENSIONS:
            if dim not in self.request:
                continue
            if not isinstance(self.request[dim], list):
                continue
            request_dimensions[dim] = self.request[dim]
        return request_dimensions

    def get_chunks(self) -> dict[str, int]:
        return self.chunks

    def compute_request_coords(self) -> dict[str, Any]:
        self.chunks = {}
        self.chunk_requests = {}
        coords = {}

        if isinstance(self.request.get("date"), list) or isinstance(
            self.request.get("year"), list
        ):
            time, time_chunk, time_chunk_requests = client_common.build_chunk_requests(
                self.request, self.request_chunks
            )
            self.chunks["time"] = time_chunk
            self.chunk_requests["time"] = time_chunk_requests
            coords["time"] = xr.IndexVariable("time", time, {})  # type: ignore

        if isinstance(self.request.get("leadtime_hour"), list):
            (
                step,
                step_chunk,
                step_chunk_request,
            ) = client_common.build_chunks_header_requests(
                "leadtime_hour", self.request, self.request_chunks, dtype="int32"
            )
            self.chunks["step"] = step_chunk
            self.chunk_requests["step"] = step_chunk_request
            coords["step"] = xr.IndexVariable(  # type: ignore
                "step", step * np.timedelta64(1, "h"), {}
            )

        if isinstance(self.request.get("pressure_level"), list):
            (
                level,
                level_chunk,
                level_chunk_request,
            ) = client_common.build_chunks_header_requests(
                "pressure_level", self.request, self.request_chunks, dtype="int32"
            )
            self.chunks["isobaricInhPa"] = level_chunk
            self.chunk_requests["isobaricInhPa"] = level_chunk_request
            coords["isobaricInhPa"] = xr.IndexVariable(  # type: ignore
                "isobaricInhPa", level, {"units": "hPa"}
            )

        # `number` is last because some CDS datasets do not allow to select
        # ensemble members in the request and always return all of them.
        # In this case we set the dimension in `get_coords_attrs_and_dtype`
        if isinstance(self.request.get("number"), list):
            (
                number,
                number_chunk,
                number_chunk_request,
            ) = client_common.build_chunks_header_requests(
                "number", self.request, self.request_chunks, dtype="int32"
            )
            self.chunks["number"] = number_chunk
            self.chunk_requests["number"] = number_chunk_request
            coords["number"] = xr.IndexVariable("number", number, {})  # type: ignore

        return coords

    def get_coords_attrs_and_dtype(
        self, dataset_cacher: client_common.DatasetCacherProtocol
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
        coords = self.compute_request_coords()
        self.request_dims = list(coords)
        self.file_dims = []
        with dataset_cacher.retrieve(self.request) as sample_ds:
            da = list(sample_ds.data_vars.values())[0]
            for name in da.coords:
                if name not in coords and name in da.dims:
                    assert isinstance(name, str)
                    coords[name] = da.coords[name]
                    self.file_dims.append(name)
            self.dims = list(coords)
            return coords, sample_ds.attrs, da.attrs, da.dtype

    def get_variables(self) -> list[str]:
        if "variable" in self.request:
            return list(self.request["variable"])
        elif "param" in self.request:
            return list(self.request["param"])
        raise ValueError(f"'variable' parameter not found in {list(self.request)}")

    def build_requests(self, chunk_requests=None) -> dict[str, Any]:
        request = self.request.copy()
        request.update(**chunk_requests)
        return request

    def find_start(self, dim: str, key: int) -> int:
        chunk_requests = self.chunk_requests[dim]
        start_chunks = [chunk[0] for chunk in chunk_requests]
        # to check
        return bisect.bisect(start_chunks, key) - 1

    def get_chunk_values(
        self,
        key: tuple[int | slice, ...],
        dataset_cacher: ...,
    ) -> np.typing.ArrayLike:
        # XXX: only support `key` that access exactly one chunk
        assert len(key) == len(self.dims)
        request_keys = key[: len(self.request_dims)]

        chunks_requests = {}
        selection = dict(zip(self.dims, key))
        for dim, request_key in zip(self.request_dims, request_keys):
            if isinstance(request_key, slice):
                # XXX: check that the slice is exactly one chunk for everything except lat lon
                if request_key.start is None:
                    index = 0
                else:
                    index = self.find_start(dim, request_key.start)
                chunks_requests.update(**self.chunk_requests[dim][index][1])
                # compute relative index
                start = request_key.start - self.chunk_requests[dim][index][0]
                stop = request_key.stop - self.chunk_requests[dim][index][0]
                selection[dim] = slice(start, stop, request_key.step)
            elif isinstance(request_key, int):
                index = self.find_start(dim, request_key)
                chunks_requests.update(**self.chunk_requests[dim][index][1])
                selection[dim] = request_key - self.chunk_requests[dim][index][0]
            else:
                raise ValueError("key type {type(request_key)} not supported")

        field_request = self.build_requests(chunks_requests)
        with dataset_cacher.retrieve(field_request) as ds:
            da = list(ds.data_vars.values())[0]
        # XXX: check that the dimensions are in the correct order or rollaxis
        out = da.isel(**selection)
        return out.values
