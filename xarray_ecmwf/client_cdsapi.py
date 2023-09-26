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

        if isinstance(self.request.get("date"), list) or isinstance(
            self.request.get("year"), list
        ):
            time, time_chunk, time_chunk_requests = client_common.build_chunk_requests(
                self.request, self.request_chunks
            )
            self.chunks["time"] = time_chunk
            self.chunk_requests["time"] = time_chunk_requests
            coords["time"] = xr.IndexVariable(  # type: ignore
                "time", time, {}
            )

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
            coords["isobaricInhPa"] = xr.IndexVariable("isobaricInhPa", level, {"units": "hPa"})  # type: ignore

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

        return coords

    def get_coords_attrs_and_dtype(
        self, dataset_cacher: client_common.DatasetCacherProtocol
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
        coords = self.compute_request_coords()
        with dataset_cacher.retrieve(self.request) as sample_ds:
            da = list(sample_ds.data_vars.values())[0]
            for name in da.coords:
                if name not in coords and name in da.dims:
                    coords[name] = da.coords[name]
            return coords, sample_ds.attrs, da.attrs, da.dtype

    def get_variables(self) -> list[str]:
        if "variable" in self.request:
            return list(self.request["variable"])
        elif "param" in self.request:
            return list(self.request["param"])
        raise ValueError(f"'variable' parameter not found in {list(self.request)}")
