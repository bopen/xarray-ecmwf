import bisect
import logging
from typing import Any

import attrs
import cdsapi
import numpy as np
import xarray as xr

from . import client_common

LOGGER = logging.getLogger(__name__)

COORDINATES_ORDER = ("time", "step", "isobaricInhPa", "number")


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
    "step",
    "pressure_level",
    "levelist",
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

    def get_chunks(self) -> dict[str, int | tuple[int, ...]]:
        return self.chunks

    def maybe_update_coords_and_chunk_info(
        self,
        request_coord_name: str,
        coord_name: str,
        indexer_kwargs: dict["str", Any] = {},
        dtype: str = "int32",
    ) -> None:
        if request_coord_name in self.request_chunks:
            if isinstance(self.request.get(request_coord_name), list):
                (
                    coord,
                    coord_chunk,
                    coord_chunk_request,
                ) = client_common.build_chunks_header_requests(
                    request_coord_name, self.request, self.request_chunks, dtype=dtype
                )
                self.chunks[coord_name] = coord_chunk
                self.chunk_requests[coord_name] = coord_chunk_request
                if coord_name in ("step"):
                    self.chunked_coords[coord_name] = xr.IndexVariable(  # type: ignore
                        "step", coord * np.timedelta64(1, "h"), **indexer_kwargs
                    )
                else:
                    self.chunked_coords[coord_name] = xr.IndexVariable(  # type: ignore
                        coord_name, coord, {}
                    )
        return

    def compute_chunked_request_coords(self) -> dict[str, Any]:
        self.chunks = {}
        self.chunk_requests = {}
        self.chunked_coords = {}

        if "time" in self.request:
            if (
                isinstance(self.request.get("date"), list)
                or isinstance(self.request.get("year"), list)
                or isinstance(self.request.get("month"), list)
                or isinstance(self.request.get("day"), list)
            ):
                (
                    time,
                    time_chunk,
                    time_chunk_requests,
                ) = client_common.build_time_chunk_requests(
                    self.request, self.request_chunks
                )
                if len(time_chunk_requests) > 1:
                    self.chunks["time"] = time_chunk
                    self.chunk_requests["time"] = time_chunk_requests
                    self.chunked_coords["time"] = xr.IndexVariable("time", time, {})  # type: ignore
        self.maybe_update_coords_and_chunk_info("leadtime_hour", "step")
        self.maybe_update_coords_and_chunk_info("step", "step")
        self.maybe_update_coords_and_chunk_info(
            "pressure_level", "isobaricInhPa", indexer_kwargs={"units": "hPa"}
        )
        self.maybe_update_coords_and_chunk_info(
            "levelist", "isobaricInhPa", indexer_kwargs={"units": "hPa"}
        )
        # `number` is last because some CDS datasets do not allow to select
        # ensemble members in the request and always return all of them.
        # In this case we set the dimension in `get_coords_attrs_and_dtype`
        self.maybe_update_coords_and_chunk_info(
            "number",
            "number",
            dtype="int64",
        )
        return self.chunked_coords.copy()

    def is_reanalysis(self, sample_ds: xr.Dataset) -> bool:
        out = False
        if (
            "leadtime_hour" not in self.request
            and "step" not in self.request
            and "step" in sample_ds.dims
        ):
            out = True
        return out

    def get_coords_attrs_and_dtype(
        self, dataset_cacher: client_common.DatasetCacherProtocol
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
        chunked_request_coords = self.compute_chunked_request_coords()
        self.request_chunked_dims = list(self.chunked_coords)
        sample_request = self.first_chunk_request()
        # HACK: this is a horrible work-around for ERA5 derived datasets that
        #   are indexed by "time" and "step" when the request has no "step"
        self.force_valid_time_as_time = False
        with dataset_cacher.retrieve(
            sample_request, override_cache_file=True
        ) as sample_ds:
            self.force_valid_time_as_time = self.is_reanalysis(sample_ds)
        with dataset_cacher.retrieve(
            sample_request,
            override_cache_file=True,
            force_valid_time_as_time=self.force_valid_time_as_time,
        ) as sample_ds:
            da = list(sample_ds.data_vars.values())[0]
            coords: dict[str, Any] = {}
            # ensure order
            for name in COORDINATES_ORDER:
                if name in chunked_request_coords:
                    coords[name] = chunked_request_coords[name]
                elif name in da.dims:
                    assert isinstance(name, str)
                    coords[name] = da.coords[name]
            for name in da.coords:  # type: ignore
                if name not in coords and name in da.dims:
                    assert isinstance(name, str)
                    coords[name] = da.coords[name]
            self.dims = list(coords)
            return coords, sample_ds.attrs, da.attrs, da.dtype

    def get_variables(self) -> dict[str, "CdsapiRequestChunker"]:
        if "variable" in self.request:
            param = "variable"
        elif "param" in self.request:
            param = "param"
        else:
            raise ValueError(f"'variable' parameter not found in {list(self.request)}")
        retval = {}
        for name in self.request[param]:
            var_request = self.request | {param: [name]}
            retval[name] = CdsapiRequestChunker(var_request, self.request_chunks)
        return retval

    def build_requests(self, chunk_requests: dict[str, Any]) -> dict[str, Any]:
        request = self.request.copy()
        request.update(**chunk_requests)
        return request

    def find_chunk_index(self, dim: str, key: int) -> int:
        if key is not None:
            start_chunks = [chunk[0] for chunk in self.chunk_requests[dim]]
            # to check
            index = bisect.bisect(start_chunks, key) - 1
        else:
            index = 0
        return index

    def first_chunk_request(self) -> dict[str, Any]:
        request = self.request.copy()
        for chunks in self.chunk_requests.values():
            request.update(**chunks[0][1])
        return request

    def ensure_dims_order(self, da: xr.DataArray) -> xr.DataArray:
        dims = []
        for dim in self.dims:
            if dim in da.dims:
                dims.append(dim)
        return da.transpose(*dims)

    def get_chunk_values(
        self,
        key: tuple[int | slice, ...],
        dataset_cacher: client_common.DatasetCacherProtocol,
    ) -> np.typing.ArrayLike:
        # XXX: only support `key` that access exactly one chunk
        assert len(key) == len(self.dims)
        chunks_key = {}
        for name, k in zip(self.dims, key):
            if name in self.request_chunked_dims:
                chunks_key[name] = k

        chunks_requests: dict[str, Any] = {}
        selection = dict(zip(self.dims, key))
        indices = {}
        for dim, request_key in chunks_key.items():
            if isinstance(request_key, slice):
                chunk_index = self.find_chunk_index(dim, request_key.start)
                start_chunk = self.chunk_requests[dim][chunk_index][0]
                chunks_requests.update(**self.chunk_requests[dim][chunk_index][1])
                # compute relative index
                if request_key.start is None:
                    start = None
                else:
                    start = request_key.start - start_chunk
                if request_key.stop is None:
                    stop = None
                else:
                    stop = request_key.stop - start_chunk
                selection[dim] = slice(start, stop, request_key.step)

            elif isinstance(request_key, int):
                chunk_index = self.find_chunk_index(dim, request_key)
                start_chunk = self.chunk_requests[dim][chunk_index][0]
                chunks_requests.update(**self.chunk_requests[dim][chunk_index][1])
                selection[dim] = request_key - start_chunk
            else:
                raise ValueError(f"key type {type(request_key)} not supported")
            indices[dim] = chunk_index

        field_request = self.build_requests(chunks_requests)
        with dataset_cacher.retrieve(
            field_request, force_valid_time_as_time=self.force_valid_time_as_time
        ) as ds:
            da = list(ds.data_vars.values())[0]
            da = self.ensure_dims_order(da)

            axis = []
            dims = []
            for ax, dim in enumerate(self.dims):
                if dim not in da.dims:
                    axis.append(ax)
                    dims.append(dim)

            da = da.expand_dims(dims, axis=axis)

            out = da.load().isel(selection)
            return out.values
