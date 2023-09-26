import bisect
import contextlib
import logging
import os
from typing import Any, Iterable, Iterator

import attrs
import cf2cdm
import numpy as np
import xarray as xr

from . import client_cdsapi, client_common, client_ecmwf_opendata

LOGGER = logging.getLogger(__name__)

SUPPORTED_CLIENTS = {
    "cdsapi": client_cdsapi.CdsapiRequestClient,
    "ecmwf-opendata": client_ecmwf_opendata.EcmwfOpendataRequestClient,
}
SUPPORTED_CHUNKERS = {
    "cdsapi": client_cdsapi.CdsapiRequestChunker,
    "ecmwf-opendata": client_cdsapi.CdsapiRequestChunker,
}


@attrs.define(slots=False)
class ECMWFBackendArray(xr.backends.BackendArray):
    shape: Iterable[int]
    dtype: Any
    request_chunker: client_common.RequestChunkerProtocol
    dataset_cacher: client_common.DatasetCacherProtocol

    def __getitem__(self, key: xr.core.indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def build_requests(self, chunk_requests=None):
        pass

    def find_start(self, dim: str, key: int):
        chunk_requests = self.request_chunker.chunk_requests[dim]
        start_chunks = [chunk[0] for chunk in chunk_requests]
        return bisect.bisect(start_chunks, key)

    def _raw_indexing_method(self, key: tuple[int | slice, ...]) -> np.typing.ArrayLike:
        # XXX: only support `key` that access exactly one chunk
        assert len(key) == len(self.request_chunker.dims)
        request_keys = key[: len(self.request_chunker.request_dims)]

        chunks_requests = []
        for dim, request_key in zip(self.request_chunker.request_dims, request_keys):
            if isinstance(request_key, slice):
                # XXX: check that the slice is exactly one chunk for everything except lat lon
                if request_key.start is None:
                    index = 0
                else:
                    index = self.find_start(dim, request_key.start)
                chunks_requests.append(
                    self.request_chunker.chunk_requests[dim][index][1]
                )
            else:
                pass

        field_request = self.build_requests(chunks_requests)
        with self.dataset_cacher.retrieve(field_request) as ds:
            da = list(ds.data_vars.values())[0]
        # XXX: check that the dimensions are in the correct order or rollaxis
        return da.values


@attrs.define(slots=False)
class DatasetCacher:
    request_client: client_common.RequestClientProtocol
    cfgrib_kwargs: dict[str, Any] = {}
    translate_coords_kwargs: dict[str, Any] | None = {"coord_model": cf2cdm.CDS}
    cache_file: bool = True
    cache_folder: str = "./.xarray-ecmwf-cache"

    def __attrs_post_init__(self) -> None:
        if not os.path.isdir(self.cache_folder):
            os.mkdir(self.cache_folder)

    @contextlib.contextmanager
    def retrieve(
        self, request: dict[str, Any], override_cache_file: bool | None = None
    ) -> Iterator[xr.Dataset]:
        cache_file = self.cache_file
        if override_cache_file is not None:
            cache_file = override_cache_file
        cfgrib_kwargs = self.cfgrib_kwargs
        if not cache_file:
            cfgrib_kwargs = cfgrib_kwargs | {"indexpath": ""}

        result = self.request_client.submit_and_wait_on_result(request)
        filename = self.request_client.get_filename(result)
        path = os.path.join(self.cache_folder, filename)

        with xr.backends.locks.get_write_lock(filename):
            if not os.path.exists(path):
                try:
                    self.request_client.download(result, path)
                except Exception:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    raise
            ds = xr.open_dataset(path, engine="cfgrib", **cfgrib_kwargs)
            LOGGER.debug("request: %r ->\n%r", request, list(ds.data_vars.values())[0])
            try:
                yield ds
            finally:
                if not cache_file:
                    os.remove(path)


class ECMWFBackendEntrypoint(xr.backends.BackendEntrypoint):
    def open_dataset(  # type:ignore
        self,
        filename_or_obj: dict[str, Any],
        *,
        drop_variables: str | Iterable[str] | None = None,
        client: str = "cdsapi",
        client_kwargs: dict[str, Any] = {},
        chunker: str = "cdsapi",
        request_chunks: dict[str, Any] = {},
        cache_kwargs: dict[str, Any] = {},
        cfgrib_kwargs: dict[str, Any] = {},
        translate_coords_kwargs: dict[str, Any] | None = {"coord_model": cf2cdm.CDS},
    ) -> xr.Dataset:
        if not isinstance(filename_or_obj, dict):
            raise TypeError("argument must be a valid request dictionary")
        request_client_class = SUPPORTED_CLIENTS[client]
        request_chunker_class = SUPPORTED_CHUNKERS[chunker]

        request_client = request_client_class(client_kwargs)
        request_chunker = request_chunker_class(filename_or_obj, request_chunks)
        dataset_cacher = DatasetCacher(
            request_client, cfgrib_kwargs, translate_coords_kwargs, **cache_kwargs
        )
        LOGGER.info(request_chunker.get_request_dimensions())

        coords, attrs, var_attrs, dtype = request_chunker.get_coords_attrs_and_dtype(
            dataset_cacher
        )
        shape = [c.size for c in coords.values()]
        dims = list(coords)
        encoding = {"preferred_chunks": request_chunker.get_chunks()}

        data_vars = {}
        for var_name in request_chunker.get_variables():
            var_data = ECMWFBackendArray(
                shape,
                dtype,
                request_chunker,
                dataset_cacher,
            )
            lazy_var_data = xr.core.indexing.LazilyIndexedArray(var_data)
            var = xr.Variable(dims, lazy_var_data, var_attrs, encoding)
            data_vars[var_name] = var

        dataset = xr.Dataset(data_vars, coords, attrs)
        return dataset
