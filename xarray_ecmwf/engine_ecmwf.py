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

    def _raw_indexing_method(self, key: tuple[int | slice, ...]) -> np.typing.ArrayLike:
        assert len(key) == 3
        # XXX:
        itime, ilat, ilon = key
        if isinstance(itime, slice):
            start_index, stop_index = self.find_start_stop(itime.start, itime.stop)
            split_start = self.chunk_requests["time"][start_index][0]
            chunk_requests = self.chunk_requests["time"][start_index : stop_index + 1]
            htime = slice(itime.start - split_start, itime.stop - split_start)
        else:
            start_index, stop_index = self.find_start_stop(itime, itime + 1)
            split_start = self.chunk_requests["time"][start_index][0]
            chunk_requests = self.chunk_requests["time"][start_index : stop_index + 1]
            htime = itime - split_start
        chunks = []
        for field_request in self.build_requests(chunk_requests):
            with self.dataset_cacher.retrieve(field_request) as ds:
                da = list(ds.data_vars.values())[0]
                chunks.append(da)
        cda = xr.concat(chunks, dim="time")
        values = cda.isel(time=htime, lat=ilat, lon=ilon).values
        return values


@attrs.define(slots=False)
class DatasetCacher:
    request_client: client_common.RequestClientProtocol
    cfgrib_kwargs: dict[str, Any] = {"time_dims": ["valid_time"]}
    translate_coords_kwargs: dict[str, Any] | None = {"coord_model": cf2cdm.CDS}
    cache_file: bool = False
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
            if self.translate_coords_kwargs is not None:
                ds = cf2cdm.translate_coords(ds, **self.translate_coords_kwargs)
            LOGGER.debug("request: %r ->\n%r", request, ds)
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
        cfgrib_kwargs: dict[str, Any] = {"time_dims": ["valid_time"]},
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

        coords, attrs, var_attrs, dtype = request_chunker.get_coords_attrs_and_dtype(
            dataset_cacher
        )
        shape = [c.size for c in coords.values()]
        dims = list(coords)
        encoding = {}  # {"preferred_chunks": request_chunker.get_chunks()}

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
