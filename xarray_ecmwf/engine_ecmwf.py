import contextlib
import os
from typing import Any, Iterable

import attrs
import cf2cdm
import xarray as xr

from . import client_protocol

SUPPORTED_CLIENTS = {"cdsapi": None}
SUPPORTED_CHUNKERS = {"cdsapi": None}


class ECMWFBackendArray(xr.backends.BackendArray):
    ...


attrs.define(slots=False)


class DatasetsCacher:
    request_client: client_protocol.RequestClientProtocol
    cfgrib_kwargs: dict[str, Any] = {"time_dims": ["valid_time"]}
    translate_coords_kwargs: dict[str, Any] = {"coord_model": cf2cdm.CDS}
    cache_file: bool = False
    cache_folder: str = "./.xarray-ecmwf-cache"

    def __attrs_post_init__(self):
        if not os.path.isdir(self.cache_folder):
            os.mkdir(self.cache_folder)

    @contextlib.contextmanager
    def dataset(self, request: dict[str, Any], override_cache_file: bool | None = None):
        cache_file = self.cache_file
        if override_cache_file is not None:
            cache_file = override_cache_file
        cfgrib_kwargs = self.cfgrib_kwargs
        if not cache_file:
            cfgrib_kwargs = cfgrib_kwargs | {"indexpath": ""}

        result = self.request_client.retrieve(request)
        filename = self.request_client.get_filename(result)
        path = os.path.join(self.cache_folder, filename)

        with xr.backends.locks.get_write_lock(filename):
            if not os.path.exists(path):
                try:
                    self.request_client.download(path)
                except Exception:
                    os.remove(path)
            ds = xr.open_dataset(path, engine="cfgrib", **cfgrib_kwargs)
            yield cf2cdm.translate_coords(ds, **self.translate_coords_kwargs)
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
        translate_coords_kwargs: dict[str, Any] = {"coord_model": cf2cdm.CDS},
    ) -> xr.Dataset:
        if not isinstance(filename_or_obj, dict):
            raise TypeError("argument must be a valid request dictionary")
        request_client_class = SUPPORTED_CLIENTS[client]
        request_chunker_class = SUPPORTED_CHUNKERS[chunker]

        request_client = request_client_class(client_kwargs)
        request_chunker = request_chunker_class(filename_or_obj, request_chunks)
        dataset_cacher = DatasetsCacher(
            request_client, cfgrib_kwargs, translate_coords_kwargs, **cache_kwargs
        )

        coords, attrs, dtype = request_chunker.get_coords_attrs_and_dtype(
            dataset_cacher
        )
        shape = [c.size for c in coords.values()]
        dims = list(coords)
        encoding = {"preferred_chunks": request_chunker.get_chunks()}

        data_vars = {}
        for var_name in request_client.get_variables():
            var_data = ECMWFBackendArray(
                shape,
                dtype,
                request_chunker,
                request_client,
                cache_kwargs,
            )
            lazy_var_data = xr.core.indexing.LazilyIndexedArray(var_data)
            var = xr.Variable(dims, lazy_var_data, attrs, encoding)
            data_vars[var_name] = var

        dataset = xr.Dataset(data_vars=data_vars, coords=coords)
        return dataset
