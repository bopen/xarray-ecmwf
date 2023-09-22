from typing import Any, Iterable

import xarray as xr

SUPPORTED_CLIENTS = {"cdsapi": None}
SUPPORTED_CHUNKERS = {"cdsapi": None}


class ECMWFBackendArray(xr.backends.BackendArray):
    ...


class ECMWFBackendEntrypoint(xr.backends.BackendEntrypoint):
    def open_dataset(  # type:ignore
        self,
        filename_or_obj: dict[str, Any],
        *,
        drop_variables: str | Iterable[str] | None = None,
        client: str = "cdsapi",
        chunker: str = "cdsapi",
        request_chunks: dict[str, Any] = {},
        cache_kwargs: dict[str, Any] = {},
    ) -> xr.Dataset:
        if not isinstance(filename_or_obj, dict):
            raise TypeError("argument must be a valid request dictionary")
        request_client = SUPPORTED_CLIENTS[client]
        request_chunker_class = SUPPORTED_CHUNKERS[chunker]

        request_chunker = request_chunker_class(filename_or_obj, request_chunks)
        coords, attrs, dtype = request_chunker.get_coords_attrs_and_dtype(
            request_client, cache_kwargs
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
