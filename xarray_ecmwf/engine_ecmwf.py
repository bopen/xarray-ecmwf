import contextlib
import logging
import os
import uuid
from typing import Any, Callable, Iterable, Iterator, Sequence

import attrs
import numpy as np
import xarray as xr

from . import client_cdsapi, client_common, client_ecmwf_opendata

LOGGER = logging.getLogger(__name__)

SUPPORTED_CLIENTS: dict[str, type[client_common.RequestClientProtocol]] = {
    "cdsapi": client_cdsapi.CdsapiRequestClient,
    "ecmwf-opendata": client_ecmwf_opendata.EcmwfOpendataRequestClient,
}
SUPPORTED_CHUNKERS: dict[str, type[client_common.RequestChunkerProtocol]] = {
    "cdsapi": client_cdsapi.CdsapiRequestChunker,
    "ecmwf-opendata": client_cdsapi.CdsapiRequestChunker,
}


@attrs.define(slots=False)
class ECMWFBackendArray(xr.backends.BackendArray):
    shape: tuple[int, ...]
    dtype: Any
    request_chunker: client_common.RequestChunkerProtocol
    dataset_cacher: client_common.DatasetCacherProtocol

    def __getitem__(
        self, key: xr.core.indexing.ExplicitIndexer
    ) -> np.typing.NDArray[np.float32]:
        data = xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )
        return data  # type: ignore

    def _raw_indexing_method(self, key: tuple[int | slice, ...]) -> np.typing.ArrayLike:
        out = self.request_chunker.get_chunk_values(key, self.dataset_cacher)
        return out


def robust_save_to_file(
    saver: Callable[..., Any], args: Sequence[Any], path: str
) -> None:
    tmp_path = path + "." + str(uuid.uuid4())[:8]

    ex = None
    try:
        saver(*args, tmp_path)
    except Exception:
        logging.exception("Save failed")

    if ex is not None:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise ex

    os.rename(tmp_path, path)


@attrs.define(slots=False)
class DatasetCacher:
    request_client: client_common.RequestClientProtocol
    cfgrib_kwargs: dict[str, Any] = {}
    cache_file: bool = True
    cache_folder: str = "./.xarray-ecmwf-cache"

    @contextlib.contextmanager
    def retrieve(
        self,
        request: dict[str, Any],
        override_cache_file: bool | None = None,
    ) -> Iterator[xr.Dataset]:
        LOGGER.info(f"retrieving {request}")
        cache_file = self.cache_file
        if override_cache_file is not None:
            cache_file = override_cache_file
        cfgrib_kwargs = self.cfgrib_kwargs
        if not cache_file:
            cfgrib_kwargs = cfgrib_kwargs | {"indexpath": ""}

        result = self.request_client.submit_and_wait_on_result(request)
        filename = self.request_client.get_filename(result)
        path = os.path.join(self.cache_folder, filename)

        if not os.path.isdir(self.cache_folder):
            os.makedirs(self.cache_folder, exist_ok=True)

        if not os.path.exists(path):
            robust_save_to_file(self.request_client.download, (result,), path)
        ds = xr.open_dataset(path, engine="cfgrib", **cfgrib_kwargs)
        LOGGER.debug("request: %r ->\n%r", request, list(ds.data_vars.values())[0])
        try:
            yield ds
        finally:
            if not cache_file:
                try:
                    os.remove(path)
                except Exception:
                    logging.exception("While removing a cache file")


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
        request_chunker_kwargs: dict[str, Any] = {},
        request_client_class: type[client_common.RequestClientProtocol] | None = None,
    ) -> xr.Dataset:
        if not isinstance(filename_or_obj, dict):
            raise TypeError("argument must be a valid request dictionary")
        request_client_class = request_client_class or SUPPORTED_CLIENTS[client]
        request_chunker_class = SUPPORTED_CHUNKERS[chunker]

        request_client = request_client_class(client_kwargs)
        request_chunker = request_chunker_class(
            filename_or_obj, request_chunks, **request_chunker_kwargs
        )
        dataset_cacher = DatasetCacher(request_client, cfgrib_kwargs, **cache_kwargs)
        LOGGER.info(request_chunker.get_request_dimensions())

        data_vars = {}
        for var_name, var_request_chunker in request_chunker.get_variables().items():
            (
                name,
                coords,
                attrs,
                var_attrs,
                dtype,
            ) = var_request_chunker.get_coords_attrs_and_dtype(dataset_cacher)
            shape = tuple(c.size for c in coords.values())
            dims = list(coords)
            encoding = {
                "preferred_chunks": var_request_chunker.get_chunks(),
                "request_chunker": var_request_chunker,
            }

            var_data = ECMWFBackendArray(
                shape,
                dtype,
                var_request_chunker,
                dataset_cacher,
            )
            lazy_var_data = xr.core.indexing.LazilyIndexedArray(var_data)  # type: ignore
            var = xr.Variable(dims, lazy_var_data, var_attrs, encoding)
            data_vars[name] = var

        dataset = xr.Dataset(data_vars, coords, attrs)
        return dataset
