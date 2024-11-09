import contextlib
import functools
import hashlib
import logging
import os
import socket
import uuid
from typing import Any, Callable, Iterable, Iterator, Sequence

import attrs
import numpy as np
import xarray as xr

from . import client_cdsapi, client_common, client_ecmwf_opendata, client_polytope

LOGGER = logging.getLogger(__name__)
HOSTNAME = socket.gethostname()

SUPPORTED_CLIENTS: dict[str, type[client_common.RequestClientProtocol]] = {
    "cdsapi": client_cdsapi.CdsapiRequestClient,
    "ecmwf-opendata": client_ecmwf_opendata.EcmwfOpendataRequestClient,
    "polytope": client_polytope.PolytopeRequestClient,
}
SUPPORTED_CHUNKERS: dict[str, type[client_common.RequestChunkerProtocol]] = {
    "cdsapi": client_cdsapi.CdsapiRequestChunker,  # type: ignore
    "ecmwf-opendata": client_cdsapi.CdsapiRequestChunker,  # type: ignore
    "polytope": client_cdsapi.CdsapiRequestChunker,  # type: ignore
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

    try:
        saver(*args, tmp_path)
    except Exception as ex:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise ex

    os.rename(tmp_path, path)


@attrs.define(slots=False)
class DatasetCacher:
    request_client: client_common.RequestClientProtocol
    open_dataset: Callable[..., xr.Dataset] = xr.open_dataset
    cache_file: bool = True
    cache_folder: str = "./.xarray-ecmwf-cache"

    @contextlib.contextmanager
    def retrieve(
        self,
        request: dict[str, Any],
        override_cache_file: bool | None = None,
        tries: int = 2,
    ) -> Iterator[xr.Dataset]:
        for try_ in range(tries):
            try:
                with self.retrieve_once(request, override_cache_file) as ds:
                    yield ds
                break
            except RuntimeError:
                LOGGER.exception(f"Failed retrieve: {try_} / {tries}")
        else:
            raise RuntimeError(f"too many retries {tries}")

    @contextlib.contextmanager
    def retrieve_once(
        self, request: dict[str, Any], override_cache_file: bool | None = None
    ) -> Iterator[xr.Dataset]:
        LOGGER.info(f"retrieving {request}")
        cache_file = self.cache_file
        if override_cache_file is not None:
            cache_file = override_cache_file

        result = self.request_client.submit_and_wait_on_result(request)
        filename = self.request_client.get_filename(result)
        path = os.path.join(self.cache_folder, filename)

        if not os.path.isdir(self.cache_folder):
            os.makedirs(self.cache_folder, exist_ok=True)

        with xr.backends.locks.get_write_lock(f"{HOSTNAME}-grib"):  # type: ignore
            if not os.path.exists(path):
                robust_save_to_file(self.request_client.download, (result,), path)
        ds = self.open_dataset(path)
        LOGGER.debug("request: %r ->\n%r", request, list(ds.data_vars.values())[0])
        try:
            yield ds
        finally:
            if not cache_file:
                try:
                    os.remove(path)
                    # remove the associated cfgrib index file
                    try:
                        os.remove(path + ".idx")
                    except Exception:
                        pass
                except Exception:
                    LOGGER.exception("While removing a cache file")

    @contextlib.contextmanager
    def cached_empty_dataset(self, request: dict[str, Any]) -> Iterator[xr.Dataset]:
        LOGGER.info(f"cached_empty_dataset {request}")
        filename = hashlib.md5(str(request).encode("utf-8")).hexdigest() + ".zarr"
        path = os.path.join(self.cache_folder, filename)

        if not os.path.isdir(self.cache_folder):
            os.makedirs(self.cache_folder, exist_ok=True)

        if not os.path.exists(path):
            with self.retrieve(request) as read_ds:
                # check again as the retrieve may be long
                with xr.backends.locks.get_write_lock(f"{HOSTNAME}-zarr"):  # type: ignore
                    if not os.path.exists(path):
                        # NOTE: be sure that read_ds is chunked so compute=False only
                        #   writes the metadata. Some open_dataset
                        read_ds = read_ds.chunk()
                        read_ds.to_zarr(path, compute=False)
        yield xr.open_dataset(path, engine="zarr")


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
        open_dataset_kwargs: dict[str, Any] = {},
        request_chunker_kwargs: dict[str, Any] = {},
        request_client_class: type[client_common.RequestClientProtocol] | None = None,
        open_dataset: Callable[[str], xr.Dataset] = xr.open_dataset,
    ) -> xr.Dataset:
        if not isinstance(filename_or_obj, dict):
            raise TypeError("argument must be a valid request dictionary")
        request_client_class = request_client_class or SUPPORTED_CLIENTS[client]
        request_chunker_class = SUPPORTED_CHUNKERS[chunker]

        request_client = request_client_class(client_kwargs)
        request_chunker = request_chunker_class(
            filename_or_obj, request_chunks, **request_chunker_kwargs
        )
        open_dataset_kwargs = {"engine": "cfgrib", "chunks": {}} | open_dataset_kwargs
        if not cache_kwargs.get("cache_file", True):
            open_dataset_kwargs = open_dataset_kwargs | {"indexpath": ""}

        open_dataset = functools.partial(open_dataset, **open_dataset_kwargs)
        dataset_cacher = DatasetCacher(request_client, open_dataset, **cache_kwargs)
        LOGGER.info(request_chunker.get_request_dimensions())

        data_vars = {}
        for var_name, var_request_chunker in request_chunker.get_variables().items():
            # drop_variables: both on var_name...
            if drop_variables is not None and var_name in drop_variables:
                continue
            try:
                var_def = var_request_chunker.get_coords_attrs_and_dtype(dataset_cacher)
                LOGGER.info(f"found  variable {var_name} as {var_def[0]}")
            except Exception as ex:
                LOGGER.exception(f"failed to define variable {var_name}")
                latest_ex = ex
                continue
            name, coords, attrs, var_attrs, dtype = var_def
            # drop_variables: ... and on name
            if drop_variables is not None and name in drop_variables:
                continue
            shape = tuple(c.size for c in coords.values())
            dims = list(coords)
            encoding = {
                "preferred_chunks": var_request_chunker.get_chunks(),
                "request_chunker": var_request_chunker,
                "request_client": request_client,
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

        if not data_vars:
            raise latest_ex

        dataset = xr.Dataset(data_vars, coords, attrs)
        return dataset
