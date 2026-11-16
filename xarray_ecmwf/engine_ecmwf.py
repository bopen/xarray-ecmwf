import contextlib
import errno
import functools
import glob
import hashlib
import logging
import os
import shutil
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


def remove_file_or_folder(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            os.remove(path)
        except Exception:
            pass


def remove_cfgrib_indexes(path: str) -> None:
    # cfgrib names the index files of a GRIB file {path}.{short_hash}.idx
    for index_path in glob.glob(glob.escape(path) + ".*.idx"):
        remove_file_or_folder(index_path)


def private_copy_path(path: str) -> str:
    # a name only one read knows, so no other reader can use, replace or delete it
    root, ext = os.path.splitext(path)
    return f"{root}-{str(uuid.uuid4())[:8]}{ext}"


def robust_save_to_file(
    saver: Callable[..., Any], args: Sequence[Any], path: str
) -> None:
    tmp_path = path + "." + str(uuid.uuid4())[:8]

    try:
        saver(*args, tmp_path)
    except Exception as ex:
        remove_file_or_folder(tmp_path)
        raise ex

    try:
        if os.path.isdir(tmp_path):
            # a folder, like a .zarr schema file, can't be hard-linked: renaming it
            #   fails if path is already there
            os.rename(tmp_path, path)
        else:
            # a file is linked, not renamed: renaming would replace the file already
            #   there, linking fails
            os.link(tmp_path, path)
    except OSError as ex:
        # another process saved it first: keep that one, so that its readers
        #   don't see it replaced
        if ex.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise ex
    finally:
        # the temporary name: a file linked to path stays there
        remove_file_or_folder(tmp_path)


@attrs.define(slots=False)
class DatasetCacher:
    request_client: client_common.RequestClientProtocol
    open_dataset: Callable[..., xr.Dataset] = xr.open_dataset
    # "keep" the data files in the cache and reuse them, "delete" a private copy
    #   after each read, or "refresh" the cached ones
    data_files: str = attrs.field(
        default="keep", validator=attrs.validators.in_(["keep", "delete", "refresh"])
    )
    cache_folder: str = "./.xarray-ecmwf-cache"

    def request_data_file(self, request: dict[str, Any]) -> tuple[Any, str]:
        # asks the provider: returns the result to download, and where the data
        #   file is kept in the cache folder, shared by every reader of the request
        LOGGER.info(f"retrieving {request}")
        result = self.request_client.submit_and_wait_on_result(request)
        filename = self.request_client.get_filename(result)
        return result, os.path.join(self.cache_folder, filename)

    def maybe_download_data_file(self, result: Any, path: str) -> None:
        # downloads the data file to path, unless it's already there. A private copy
        #   path is new, so it's always downloaded
        if not os.path.isdir(self.cache_folder):
            os.makedirs(self.cache_folder, exist_ok=True)

        with xr.backends.locks.get_write_lock(f"{HOSTNAME}-grib"):
            if not os.path.exists(path):
                robust_save_to_file(self.request_client.download, (result,), path)

    @contextlib.contextmanager
    def open_data_file_and_clean(
        self, request: dict[str, Any], tries: int = 2
    ) -> Iterator[xr.Dataset]:
        # opens the cached data file, or with "delete" and "refresh" a private copy,
        #   then deleted or moved into the cache
        private_copy = self.data_files != "keep"
        # only the request and the download are tried again, e.g. when polytope
        #   returns an empty file
        for try_ in range(tries):
            try:
                result, cache_path = self.request_data_file(request)
                read_path = cache_path
                if private_copy:
                    read_path = private_copy_path(cache_path)
                self.maybe_download_data_file(result, read_path)
                break
            except RuntimeError:
                LOGGER.exception(f"Failed retrieve: {try_} / {tries}")
        else:
            raise RuntimeError(f"too many retries {tries}")

        try:
            # cfgrib reopens the file by name for each field: if a refresh replaced
            #   it meanwhile, a wrong field could be read with no error: fail instead
            before = os.stat(read_path)
            ds = self.open_dataset(read_path)
            LOGGER.debug("request: %r ->\n%r", request, list(ds.data_vars.values())[0])
            yield ds
            if not os.path.samestat(before, os.stat(read_path)):
                raise RuntimeError(f"{read_path} was replaced during the read")
            if self.data_files == "refresh":
                # the new data file was read fine: it goes into the cache, and the
                #   indexes of the old one are stale
                try:
                    # cfgrib ignores an index older than its data file: with the time
                    #   of now, all indexes of the old file are older, even recent ones
                    os.utime(read_path)
                    os.replace(read_path, cache_path)
                    remove_cfgrib_indexes(cache_path)
                except OSError:
                    # the read worked: don't fail it because the cache can't be
                    #   updated
                    LOGGER.exception(f"failed to replace {cache_path}")
        finally:
            if private_copy:
                # already gone if it was moved into the cache
                remove_file_or_folder(read_path)

    @contextlib.contextmanager
    def cached_empty_dataset(self, request: dict[str, Any]) -> Iterator[xr.Dataset]:
        LOGGER.info(f"cached_empty_dataset {request}")
        request_stable = {k: v for k, v in request.items() if k != "download_format"}
        filename = (
            hashlib.md5(str(request_stable).encode("utf-8")).hexdigest() + ".zarr"
        )
        path = os.path.join(self.cache_folder, filename)

        if not os.path.isdir(self.cache_folder):
            os.makedirs(self.cache_folder, exist_ok=True)

        if not os.path.exists(path):
            with self.open_data_file_and_clean(request) as read_ds:
                # check again as the download may be long
                with xr.backends.locks.get_write_lock(f"{HOSTNAME}-zarr"):
                    if not os.path.exists(path):
                        # NOTE: be sure that read_ds is chunked so compute=False only
                        #   writes the metadata. Some open_dataset
                        read_ds = read_ds.chunk()
                        # every process writes its own copy, and the first one
                        #   renamed into place is the one everybody opens
                        robust_save_to_file(
                            functools.partial(read_ds.to_zarr, compute=False), (), path
                        )
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
        for old_key in ["cache_file", "cache_data_files"]:
            if old_key in cache_kwargs:
                raise TypeError(
                    f"cache_kwargs {old_key!r} was replaced by 'data_files': "
                    "true is 'keep', false is 'delete'"
                )
        request_client_class = request_client_class or SUPPORTED_CLIENTS[client]
        request_chunker_class = SUPPORTED_CHUNKERS[chunker]

        request_client = request_client_class(client_kwargs)
        request_chunker = request_chunker_class(
            filename_or_obj, request_chunks, **request_chunker_kwargs
        )
        open_dataset_kwargs = {"engine": "cfgrib", "chunks": {}} | open_dataset_kwargs
        if (
            open_dataset_kwargs["engine"] == "cfgrib"
            and cache_kwargs.get("data_files", "keep") != "keep"
        ):
            # a private data file is read once: don't write a cfgrib index for it.
            #   indexpath is a cfgrib option: other engines fail on it
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
            lazy_var_data = xr.core.indexing.LazilyIndexedArray(var_data)
            var = xr.Variable(dims, lazy_var_data, var_attrs, encoding)
            data_vars[name] = var

        if not data_vars:
            raise latest_ex

        dataset = xr.Dataset(data_vars, coords, attrs)
        return dataset
