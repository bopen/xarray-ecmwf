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


def unique_path(path: str) -> str:
    # a new path next to path, e.g. abc-1a2b3c4d.grib for abc.grib, that only the
    #   caller knows, so no other process can use, replace or delete its file
    root, ext = os.path.splitext(path)
    return f"{root}-{str(uuid.uuid4())[:8]}{ext}"


def link_data_file(data_file: str, link: str) -> None:
    # makes link point to data_file, replacing the link already there. The new link
    #   holds only the file name and is renamed onto the old one, so it's never missing
    tmp_link = unique_path(link)
    os.symlink(os.path.basename(data_file), tmp_link)
    try:
        os.replace(tmp_link, link)
    finally:
        remove_file_or_folder(tmp_link)


def replace_data_file(data_file: str, link: str) -> None:
    # points link to data_file, then deletes the file it pointed to and that file's
    #   cfgrib indexes, named {path}.{short_hash}.idx
    old_path = os.path.realpath(link) if os.path.islink(link) else None
    link_data_file(data_file, link)
    if old_path:
        for path in [old_path, *glob.glob(glob.escape(old_path) + ".*.idx")]:
            remove_file_or_folder(path)


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
        os.rename(tmp_path, path)
    except OSError as ex:
        remove_file_or_folder(tmp_path)
        # another process saved it first and it can't be replaced: keep that one
        if ex.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise ex


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

    def maybe_download_data_file(self, result: Any, cache_path: str) -> str:
        # returns the data file to read: the one cache_path links to, downloaded and
        #   linked only if missing, or with "delete" and "refresh" a new private copy
        os.makedirs(self.cache_folder, exist_ok=True)
        with xr.backends.locks.get_write_lock(f"{HOSTNAME}-grib"):
            # exists follows the link: one to a file deleted by hand counts as missing
            if self.data_files == "keep" and os.path.exists(cache_path):
                return os.path.realpath(cache_path)
            download_path = unique_path(cache_path)
            robust_save_to_file(self.request_client.download, (result,), download_path)
            if self.data_files != "keep":
                return download_path
            try:
                # unlike a rename, fails if another process linked its copy first
                os.symlink(os.path.basename(download_path), cache_path)
            except FileExistsError:
                if os.path.exists(cache_path):  # use the copy linked first
                    remove_file_or_folder(download_path)
                else:  # a link to a file deleted by hand
                    link_data_file(download_path, cache_path)
        return os.path.realpath(cache_path)

    @contextlib.contextmanager
    def open_data_file_and_clean(
        self, request: dict[str, Any], tries: int = 2
    ) -> Iterator[xr.Dataset]:
        # only the request and the download are tried again, e.g. when polytope
        #   returns an empty file
        for try_ in range(tries):
            try:
                result, cache_path = self.request_data_file(request)
                read_path = self.maybe_download_data_file(result, cache_path)
                break
            except RuntimeError:
                LOGGER.exception(f"Failed retrieve: {try_} / {tries}")
        else:
            raise RuntimeError(f"too many retries {tries}")

        # a private copy, unless a refresh links it into the cache
        delete_after = self.data_files != "keep"
        try:
            ds = self.open_dataset(read_path)
            LOGGER.debug("request: %r ->\n%r", request, list(ds.data_vars.values())[0])
            yield ds
            if self.data_files == "refresh":
                try:
                    replace_data_file(read_path, cache_path)
                except OSError:
                    # the read worked: don't fail it because the cache can't be updated
                    LOGGER.exception(f"failed to replace {cache_path}")
                else:
                    delete_after = False
        finally:
            if delete_after:
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
            # no cfgrib index for a new delete or refresh copy: keep reads write one.
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
