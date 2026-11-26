import concurrent.futures
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from xarray_ecmwf import engine_ecmwf

REQUEST = {"dataset": "dummy-dataset", "variable": ["dummy_variable"]}


# every download of the request gets the same file name, as with CDS
class DummyClient:
    def __init__(self, client_kwargs: dict[str, Any]) -> None:
        self.downloads = 0

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        return request

    def get_filename(self, result: Any) -> str:
        return "request.grib"

    def download(self, result: Any, target: str | None = None) -> str:
        assert target is not None
        self.downloads += 1
        Path(target).write_text(f"download {self.downloads}")
        return target


# the dataset records the file it was opened from
def open_dummy(path: str, **kwargs: Any) -> xr.Dataset:
    return xr.Dataset(
        {"t": ("time", np.zeros(3))}, coords={"time": [0, 1, 2]}, attrs={"path": path}
    )


# a downloaded file that can't be opened, like a corrupt GRIB
def open_broken(path: str, **kwargs: Any) -> xr.Dataset:
    raise ValueError(f"can't open {path}")


# reads the request once, and returns the path of the file read
def read_once(
    tmp_path: Path, data_files: str = "keep", client: DummyClient | None = None
) -> str:
    cacher = engine_ecmwf.DatasetCacher(
        client or DummyClient({}),
        open_dummy,
        cache_folder=str(tmp_path),
        data_files=data_files,
    )
    with cacher.open_data_file_and_clean(REQUEST) as ds:
        return str(ds.attrs["path"])


# several processes can build the same .zarr schema sample at once: each one
# writes its own copy under a temporary name, and the first copy renamed into
# place is the one everybody opens.


def open_sample(cache_folder: Path) -> xr.Dataset:
    cacher = engine_ecmwf.DatasetCacher(
        DummyClient({}), open_dummy, cache_folder=str(cache_folder)
    )
    with cacher.cached_empty_dataset(REQUEST) as ds:
        return ds


def test_cached_empty_dataset_keeps_the_sample_saved_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    to_zarr = xr.Dataset.to_zarr

    def to_zarr_after_another_process(
        self: xr.Dataset, store: Any, *args: Any, **kw: Any
    ) -> Any:
        sample = os.path.splitext(store)[0]
        # another process saves its complete sample while this one is writing
        to_zarr(self.assign_attrs(saved_by="another process"), sample, *args, **kw)
        return to_zarr(self, store, *args, **kw)

    monkeypatch.setattr(xr.Dataset, "to_zarr", to_zarr_after_another_process)

    ds = open_sample(tmp_path)

    assert ds.attrs["saved_by"] == "another process"
    assert len([name for name in os.listdir(tmp_path) if ".zarr" in name]) == 1


def test_robust_save_to_file_removes_a_folder_it_failed_to_write(
    tmp_path: Path,
) -> None:
    def failing_saver(target: str) -> None:
        os.makedirs(os.path.join(target, "half-written"))
        raise ValueError("write failed")

    with pytest.raises(ValueError):
        engine_ecmwf.robust_save_to_file(failing_saver, (), str(tmp_path / "a.zarr"))

    assert os.listdir(tmp_path) == []


def test_robust_save_to_file_raises_when_a_file_can_not_be_renamed(
    tmp_path: Path,
) -> None:
    os.makedirs(tmp_path / "a.grib" / "not-a-grib")

    with pytest.raises(OSError):
        engine_ecmwf.robust_save_to_file(
            lambda target: Path(target).touch(), (), str(tmp_path / "a.grib")
        )

    assert os.listdir(tmp_path) == ["a.grib"]


# with data_files="delete" or "refresh" each read downloads its own copy and deletes
# only that one, so readers don't delete each other's files


@pytest.mark.parametrize("data_files, files_left", [("delete", 0), ("refresh", 2)])
def test_readers_keep_their_own_files(
    tmp_path: Path, data_files: str, files_left: int
) -> None:
    client = DummyClient({})
    cacher = engine_ecmwf.DatasetCacher(
        client, open_dummy, cache_folder=str(tmp_path), data_files=data_files
    )

    # two readers of the same request at the same time
    with cacher.open_data_file_and_clean(REQUEST) as outer:
        with cacher.open_data_file_and_clean(REQUEST):
            pass
        assert os.path.exists(outer.attrs["path"])

    assert client.downloads == 2
    assert len(os.listdir(tmp_path)) == files_left
    # a refresh links its copy into the cache
    assert os.path.exists(tmp_path / "request.grib") == (data_files == "refresh")


# a downloaded file that can't be opened is deleted, and the cached one is kept
@pytest.mark.parametrize("data_files", ["delete", "refresh"])
def test_a_private_copy_that_cannot_be_opened_is_deleted(
    tmp_path: Path, data_files: str
) -> None:
    cached = tmp_path / "request.grib"
    cached.write_text("cached by an earlier run")
    cacher = engine_ecmwf.DatasetCacher(
        DummyClient({}), open_broken, cache_folder=str(tmp_path), data_files=data_files
    )

    with pytest.raises(ValueError):
        with cacher.open_data_file_and_clean(REQUEST):
            pass

    assert cached.read_text() == "cached by an earlier run"
    assert os.listdir(tmp_path) == ["request.grib"]


@pytest.mark.parametrize("old_key", ["cache_file", "cache_data_files"])
def test_open_dataset_names_the_new_key_for_old_ones(old_key: str) -> None:
    with pytest.raises(TypeError, match="replaced by 'data_files'"):
        xr.open_dataset(
            REQUEST,  # type: ignore
            engine="ecmwf",
            request_client_class=DummyClient,
            cache_kwargs={old_key: False},
        )


# indexpath is a cfgrib option: other engines, e.g. h5netcdf, fail on it.
#   data_files None: not set, so the default, delete
@pytest.mark.parametrize(
    "engine, data_files, indexpath",
    [
        ("cfgrib", "delete", ""),
        ("h5netcdf", "delete", None),
        ("cfgrib", "refresh", ""),
        ("cfgrib", None, ""),
    ],
)
def test_open_dataset_passes_indexpath_only_to_cfgrib(
    tmp_path: Path, engine: str, data_files: str | None, indexpath: str | None
) -> None:
    indexpaths = []

    def open_recording_indexpath(path: str, **kwargs: Any) -> xr.Dataset:
        indexpaths.append(kwargs.get("indexpath"))
        return open_dummy(path)

    cache_kwargs = {"cache_folder": str(tmp_path)}
    if data_files:
        cache_kwargs["data_files"] = data_files
    xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_client_class=DummyClient,
        cache_kwargs=cache_kwargs,
        open_dataset_kwargs={"engine": engine},
        open_dataset=open_recording_indexpath,
    )

    assert indexpaths == [indexpath]


def test_data_files_delete_is_the_default(tmp_path: Path) -> None:
    cacher = engine_ecmwf.DatasetCacher(
        DummyClient({}), open_dummy, cache_folder=str(tmp_path)
    )

    with cacher.open_data_file_and_clean(REQUEST):
        pass

    assert os.listdir(tmp_path) == []


def test_data_files_keep_reuses_and_keeps_the_file(tmp_path: Path) -> None:
    client = DummyClient({})
    read_once(tmp_path, client=client)
    read_path = read_once(tmp_path, client=client)

    cached = tmp_path / "request.grib"
    assert client.downloads == 1
    assert cached.read_text() == "download 1"
    # a relative link, so that it works wherever the cache folder is mounted
    assert os.readlink(cached) == os.path.basename(read_path)
    # the file read is the one linked, not the link: if a refresh deletes it, the
    #   read fails instead of reading fields of the new file
    assert read_path == os.path.realpath(cached)
    assert len(os.listdir(tmp_path)) == 2


# the download lock is per host: another worker pod can link its copy first


def test_data_files_keep_uses_the_copy_linked_first(tmp_path: Path) -> None:
    class OtherPodFirstClient(DummyClient):
        def download(self, result: Any, target: str | None = None) -> str:
            (tmp_path / "request-otherpod.grib").write_text("another pod's download")
            os.symlink("request-otherpod.grib", tmp_path / "request.grib")
            return super().download(result, target)

    read_path = read_once(tmp_path, client=OtherPodFirstClient({}))

    assert read_path.endswith("request-otherpod.grib")
    assert sorted(os.listdir(tmp_path)) == ["request-otherpod.grib", "request.grib"]


def test_data_files_keep_deletes_its_download_if_it_cannot_link_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_links(src: str, dst: str) -> None:
        raise PermissionError("symbolic links aren't supported")

    monkeypatch.setattr(os, "symlink", no_links)

    with pytest.raises(PermissionError):
        read_once(tmp_path)

    assert os.listdir(tmp_path) == []


def test_data_files_keep_downloads_again_a_file_deleted_by_hand(
    tmp_path: Path,
) -> None:
    os.symlink("request-deleted.grib", tmp_path / "request.grib")
    client = DummyClient({})

    read_once(tmp_path, client=client)

    assert client.downloads == 1
    assert (tmp_path / "request.grib").read_text() == "download 1"
    assert len(os.listdir(tmp_path)) == 2


# open_data_file_and_clean tries again a failed download, like polytope's empty
# files, but not an error raised inside its with block, e.g. while loading the
# values


def test_open_data_file_and_clean_tries_again_a_failed_download(
    tmp_path: Path,
) -> None:
    class EmptyFirstClient(DummyClient):
        def download(self, result: Any, target: str | None = None) -> str:
            path = super().download(result, target)
            if self.downloads == 1:
                raise RuntimeError("polytope returned an empty file")
            return path

    client = EmptyFirstClient({})

    read_once(tmp_path, client=client)

    assert client.downloads == 2
    assert (tmp_path / "request.grib").read_text() == "download 2"
    assert len(os.listdir(tmp_path)) == 2


def test_open_data_file_and_clean_does_not_try_again_an_error_inside_the_with_block(
    tmp_path: Path,
) -> None:
    client = DummyClient({})
    cacher = engine_ecmwf.DatasetCacher(
        client, open_dummy, data_files="delete", cache_folder=str(tmp_path)
    )

    with pytest.raises(RuntimeError, match="raised while loading the values"):
        with cacher.open_data_file_and_clean(REQUEST):
            raise RuntimeError("raised while loading the values")

    assert client.downloads == 1


# with data_files="refresh" each read downloads the data file again, reads its own
# copy, then points the link to it and deletes the old file


def test_data_files_refresh_downloads_again_and_refreshes_the_cache(
    tmp_path: Path,
) -> None:
    old = read_once(tmp_path)
    Path(old + ".5b7b6.idx").write_text("cfgrib index of the old file")

    new = os.path.basename(read_once(tmp_path, "refresh"))

    # the link points to the new file; the old file and its index are deleted
    assert os.readlink(tmp_path / "request.grib") == new
    assert sorted(os.listdir(tmp_path)) == sorted([new, "request.grib"])


# a link made by hand to a file outside the cache folder: refresh moves the link,
# but must not delete that file
def test_data_files_refresh_deletes_only_files_in_the_cache_folder(
    tmp_path: Path,
) -> None:
    archived = tmp_path / "archive.grib"
    archived.write_text("an archived file")
    cache_folder = tmp_path / "cache"
    cache_folder.mkdir()
    os.symlink(str(archived), cache_folder / "request.grib")

    new = os.path.basename(read_once(cache_folder, "refresh"))

    assert os.readlink(cache_folder / "request.grib") == new
    assert archived.read_text() == "an archived file"


def test_data_files_refresh_keeps_the_read_if_the_cache_cannot_be_updated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    old = read_once(tmp_path)

    def replace_denied(src: str, dst: str) -> None:
        raise PermissionError(f"can't replace {dst}")

    monkeypatch.setattr(os, "replace", replace_denied)

    read_once(tmp_path, "refresh")

    # the link and the old file are untouched
    assert "failed to replace" in caplog.text
    assert os.path.realpath(tmp_path / "request.grib") == old
    assert sorted(os.listdir(tmp_path)) == sorted(
        [os.path.basename(old), "request.grib"]
    )


def test_data_files_rejects_an_unknown_value() -> None:
    with pytest.raises(ValueError, match="'data_files' must be in"):
        engine_ecmwf.DatasetCacher(DummyClient({}), data_files="refesh")


# downloads that may overlap wait for each other to start: if they ran one at a
#   time, the first one would fail with a timeout
class ParallelClient(DummyClient):
    def __init__(self, client_kwargs: dict[str, Any]) -> None:
        super().__init__(client_kwargs)
        self.barrier = threading.Barrier(2, timeout=5)

    def get_filename(self, result: Any) -> str:
        return f"{result['dataset']}.grib"

    def download(self, result: Any, target: str | None = None) -> str:
        self.barrier.wait()
        return super().download(result, target)


# reads each request on its own thread, and returns the paths of the files read
def read_concurrently(
    tmp_path: Path, data_files: str, client: DummyClient, requests: list[Any]
) -> list[str]:
    cacher = engine_ecmwf.DatasetCacher(
        client, open_dummy, cache_folder=str(tmp_path), data_files=data_files
    )

    def read(request: dict[str, Any]) -> str:
        with cacher.open_data_file_and_clean(request) as ds:
            return str(ds.attrs["path"])

    with concurrent.futures.ThreadPoolExecutor(len(requests)) as pool:
        return list(pool.map(read, requests))


# with data_files="delete" and "refresh" each read downloads its private copy, so
#   downloads of the same request overlap
@pytest.mark.parametrize("data_files", ["delete", "refresh"])
def test_private_copies_download_in_parallel(tmp_path: Path, data_files: str) -> None:
    client = ParallelClient({})
    read_concurrently(tmp_path, data_files, client, [REQUEST, REQUEST])
    assert client.downloads == 2


# with data_files="keep" downloads of different requests overlap
def test_data_files_keep_downloads_different_requests_in_parallel(
    tmp_path: Path,
) -> None:
    client = ParallelClient({})
    requests = [REQUEST, REQUEST | {"dataset": "other-dataset"}]
    read_concurrently(tmp_path, "keep", client, requests)
    assert client.downloads == 2


# with data_files="keep" concurrent reads of the same request download it once
def test_data_files_keep_downloads_a_request_once(tmp_path: Path) -> None:
    class SlowClient(DummyClient):
        def download(self, result: Any, target: str | None = None) -> str:
            time.sleep(0.2)
            return super().download(result, target)

    client = SlowClient({})
    paths = read_concurrently(tmp_path, "keep", client, [REQUEST] * 4)
    assert client.downloads == 1
    assert len(set(paths)) == 1
