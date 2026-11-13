import os
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


# with cache_data_files=False each read downloads a file of its own and deletes only
# that one, so readers of the same request don't delete each other's files.


def test_cache_data_files_false_readers_keep_their_own_files(tmp_path: Path) -> None:
    client = DummyClient({})
    cacher = engine_ecmwf.DatasetCacher(
        client, open_dummy, cache_data_files=False, cache_folder=str(tmp_path)
    )

    # two readers of the same request at the same time
    with cacher.open_data_file_and_clean(REQUEST) as outer:
        with cacher.open_data_file_and_clean(REQUEST):
            pass
        assert os.path.exists(outer.attrs["path"])

    assert client.downloads == 2
    assert os.listdir(tmp_path) == []


def test_cache_data_files_false_removes_a_file_that_cannot_be_opened(
    tmp_path: Path,
) -> None:
    cacher = engine_ecmwf.DatasetCacher(
        DummyClient({}), open_broken, cache_data_files=False, cache_folder=str(tmp_path)
    )

    with pytest.raises(ValueError):
        with cacher.open_data_file_and_clean(REQUEST):
            pass

    assert os.listdir(tmp_path) == []


def test_cache_data_files_false_leaves_a_kept_file_alone(tmp_path: Path) -> None:
    client = DummyClient({})
    kept = tmp_path / "request.grib"
    kept.write_text("kept by a run with cache_data_files=True")
    cacher = engine_ecmwf.DatasetCacher(
        client, open_dummy, cache_data_files=False, cache_folder=str(tmp_path)
    )

    with cacher.open_data_file_and_clean(REQUEST):
        pass

    assert client.downloads == 1
    assert kept.read_text() == "kept by a run with cache_data_files=True"


def test_open_dataset_names_the_new_key_for_cache_file() -> None:
    with pytest.raises(TypeError, match="renamed to 'cache_data_files'"):
        xr.open_dataset(
            REQUEST,  # type: ignore
            engine="ecmwf",
            request_client_class=DummyClient,
            cache_kwargs={"cache_file": False},
        )


# a private data file is read once, so cfgrib writes no index for it. indexpath is
# a cfgrib option: other engines, e.g. h5netcdf, fail on it
@pytest.mark.parametrize(
    "engine, cache_kwargs, indexpath",
    [
        ("cfgrib", {"cache_data_files": False}, ""),
        ("h5netcdf", {"cache_data_files": False}, None),
        ("cfgrib", {"override_data_files": True}, ""),
    ],
)
def test_open_dataset_passes_indexpath_only_to_cfgrib(
    tmp_path: Path, engine: str, cache_kwargs: dict[str, Any], indexpath: str | None
) -> None:
    indexpaths = []

    def open_recording_indexpath(path: str, **kwargs: Any) -> xr.Dataset:
        indexpaths.append(kwargs.get("indexpath"))
        return open_dummy(path)

    xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_client_class=DummyClient,
        cache_kwargs={"cache_folder": str(tmp_path), **cache_kwargs},
        open_dataset_kwargs={"engine": engine},
        open_dataset=open_recording_indexpath,
    )

    assert indexpaths == [indexpath]


def test_cache_data_files_true_reuses_and_keeps_the_file(tmp_path: Path) -> None:
    client = DummyClient({})
    cacher = engine_ecmwf.DatasetCacher(client, open_dummy, cache_folder=str(tmp_path))

    with cacher.open_data_file_and_clean(REQUEST):
        pass
    with cacher.open_data_file_and_clean(REQUEST):
        pass

    assert client.downloads == 1
    assert os.listdir(tmp_path) == ["request.grib"]


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
    cacher = engine_ecmwf.DatasetCacher(client, open_dummy, cache_folder=str(tmp_path))

    with cacher.open_data_file_and_clean(REQUEST):
        pass

    assert client.downloads == 2
    assert os.listdir(tmp_path) == ["request.grib"]


def test_open_data_file_and_clean_does_not_try_again_an_error_inside_the_with_block(
    tmp_path: Path,
) -> None:
    client = DummyClient({})
    cacher = engine_ecmwf.DatasetCacher(
        client, open_dummy, cache_data_files=False, cache_folder=str(tmp_path)
    )

    with pytest.raises(RuntimeError, match="raised while loading the values"):
        with cacher.open_data_file_and_clean(REQUEST):
            raise RuntimeError("raised while loading the values")

    assert client.downloads == 1


# with override_data_files each read downloads the data file again, reads its own
# copy, and then puts it into the cache in place of the old one.


def test_override_data_files_downloads_again_and_refreshes_the_cache(
    tmp_path: Path,
) -> None:
    client = DummyClient({})
    cached = tmp_path / "request.grib"
    cached.write_text("cached by an earlier run")
    (tmp_path / "request.grib.5b7b6.idx").write_text("cfgrib index of the old file")
    cacher = engine_ecmwf.DatasetCacher(
        client, open_dummy, cache_folder=str(tmp_path), override_data_files=True
    )

    with cacher.open_data_file_and_clean(REQUEST) as ds:
        assert ds.attrs["path"] != str(cached)

    assert client.downloads == 1
    assert cached.read_text() == "download 1"
    assert os.listdir(tmp_path) == ["request.grib"]


def test_override_data_files_readers_keep_their_own_files(tmp_path: Path) -> None:
    client = DummyClient({})
    cacher = engine_ecmwf.DatasetCacher(
        client, open_dummy, cache_folder=str(tmp_path), override_data_files=True
    )

    # two readers of the same request at the same time
    with cacher.open_data_file_and_clean(REQUEST) as outer:
        with cacher.open_data_file_and_clean(REQUEST):
            pass
        assert os.path.exists(outer.attrs["path"])

    assert client.downloads == 2
    assert os.listdir(tmp_path) == ["request.grib"]


def test_override_data_files_keeps_the_cached_file_if_the_new_one_is_broken(
    tmp_path: Path,
) -> None:
    cached = tmp_path / "request.grib"
    cached.write_text("cached by an earlier run")
    cacher = engine_ecmwf.DatasetCacher(
        DummyClient({}),
        open_broken,
        cache_folder=str(tmp_path),
        override_data_files=True,
    )

    with pytest.raises(ValueError):
        with cacher.open_data_file_and_clean(REQUEST):
            pass

    assert cached.read_text() == "cached by an earlier run"
    assert os.listdir(tmp_path) == ["request.grib"]


def test_override_data_files_keeps_the_read_if_the_cache_cannot_be_updated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cached = tmp_path / "request.grib"
    cached.write_text("cached by an earlier run")
    cacher = engine_ecmwf.DatasetCacher(
        DummyClient({}),
        open_dummy,
        cache_folder=str(tmp_path),
        override_data_files=True,
    )

    def replace_denied(src: str, dst: str) -> None:
        raise PermissionError(f"can't replace {dst}")

    monkeypatch.setattr(os, "replace", replace_denied)

    with cacher.open_data_file_and_clean(REQUEST):
        pass

    assert "failed to replace" in caplog.text
    assert cached.read_text() == "cached by an earlier run"
    assert os.listdir(tmp_path) == ["request.grib"]


def test_cache_data_files_true_fails_a_read_of_a_file_replaced_meanwhile(
    tmp_path: Path,
) -> None:
    cached = tmp_path / "request.grib"
    cached.write_text("cached by an earlier run")
    cacher = engine_ecmwf.DatasetCacher(
        DummyClient({}), open_dummy, cache_folder=str(tmp_path)
    )

    with pytest.raises(RuntimeError, match="replaced during the read"):
        with cacher.open_data_file_and_clean(REQUEST):
            # another run with override_data_files puts its new file in place
            new = tmp_path / "request-1a2b3c4d.grib"
            new.write_text("downloaded by the other run")
            os.replace(new, cached)


def test_override_data_files_needs_cache_data_files() -> None:
    with pytest.raises(ValueError, match="needs cache_data_files"):
        engine_ecmwf.DatasetCacher(
            DummyClient({}), cache_data_files=False, override_data_files=True
        )
