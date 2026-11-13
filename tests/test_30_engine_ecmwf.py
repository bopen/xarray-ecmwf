import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

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
        Path(target).touch()
        return target


# the dataset records the file it was opened from
def open_dummy(path: str, **kwargs: Any) -> xr.Dataset:
    return xr.Dataset(
        {"t": ("time", np.zeros(3))}, coords={"time": [0, 1, 2]}, attrs={"path": path}
    )


# a downloaded file that can't be opened, like a corrupt GRIB
def open_broken(path: str, **kwargs: Any) -> xr.Dataset:
    raise ValueError(f"can't open {path}")


# a dataset opened with an attribute telling which version of the data it holds
def open_version(version: int) -> Callable[..., xr.Dataset]:
    def open_dataset(path: str, **kwargs: Any) -> xr.Dataset:
        return open_dummy(path).assign_attrs(version=version)

    return open_dataset


def open_schema_dataset(
    cache_folder: Path,
    open_dataset: Callable[..., xr.Dataset] = open_dummy,
    **cache_kwargs: Any,
) -> xr.Dataset:
    cacher = engine_ecmwf.DatasetCacher(
        DummyClient({}), open_dataset, cache_folder=str(cache_folder), **cache_kwargs
    )
    with cacher.cached_schema_dataset(REQUEST) as ds:
        return ds


# several processes can build the same .zarr schema file at once: each one
# writes its own copy under a temporary name, and the first copy renamed into
# place is the one everybody opens.


def test_cached_schema_dataset_keeps_the_schema_file_saved_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    to_zarr = xr.Dataset.to_zarr

    def to_zarr_after_another_process(
        self: xr.Dataset, store: Any, *args: Any, **kw: Any
    ) -> Any:
        final_path = os.path.splitext(store)[0]
        # another process saves its complete schema file while this one is writing
        to_zarr(self.assign_attrs(saved_by="another process"), final_path, *args, **kw)
        return to_zarr(self, store, *args, **kw)

    monkeypatch.setattr(xr.Dataset, "to_zarr", to_zarr_after_another_process)

    ds = open_schema_dataset(tmp_path)

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
    with cacher.retrieve_once(REQUEST) as outer:
        with cacher.retrieve_once(REQUEST):
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
        with cacher.retrieve_once(REQUEST):
            pass

    assert os.listdir(tmp_path) == []


def test_cache_data_files_false_leaves_a_kept_file_alone(tmp_path: Path) -> None:
    client = DummyClient({})
    kept = tmp_path / "request.grib"
    kept.write_text("kept by a run with cache_data_files=True")
    cacher = engine_ecmwf.DatasetCacher(
        client, open_dummy, cache_data_files=False, cache_folder=str(tmp_path)
    )

    with cacher.retrieve_once(REQUEST):
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


# indexpath is a cfgrib option: other engines, e.g. h5netcdf, fail on it
@pytest.mark.parametrize("engine, indexpath", [("cfgrib", ""), ("h5netcdf", None)])
def test_open_dataset_passes_indexpath_only_to_cfgrib(
    tmp_path: Path, engine: str, indexpath: str | None
) -> None:
    indexpaths = []

    def open_recording_indexpath(path: str, **kwargs: Any) -> xr.Dataset:
        indexpaths.append(kwargs.get("indexpath"))
        return open_dummy(path)

    xr.open_dataset(
        REQUEST,  # type: ignore
        engine="ecmwf",
        request_client_class=DummyClient,
        cache_kwargs={"cache_folder": str(tmp_path), "cache_data_files": False},
        open_dataset_kwargs={"engine": engine},
        open_dataset=open_recording_indexpath,
    )

    assert indexpaths == [indexpath]


def test_cache_data_files_true_reuses_and_keeps_the_file(tmp_path: Path) -> None:
    client = DummyClient({})
    cacher = engine_ecmwf.DatasetCacher(client, open_dummy, cache_folder=str(tmp_path))

    with cacher.retrieve_once(REQUEST):
        pass
    with cacher.retrieve_once(REQUEST):
        pass

    assert client.downloads == 1
    assert os.listdir(tmp_path) == ["request.grib"]


# with override_schema_files each open builds the schema file again and puts it
# in place of the old one.


def test_override_schema_files_rebuilds_the_schema_file(tmp_path: Path) -> None:
    open_schema_dataset(tmp_path, open_version(1))

    ds = open_schema_dataset(tmp_path, open_version(2), override_schema_files=True)

    assert ds.attrs["version"] == 2
    # later opens without the override use the new one
    assert open_schema_dataset(tmp_path, open_version(3)).attrs["version"] == 2
    assert len([name for name in os.listdir(tmp_path) if ".zarr" in name]) == 1


def test_override_schema_files_repairs_a_broken_schema_file(tmp_path: Path) -> None:
    open_schema_dataset(tmp_path)
    # break the schema file: empty one of its chunks
    (chunk,) = tmp_path.glob("*.zarr/time/c/*")
    chunk.write_bytes(b"")
    with pytest.raises(RuntimeError, match="decompression"):
        open_schema_dataset(tmp_path)

    open_schema_dataset(tmp_path, override_schema_files=True)

    # later opens without the override use the repaired one
    assert open_schema_dataset(tmp_path)["time"].values.tolist() == [0, 1, 2]


# what another process does during the first open: the schema file is missing
# for an instant, or replaced as with override_schema_files
def make_it_missing(path: str) -> None:
    raise FileNotFoundError(path)


def replace_it(path: str) -> None:
    engine_ecmwf.robust_save_to_file(shutil.copytree, (path,), path, replace=True)


@pytest.mark.parametrize("another_process", [make_it_missing, replace_it])
def test_cached_schema_dataset_opens_again_a_schema_file_missing_or_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    another_process: Callable[[str], None],
) -> None:
    open_schema_dataset(tmp_path)
    open_dataset = xr.open_dataset
    opened = []

    def open_during_another_process(path: Any, **kwargs: Any) -> Any:
        opened.append(path)
        if len(opened) == 1:
            another_process(path)
        return open_dataset(path, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", open_during_another_process)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    open_schema_dataset(tmp_path)

    assert len(opened) == 2


def test_cached_schema_dataset_gives_up_on_a_schema_file_that_stays_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_schema_dataset(tmp_path)

    def open_missing(path: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError(path)

    sleeps: list[float] = []
    monkeypatch.setattr(xr, "open_dataset", open_missing)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="missing or replaced at each of 3 opens"):
        open_schema_dataset(tmp_path)
    # no wait after the last open
    assert sleeps == [1, 1]
