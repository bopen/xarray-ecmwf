import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from xarray_ecmwf import engine_ecmwf

# several processes can build the same .zarr schema sample at once: each one
# writes its own copy under a temporary name, and the first copy renamed into
# place is the one everybody opens.
REQUEST = {"dataset": "dummy-dataset", "variable": ["dummy_variable"]}


# every download of the request gets the same file name, as with CDS
class DummyClient:
    def __init__(self, client_kwargs: dict[str, Any]) -> None:
        pass

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        return request

    def get_filename(self, result: Any) -> str:
        return "request.grib"

    def download(self, result: Any, target: str | None = None) -> str:
        assert target is not None
        Path(target).touch()
        return target


def open_dummy(path: str, **kwargs: Any) -> xr.Dataset:
    return xr.Dataset({"t": ("time", np.zeros(3))}, coords={"time": [0, 1, 2]})


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
