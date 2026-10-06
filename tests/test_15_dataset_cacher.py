from typing import Any

import numpy as np
import xarray as xr

from xarray_ecmwf.engine_ecmwf import DatasetCacher


class DummyRequestClient:
    def __init__(self) -> None:
        self.submit_calls = 0

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        self.submit_calls += 1
        return {"request": request}

    def get_filename(self, result: Any) -> str:
        return "unused-by-request-hash-cache-key.grib"

    def download(self, result: Any, target: str | None = None) -> str:
        xr.Dataset({"t": ("x", np.arange(3, dtype="float32"))}).to_zarr(target, mode="w")
        return target  # type: ignore


def _open_zarr(path: str) -> xr.Dataset:
    return xr.open_dataset(path, engine="zarr")


def test_retrieve_once_checks_the_local_cache_before_contacting_the_client(tmp_path: Any) -> None:
    client = DummyRequestClient()
    cacher = DatasetCacher(client, open_dataset=_open_zarr, cache_folder=str(tmp_path))
    request = {"dataset": "reanalysis-era5-pressure-levels", "day": "01"}

    with cacher.retrieve_once(request) as ds:
        assert "t" in ds.data_vars
    assert client.submit_calls == 1

    with cacher.retrieve_once(request) as ds:
        assert "t" in ds.data_vars
    assert client.submit_calls == 1


def test_retrieve_once_still_submits_a_genuinely_different_request(tmp_path: Any) -> None:
    client = DummyRequestClient()
    cacher = DatasetCacher(client, open_dataset=_open_zarr, cache_folder=str(tmp_path))

    with cacher.retrieve_once({"dataset": "reanalysis-era5-pressure-levels", "day": "01"}):
        pass
    with cacher.retrieve_once({"dataset": "reanalysis-era5-pressure-levels", "day": "02"}):
        pass

    assert client.submit_calls == 2


def test_retrieve_once_with_cache_file_false_still_skips_a_within_context_resubmit(tmp_path: Any) -> None:
    # override_cache_file only controls whether the file is deleted after the context manager exits 
    # it must not affect whether an already-present file is reused.
    client = DummyRequestClient()
    cacher = DatasetCacher(client, open_dataset=_open_zarr, cache_folder=str(tmp_path), cache_file=False)
    request = {"dataset": "reanalysis-era5-pressure-levels", "day": "01"}

    with cacher.retrieve_once(request):
        with cacher.retrieve_once(request):
            pass

    assert client.submit_calls == 1
