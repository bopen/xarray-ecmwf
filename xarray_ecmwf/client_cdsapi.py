import logging
from typing import Any

import attrs
import cdsapi

from . import client_protocol

LOGGER = logging.getLogger(__name__)

SUPPORTED_DATASETS = {"reanalysis-era5-single-levels", "reanalysis-era5-land"}


@attrs.define
class CdsapiRequestClient:
    client_kwargs: dict[str, Any] = {"quiet": True, "retry_max": 1}

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        request = request.copy()
        dataset = request.pop("dataset")
        if dataset not in SUPPORTED_DATASETS:
            LOGGER.warning(f"{dataset=} not supported")
        client = cdsapi.Client(**self.client_kwargs)
        return client.retrieve(dataset, request | {"format": "grib"})

    def get_filename(self, result: Any) -> str:
        return result.location.split("/")[-1]  # type: ignore

    def download(self, result: Any, target: str | None = None) -> str:
        return result.download(target)  # type: ignore


@attrs.define(slots=False)
class CdsapiRequestChunker:
    request: dict[str, Any]
    request_chunks: dict[str, Any]

    def compute_request_coords(self) -> dict[str, Any]:
        return {}

    def get_coords_attrs_and_dtype(
        self, dataset_cacher=client_protocol.DatasetCacherProtocol
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
        coords = self.compute_request_coords()
        with dataset_cacher.retrieve(self.request) as sample_ds:
            da = list(sample_ds.data_vars.values())[0]
            coords["lat"] = ("lat", da.lat.values, da.lat.attrs)
            coords["lon"] = ("lon", da.lon.values, da.lon.attrs)
            return coords, sample_ds.attrs, da.attrs, da.dtype
