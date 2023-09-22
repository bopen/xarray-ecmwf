import logging
from typing import Any

import attrs
import cdsapi

LOGGER = logging.getLogger(__name__)

SUPPORTED_DATASETS = {"reanalysis-era5-single-levels", "reanalysis-era5-land"}


@attrs.define(slots=False)
class CdsapiRequestClient:
    client_kwargs: dict[str, Any] = {"quiet": True, "retry_max": 1}

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        request = self.request.copy()
        dataset = request.pop("dataset")
        if dataset not in SUPPORTED_DATASETS:
            LOGGER.warning(f"{self.dataset=} not supported")
        client = cdsapi.Client(**self.client_kwargs)
        return client.retrieve(dataset, request | {"format": "grib"})

    def get_filename(self, result: Any) -> str:
        return result.location.split("/")[-1]

    def download(self, result: Any, target: str | None = None) -> str:
        return result.download(target)
