import logging
from typing import Any

import attrs
import cdsapi

SUPPORTED_DATASETS = {"reanalysis-era5-single-levels", "reanalysis-era5-land"}
LOGGER = logging.getLogger(__name__)


@attrs.define(slots=False)
class CdsapiRequestClient:
    client_kwargs: dict[str, Any]

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        request = self.request.copy()
        self.dataset = request.pop("dataset")
        if self.dataset not in SUPPORTED_DATASETS:
            LOGGER.warning(f"{self.dataset=} not supported")
        client = cdsapi.Client(**{"quiet": True, "retry_max": 1} | self.client_kwargs)
        return client.retrieve(self.dataset, request | {"format": "grib"})

    def get_filename(self, result: Any) -> str:
        return result.location.split("/")[-1]

    def download(self, result: Any, target: str | None = None) -> str:
        return result.download(target)
