import hashlib
import logging
from typing import Any

import attrs
import polytope.api

LOGGER = logging.getLogger(__name__)


@attrs.define
class PolytopeRequestClient:
    client_kwargs: dict[str, Any] = {}

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        target = hashlib.md5(str(request).encode("utf-8")).hexdigest() + ".grib"
        return {"request": request.copy(), "target": target}

    def get_filename(self, result: Any) -> str:
        return result["target"]  # type: ignore

    def download(self, result: Any, target: str | None = None) -> str:
        client = polytope.api.Client(**self.client_kwargs)
        return client.retrieve("destination-earth", result["request"], target)  # type: ignore
