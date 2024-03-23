import hashlib
import logging
import os
from typing import Any

import attrs
import polytope.api

LOGGER = logging.getLogger(__name__)

CLIENT_KWARGS_DEFAULTS = {"quiet": True, "verbose": False}


@attrs.define
class PolytopeRequestClient:
    client_kwargs: dict[str, Any] = {}

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        target = hashlib.md5(str(request).encode("utf-8")).hexdigest() + ".grib"
        return {"request": request.copy(), "target": target}

    def get_filename(self, result: Any) -> str:
        return result["target"]  # type: ignore

    def download(self, result: Any, target: str | None = None) -> str:
        assert target is not None
        client = polytope.api.Client(**CLIENT_KWARGS_DEFAULTS | self.client_kwargs)
        client.retrieve("destination-earth", result["request"], target)
        if os.stat(target).st_size == 0:
            raise TypeError(f"polytope returned an empty file: {target}")
        return target
