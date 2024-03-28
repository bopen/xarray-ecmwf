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
        path = hashlib.md5(str(request).encode("utf-8")).hexdigest() + ".grib"
        client = polytope.api.Client(**CLIENT_KWARGS_DEFAULTS | self.client_kwargs)
        res = client.retrieve("destination-earth", request, path, asynchronous=True)[0]
        # the following doesn't downloads, it just waits until the result is ready
        res.download(pointer=True)
        return res

    def get_filename(self, result: Any) -> str:
        return result.output_file  # type: ignore

    def download(self, result: Any, target: str | None = None) -> str:
        assert target is not None
        result.download(output_file=target)
        if os.stat(target).st_size == 0:
            request = result.describe()["user_request"]
            raise TypeError(f"polytope returned an empty file: {target} for {request}")
        return target
