import contextlib
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
    retrieve_lock = contextlib.nullcontext
    download_lock = contextlib.nullcontext

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        path = hashlib.md5(str(request).encode("utf-8")).hexdigest() + ".grib"
        client = polytope.api.Client(**CLIENT_KWARGS_DEFAULTS | self.client_kwargs)
        # polytope-server appears not to support concurrent resolution=high requests
        with self.retrieve_lock:
            res = client.retrieve("destination-earth", request, path, asynchronous=True)
            # the following doesn't downloads, it just waits until the result is ready
            res[0].download(pointer=True)
        return res[0]

    def get_filename(self, result: Any) -> str:
        return result.output_file  # type: ignore

    def download(self, result: Any, target: str | None = None) -> str:
        assert target is not None
        # this is an optimization of network bandwidth
        with self.download_lock:
            result.download(output_file=target)
        if os.stat(target).st_size == 0:
            request = result.describe()["user_request"]
            raise TypeError(f"polytope returned an empty file: {target} for {request}")
        return target
