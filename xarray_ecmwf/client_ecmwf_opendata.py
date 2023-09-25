import hashlib
import logging
from typing import Any

import attrs
import ecmwf.opendata

LOGGER = logging.getLogger(__name__)


@attrs.define
class EcmwfOpendataRequestClient:
    client_kwargs: dict[str, Any] = {}

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        target = hashlib.md5(str(request).encode("utf-8")).hexdigest() + ".grib"
        return {"request": request.copy(), "target": target}

    def get_filename(self, result: Any) -> str:
        return result["target"]  # type: ignore

    def download(self, result: Any, target: str | None = None) -> str:
        source = result["request"].pop("source")
        client = ecmwf.opendata.Client(source=source, **self.client_kwargs)
        return client.retrieve(request=result["request"], target=target)
