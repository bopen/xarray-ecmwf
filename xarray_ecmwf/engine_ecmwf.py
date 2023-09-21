from typing import Any, Iterable

import xarray as xr


class ECMWFBackendEntrypoint(xr.backends.BackendEntrypoint):
    def open_dataset(  # type: ignore
        self,
        filename_or_obj: dict[str, Any],
        *,
        drop_variables: str | Iterable[str] | None = None,
        request_split: dict[str, int] = {"day": 1},
    ) -> xr.Dataset:
        pass

    def guess_can_open(self, filename_or_obj: Any) -> bool:
        return False
