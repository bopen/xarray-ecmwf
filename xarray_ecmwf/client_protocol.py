from typing import Any, Protocol


class RequestClientProtocol(Protocol):
    def __init__(self, client_kwargs: dict[str, Any]) -> None:
        ...

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> Any:
        ...

    def get_filename(self, result: Any) -> str:
        ...

    def download(self, result: Any, target: str | None = None) -> str:
        ...


class RequestChunkerProtocol(Protocol):
    def __init__(self, request: dict[str, Any], request_chunks: dict[str, Any]) -> None:
        ...

    def get_coords_attrs_and_dtype(
        self,
        request_client: RequestClientProtocol | None = None,
        cache_kwargs: dict[str, Any] = {"cache_file": True},
    ):
        ...

    def get_variables(self) -> list[str]:
        ...

    def get_chunks(self) -> dict[str, Any]:
        ...
