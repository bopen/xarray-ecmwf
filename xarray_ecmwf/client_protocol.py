from typing import Any, Protocol


class RequestClientProtocol(Protocol):
    def __init__(self, client_kwargs: dict[str, Any]) -> None:
        ...

    def submit_and_wait_on_result(self, request: dict[str, Any]) -> str:
        ...

    def get_filename(self, request_uid: str) -> str:
        ...

    def download(self, request_uid: str, target: str | None = None) -> str:
        ...


RequestClientType = type[RequestClientProtocol]


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

    def a(self):
        ...

    def get_chunks(self) -> dict[str, Any]:
        ...
