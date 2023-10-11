import calendar
import itertools
from typing import Any

import pandas as pd
import pytest

from xarray_ecmwf import client_common

ALL_MONTHS = [f"{m:02}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02}" for d in range(1, 32)]
ALL_TIMES = [f"{t:02}:00" for t in range(24)]


class DummyRequestClient:
    def __init__(self, request: dict[str, Any], client_kwargs: dict[str, Any]) -> None:
        pass

    def retrieve(self) -> None:
        pass

    def get_filename(self) -> str:
        return "dummy"

    def download(self, target: str | None = None) -> str:
        return target or "dummy"


@pytest.mark.parametrize(
    "start_date, end_date, split_days",
    [
        ("2023-06-01", "2023-07-29", 45),  # stop date before
        ("2023-06-10", "2023-07-15", 45),  # smaller chunk
        ("2023-06-01", "2023-07-15", 45),  # perfect chunk
    ],
)
def test_build_chunk_date_requests(
    start_date: str, end_date: str, split_days: int
) -> None:
    # date request
    request_chunks = {"day": split_days}
    request = {
        "date": [f"{start_date}/{end_date}"],
        "time": ALL_TIMES,
    }

    (
        time,
        time_chunk,
        time_chunk_requests,
    ) = client_common.build_chunk_date_requests(request, request_chunks)

    diff = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    assert len(time) == 24 * (diff + 1)  # 24h default value
    assert time_chunk == 24 * split_days
    assert len(time_chunk_requests) == 1 + diff // 45


@pytest.mark.parametrize(
    "years, months, days",
    [
        (["2022"], ALL_MONTHS, ALL_DAYS),
        (["2020"], ALL_MONTHS, ALL_DAYS),  # leap year
        (
            [str(y) for y in range(2015, 2020)],
            ALL_MONTHS,
            ALL_DAYS,
        ),  # consecutive years
        (["2010", "2015", "2020"], ALL_MONTHS, ALL_DAYS),
        (["2010", "2015", "2020"], ["01", "02", "09"], ALL_DAYS),
        (["2010", "2015", "2020"], ["01", "02", "09"], ["01", "29", "30", "31"]),
    ],
)
def test_build_chunk_ymd_requests(
    years: list[str], months: list[str], days: list[str]
) -> None:
    request_chunks = {"day": 1}
    request = {
        k: v
        for k, v in filter(
            lambda x: x[1], [("year", years), ("month", months), ("day", days)]
        )
    }
    request["time"] = ALL_TIMES

    (
        time,
        time_chunk,
        time_chunk_requests,
    ) = client_common.build_chunk_ymd_requests(request, request_chunks)
    total_days = 0
    for year, month, day in itertools.product(
        map(int, request["year"]),
        map(int, request["month"]),
        map(int, request["day"]),
    ):
        if day <= calendar.monthrange(year, month)[1]:
            total_days += 1

    assert len(time) == 24 * total_days  # 24h default value
    assert time_chunk == 24
    assert len(time_chunk_requests) == total_days

    request_chunks = {"day": 45}
    with pytest.raises(ValueError):
        client_common.build_chunk_ymd_requests(request, request_chunks)


def test_build_chunk_request() -> None:
    coord, chunk, chunk_request = client_common.build_chunks_header_requests(
        dim="x",
        request={"x": ["a", "b", "c", "d", "e"], "y": [1]},
        request_chunks={"x": 2},
        dtype="str",
    )
    assert chunk == 2
    assert chunk_request[0][0] == 0
    assert chunk_request[1][0] == 2
    assert chunk_request[2][0] == 4
    assert chunk_request[0][1] == {"x": ["a", "b"]}
    assert chunk_request[1][1] == {"x": ["c", "d"]}
    assert chunk_request[2][1] == {"x": ["e"]}

    coord, chunk, chunk_request = client_common.build_chunks_header_requests(
        dim="x",
        request={"x": ["a", "b", "c", "d", "e", "f"], "y": [1]},
        request_chunks={},
        dtype="str",
    )
    assert chunk == 6
    assert chunk_request[0][0] == 0
    assert chunk_request[0][1] == {"x": ["a", "b", "c", "d", "e", "f"]}
