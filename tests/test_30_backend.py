import calendar
import itertools

import pandas as pd
import pytest

from xarray_ecmwf import client_common


class DummyRequestClient:
    def __init__(self, request, client_kwargs) -> None:
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
def test_build_chunk_date_requests(start_date, end_date, split_days) -> None:
    # date request
    request_chunks = {"day": split_days}
    request = {
        "date": [f"{start_date}/{end_date}"],
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
        (["2022"], None, None),
        (["2020"], None, None),  # leap year
        ([str(y) for y in range(2015, 2020)], None, None),  # consecutive years
        (["2010", "2015", "2020"], None, None),
        (["2010", "2015", "2020"], ["01", "02", "09"], None),
        (["2010", "2015", "2020"], ["01", "02", "09"], ["01", "29", "30", "31"]),
    ],
)
def test_build_chunk_ymd_requests(years, months, days) -> None:
    request_chunks = {"day": 1}
    request = {
        k: v
        for k, v in filter(
            lambda x: x[1], [("year", years), ("month", months), ("day", days)]
        )
    }

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

    request_chunks = {"month": 6}
    with pytest.raises(ValueError):
        client_common.build_chunk_ymd_requests(request, request_chunks)
