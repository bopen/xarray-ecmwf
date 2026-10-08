import calendar
import itertools
from typing import Any

import numpy as np
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
        )  # why this filter if they are never empty?
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
def test_build_chunk_ymd_year_requests(
    years: list[str], months: list[str], days: list[str]
) -> None:
    request_chunks = {"year": 1}
    request = {
        "year": years,
        "month": months,
        "day": days,
        "time": ALL_TIMES,
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
    assert len(time_chunk_requests) == len(years)
    assert isinstance(time_chunk, tuple)
    assert sum(time_chunk) == len(time)

    offset = 0
    for (start, chunk_request), year_str, size in zip(
        time_chunk_requests, years, time_chunk
    ):
        assert start == offset
        assert chunk_request == {"year": [year_str]}
        offset += size


@pytest.mark.parametrize(
    "year_chunk, expected_year_groups",
    [
        (1, [["2015"], ["2016"], ["2017"]]),
        (2, [["2015", "2016"], ["2017"]]),  # leftover chunk
        (5, [["2015", "2016", "2017"]]),  # N larger than n years
    ],
)
def test_build_chunk_ymd_year_requests_groups_years(
    year_chunk: int, expected_year_groups: list[list[str]]
) -> None:
    request = {
        "year": ["2015", "2016", "2017"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
    }  # one time stamp per year
    request_chunks = {"year": year_chunk}
    time, time_chunk, time_chunk_requests = client_common.build_chunk_ymd_requests(
        request, request_chunks
    )
    assert len(time) == 3
    assert [chunk_request["year"] for _, chunk_request in time_chunk_requests] == (
        expected_year_groups
    )
    assert time_chunk == tuple(len(group) for group in expected_year_groups)


def _monthly_year_request() -> dict[str, Any]:
    return {
        "year": ["2024", "2025", "2026"],
        "month": [f"{month:02}" for month in range(1, 13)],
        "day": ["01"],
        "time": ["00:00"],
    }


def test_expected_end_date_trims_partial_last_year() -> None:
    time, time_chunk, time_chunk_requests = client_common.build_time_chunk_requests(
        _monthly_year_request(),
        {"year": 1},
        expected_end_date="2026-07-01",
    )
    assert len(time) == 12 + 12 + 7
    assert time[-1] == np.datetime64("2026-07-01T00:00", "ns")
    assert time_chunk == (12, 12, 7)
    assert [chunk_request["year"] for _, chunk_request in time_chunk_requests] == [
        ["2024"],
        ["2025"],
        ["2026"],
    ]


def test_expected_end_date_drops_years_after_the_cutoff() -> None:
    time, time_chunk, time_chunk_requests = client_common.build_time_chunk_requests(
        _monthly_year_request(),
        {"year": 1},
        expected_end_date="2025-12-01",
    )
    assert len(time) == 24
    assert time[-1] == np.datetime64("2025-12-01T00:00", "ns")
    assert time_chunk == (12, 12)
    assert [chunk_request["year"] for _, chunk_request in time_chunk_requests] == [
        ["2024"],
        ["2025"],
    ]


def test_expected_end_date_shrinks_a_multi_year_chunk() -> None:
    time, time_chunk, time_chunk_requests = client_common.build_time_chunk_requests(
        _monthly_year_request(),
        {"year": 2},
        expected_end_date="2025-06-01",
    )
    assert len(time) == 12 + 6
    assert time[-1] == np.datetime64("2025-06-01T00:00", "ns")
    assert time_chunk == (18,)
    assert time_chunk_requests == [(0, {"year": ["2024", "2025"]})]


def test_expected_end_date_on_a_chunk_boundary_keeps_equal_day_chunks() -> None:
    request = {
        "year": ["2024"],
        "month": ["01", "02"],
        "day": ["01", "02"],
        "time": ["00:00"],
    }
    time, time_chunk, time_chunk_requests = client_common.build_time_chunk_requests(
        request,
        {"day": 1},
        expected_end_date="2024-01-02",
    )
    assert list(time) == [
        np.datetime64("2024-01-01T00:00", "ns"),
        np.datetime64("2024-01-02T00:00", "ns"),
    ]
    assert time_chunk == 1
    assert len(time_chunk_requests) == 2


def test_expected_end_date_shrinks_the_last_equal_sized_chunk() -> None:
    request = {
        "year": ["2024"],
        "month": ["01"],
        "day": ["01", "02"],
        "time": ["00:00", "12:00"],
    }
    time, time_chunk, time_chunk_requests = client_common.build_time_chunk_requests(
        request,
        {"day": 1},
        expected_end_date="2024-01-02T00:00",
    )
    assert list(time) == [
        np.datetime64("2024-01-01T00:00", "ns"),
        np.datetime64("2024-01-01T12:00", "ns"),
        np.datetime64("2024-01-02T00:00", "ns"),
    ]
    assert time_chunk == (2, 1)
    assert [chunk_request["day"] for _, chunk_request in time_chunk_requests] == [
        "01",
        "02",
    ]


def test_expected_end_date_past_the_last_timestamp_is_a_no_op() -> None:
    full, full_chunk, full_requests = client_common.build_time_chunk_requests(
        _monthly_year_request(), {"year": 1}
    )
    trimmed, trimmed_chunk, trimmed_requests = client_common.build_time_chunk_requests(
        _monthly_year_request(), {"year": 1}, expected_end_date="2027-01-01"
    )
    assert list(trimmed) == list(full)
    assert trimmed_chunk == full_chunk
    assert trimmed_requests == full_requests


def test_expected_end_date_before_the_first_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="before the first timestamp"):
        client_common.build_time_chunk_requests(
            _monthly_year_request(),
            {"year": 1},
            expected_end_date="2020-01-01",
        )


def test_build_chunk_ymd_year_requests_invalid() -> None:
    request = {
        "year": ["2022"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
    }
    with pytest.raises(ValueError):
        client_common.build_chunk_ymd_requests(request, {"year": 0})


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
