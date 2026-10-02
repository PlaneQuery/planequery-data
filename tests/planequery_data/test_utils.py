from datetime import date

import pytest

from planequery_data.utils import date_range, month_start_range


def test_date_range_is_end_exclusive():
    assert list(date_range(date(2026, 4, 1), date(2026, 4, 3))) == [
        date(2026, 4, 1),
        date(2026, 4, 2),
    ]


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        (date(2026, 4, 1), date(2026, 4, 1)),
        (date(2026, 4, 2), date(2026, 4, 1)),
    ],
)
def test_date_range_requires_end_after_start(start_date, end_date):
    with pytest.raises(AssertionError):
        list(date_range(start_date, end_date))


def test_month_start_range_is_start_inclusive_and_end_exclusive():
    assert list(month_start_range(date(2025, 12, 1), date(2026, 3, 1))) == [
        date(2025, 12, 1),
        date(2026, 1, 1),
        date(2026, 2, 1),
    ]


def test_month_start_range_skips_partial_start_month():
    assert list(month_start_range(date(2026, 1, 15), date(2026, 4, 20))) == [
        date(2026, 2, 1),
        date(2026, 3, 1),
        date(2026, 4, 1),
    ]


def test_month_start_range_can_be_empty():
    assert list(month_start_range(date(2026, 1, 15), date(2026, 2, 1))) == []


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        (date(2026, 4, 1), date(2026, 4, 1)),
        (date(2026, 4, 2), date(2026, 4, 1)),
    ],
)
def test_month_start_range_requires_end_after_start(start_date, end_date):
    with pytest.raises(AssertionError):
        list(month_start_range(start_date, end_date))
