from datetime import date

import pytest

from planequery_data.utils import date_range


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
