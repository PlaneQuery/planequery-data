import os
from datetime import date
from pathlib import Path

import polars as pl

os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[2]))

from data_profiling import cli
from data_profiling.cli import (
    find_outliers,
    get_daily_flight_counts,
    split_date_range,
)


def test_missing_and_corrupt_files_have_zero_flights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def path_for_day(*args) -> Path:
        return tmp_path / f"{args[1]}.parquet"

    monkeypatch.setattr(cli, "flight_path", path_for_day)
    (tmp_path / "2026-01-02.parquet").write_text("not parquet")
    pl.DataFrame({"flight": [1, 2]}).write_parquet(
        tmp_path / "2026-01-03.parquet"
    )

    assert get_daily_flight_counts(
        date(2026, 1, 1), date(2026, 1, 4)
    ) == {
        date(2026, 1, 1): 0,
        date(2026, 1, 2): 0,
        date(2026, 1, 3): 2,
    }


def test_find_outliers() -> None:
    counts = {
        date(2026, 1, 1): 98,
        date(2026, 1, 2): 100,
        date(2026, 1, 3): 102,
        date(2026, 1, 4): 0,
    }

    outliers, lower, upper = find_outliers(
        counts,
        low_sigma=2.0,
        high_sigma=3.0,
    )

    assert outliers == [(date(2026, 1, 4), 0)]
    assert lower == 96
    assert upper == 106


def test_high_counts_use_stricter_threshold() -> None:
    counts = {
        date(2026, 1, 1): 90,
        date(2026, 1, 2): 100,
        date(2026, 1, 3): 100,
        date(2026, 1, 4): 100,
        date(2026, 1, 5): 110,
        date(2026, 1, 6): 120,
    }

    outliers, _, upper = find_outliers(
        counts,
        low_sigma=2.0,
        high_sigma=3.0,
    )

    assert upper > 120
    assert outliers == []


def test_split_date_range_into_three_month_periods() -> None:
    assert split_date_range(date(2026, 1, 1), date(2026, 8, 15)) == [
        (date(2026, 1, 1), date(2026, 4, 1)),
        (date(2026, 4, 1), date(2026, 7, 1)),
        (date(2026, 7, 1), date(2026, 8, 15)),
    ]


def test_exactly_three_months_is_one_period() -> None:
    assert split_date_range(date(2026, 1, 1), date(2026, 4, 1)) == [
        (date(2026, 1, 1), date(2026, 4, 1))
    ]
