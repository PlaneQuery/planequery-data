from __future__ import annotations

import argparse
import statistics
from calendar import monthrange
from datetime import date
from typing import Sequence

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
from flights.flight_source import flight_path
from flights.sources import ALGORITHM_NN_SOURCE
from planequery_data.utils import date_range


def add_months(day: date, months: int) -> date:
    month_index = day.year * 12 + day.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    return date(year, month, min(day.day, monthrange(year, month)[1]))


def split_date_range(
    start_date: date,
    end_date: date,
) -> list[tuple[date, date]]:
    """Split ``[start_date, end_date)`` into periods of at most three months."""
    periods = []
    period_start = start_date
    while period_start < end_date:
        period_end = min(add_months(period_start, 3), end_date)
        periods.append((period_start, period_end))
        period_start = period_end
    return periods


def get_daily_flight_counts(
    start_date: date,
    end_date: date,
) -> dict[date, int]:
    """Return counts for readable flight files in ``[start_date, end_date)``."""
    counts: dict[date, int] = {}
    for day in date_range(start_date, end_date):
        path = flight_path(ALGORITHM_NN_SOURCE, day, ADSBSource.ADSBLOL)
        if not path.is_file():
            continue
        try:
            counts[day] = int(
                pl.scan_parquet(path).select(pl.len()).collect().item()
            )
        except Exception:
            # A partial/corrupt Parquet file has no usable flight count.
            continue
    return counts


def find_outliers(
    counts: dict[date, int],
    low_sigma: float = 2.0,
    high_sigma: float = 3.0,
) -> tuple[list[tuple[date, int]], float, float]:
    """Return unusually low days and exceptionally high days.

    Empty days are tested as zero-count outliers but are excluded from the fit,
    so missing-data placeholders cannot pull the normal range toward zero.
    """
    normal_counts = [count for count in counts.values() if count > 0]
    if len(normal_counts) < 2:
        raise ValueError("at least two non-empty days are required")

    mean = statistics.mean(normal_counts)
    stddev = statistics.stdev(normal_counts)
    lower = mean - low_sigma * stddev
    upper = mean + high_sigma * stddev
    outliers = sorted(
        (day, count)
        for day, count in counts.items()
        if count < lower or count > upper
    )
    return outliers, lower, upper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print days with outlier algorithm-nn flight counts."
    )
    parser.add_argument("start_date", type=date.fromisoformat)
    parser.add_argument("end_date", type=date.fromisoformat)
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Standard deviations for low-count outliers (default: 2.0).",
    )
    parser.add_argument(
        "--high-sigma",
        type=float,
        default=3.0,
        help="Standard deviations for exceptionally high counts (default: 3.0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.end_date <= args.start_date:
        raise SystemExit("end_date must be after start_date")
    if args.sigma <= 0:
        raise SystemExit("sigma must be positive")
    if args.high_sigma <= 0:
        raise SystemExit("high-sigma must be positive")

    periods = split_date_range(args.start_date, args.end_date)
    for index, (period_start, period_end) in enumerate(periods):
        if index:
            print()
        print(f"Period: {period_start} to {period_end} (end exclusive)")

        counts = get_daily_flight_counts(period_start, period_end)
        try:
            outliers, lower, upper = find_outliers(
                counts,
                low_sigma=args.sigma,
                high_sigma=args.high_sigma,
            )
        except ValueError as exc:
            print(f"Unable to assess: {exc}")
            continue

        print(f"Low-count threshold: below {lower:,.0f} flights")
        print(f"Exceptional high-count threshold: above {upper:,.0f} flights")
        for day, count in outliers:
            print(f"{day}: {count:,} flights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
