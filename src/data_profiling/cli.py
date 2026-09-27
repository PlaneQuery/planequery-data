from __future__ import annotations

import argparse
import statistics
from datetime import date
from typing import Sequence

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
from flights.flight_source import flight_path
from flights.sources import ALGORITHM_NN_SOURCE
from planequery_data.utils import date_range


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
    sigma: float = 2.0,
) -> tuple[list[tuple[date, int]], float, float]:
    """Return days outside mean +/- ``sigma`` standard deviations.

    Empty days are tested as zero-count outliers but are excluded from the fit,
    so missing-data placeholders cannot pull the normal range toward zero.
    """
    normal_counts = [count for count in counts.values() if count > 0]
    if len(normal_counts) < 2:
        raise ValueError("at least two non-empty days are required")

    mean = statistics.mean(normal_counts)
    stddev = statistics.stdev(normal_counts)
    lower = mean - sigma * stddev
    upper = mean + sigma * stddev
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
        help="Standard deviations from the mean (default: 2.0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.end_date <= args.start_date:
        raise SystemExit("end_date must be after start_date")
    if args.sigma <= 0:
        raise SystemExit("sigma must be positive")

    counts = get_daily_flight_counts(args.start_date, args.end_date)
    try:
        outliers, lower, upper = find_outliers(counts, args.sigma)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Normal range: {lower:,.0f} to {upper:,.0f} flights")
    for day, count in outliers:
        print(f"{day}: {count:,} flights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
