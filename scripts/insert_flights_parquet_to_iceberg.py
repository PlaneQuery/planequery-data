"""Insert existing flights parquet files into the flights Iceberg table.

Usage:
    python scripts/insert_flights_parquet_to_iceberg.py
    python scripts/insert_flights_parquet_to_iceberg.py --start-date 2026-04-01 --end-date 2026-05-01
    python scripts/insert_flights_parquet_to_iceberg.py --prod
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_engineering.flights.flight_type import flights_parquet_path
from data_engineering.utils import FLIGHTS_VERSION, OUTPUT_DIR


FLIGHTS_COLUMNS = [
    "icao",
    "callsign",
    "registration",
    "takeoff_time",
    "takeoff_airport_ident",
    "landing_time",
    "landing_airport_ident",
    "pia",
    "ladd",
    "military",
    "interesting",
    "aircraft_type",
    "owner",
    "aircraft_description",
    "category",
]

BOOLEAN_COLUMNS = ["pia", "ladd", "military", "interesting"]


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> list[date]:
    if end <= start:
        raise ValueError(f"end_date ({end}) must be after start_date ({start})")

    days = []
    current = start
    while current < end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _discover_flight_dates() -> list[date]:
    flights_dir = OUTPUT_DIR / "data" / "flights" / FLIGHTS_VERSION
    if not flights_dir.exists():
        raise FileNotFoundError(f"Flights directory does not exist: {flights_dir}")

    dates: list[date] = []
    for parquet_path in flights_dir.glob("year=*/month=*/day=*/part-0.parquet"):
        try:
            year = int(parquet_path.parents[2].name.removeprefix("year="))
            month = int(parquet_path.parents[1].name.removeprefix("month="))
            day = int(parquet_path.parents[0].name.removeprefix("day="))
        except ValueError as exc:
            raise ValueError(f"Could not parse date partitions from {parquet_path}") from exc
        dates.append(date(year, month, day))

    return sorted(set(dates))


def _existing_rows_for_day(spark, table: str, target_date: date) -> int:
    next_day = target_date + timedelta(days=1)
    return spark.sql(
        f"""
        SELECT COUNT(*) AS row_count
        FROM {table}
        WHERE takeoff_time >= TIMESTAMP '{target_date:%Y-%m-%d} 00:00:00'
          AND takeoff_time < TIMESTAMP '{next_day:%Y-%m-%d} 00:00:00'
        """
    ).collect()[0]["row_count"]


def _insert_flights_file(spark, table: str, parquet_path: Path) -> int:
    df = spark.read.parquet(str(parquet_path)).select(*FLIGHTS_COLUMNS)
    df = df.fillna(False, subset=BOOLEAN_COLUMNS)
    row_count = df.count()
    if row_count == 0:
        print(f"No rows to insert from {parquet_path}")
        return 0

    df.coalesce(2).writeTo(table).append()
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insert flights parquet files into the flights Iceberg table."
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        help="Start date, inclusive, in YYYY-MM-DD format. Defaults to first discovered parquet date.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        help="End date, exclusive, in YYYY-MM-DD format. Defaults to one day after the last discovered parquet date.",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Insert into planequery_lake_prod.flights instead of planequery_lake_dev.flights.",
    )
    args = parser.parse_args()

    discovered_dates = _discover_flight_dates()
    if not discovered_dates:
        raise FileNotFoundError(
            f"No flights parquet files found under {OUTPUT_DIR / 'data' / 'flights' / FLIGHTS_VERSION}"
        )

    start_date = args.start_date or discovered_dates[0]
    end_date = args.end_date or discovered_dates[-1] + timedelta(days=1)
    candidate_dates = set(_date_range(start_date, end_date))
    dates = [day for day in discovered_dates if day in candidate_dates]
    if not dates:
        raise FileNotFoundError(f"No flights parquet files found in [{start_date}, {end_date})")

    database = "planequery_lake_prod" if args.prod else "planequery_lake_dev"
    table = f"glue.{database}.flights"
    print(f"Inserting {len(dates)} flights parquet files into {table}")

    from data_engineering.spark import get_spark_session

    spark = get_spark_session()
    inserted_rows = 0
    try:
        for target_date in dates:
            parquet_path = flights_parquet_path(target_date)
            if not parquet_path.exists():
                raise FileNotFoundError(f"Expected parquet file is missing: {parquet_path}")

            existing_row_count = _existing_rows_for_day(spark, table, target_date)
            print(f"{target_date}: {existing_row_count} existing rows in {table}")
            if existing_row_count > 0:
                print(f"Skipping {target_date} to avoid duplicates. Remove existing rows for this date to re-insert.")
                # raise RuntimeError(f"Data for {target_date} already exists in Iceberg.")

            row_count = _insert_flights_file(spark, table, parquet_path)
            inserted_rows += row_count
            print(f"{target_date}: inserted {row_count} rows from {parquet_path}")
    finally:
        spark.stop()

    print(
        f"Done. Inserted {inserted_rows} rows across {len(dates)} dates."
    )


if __name__ == "__main__":
    main()
