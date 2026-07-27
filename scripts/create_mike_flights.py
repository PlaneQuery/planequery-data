import argparse
from pathlib import Path

from data_engineering.flights.mike import (
    CSV_TIMESTAMP_FORMAT,
    DEFAULT_INPUT,
    MIKE_FLIGHTS_VERSION,
    create_mike_flights,
    get_mike_flights_for_day,
    mike_flights_parquet_path,
    normalize_mike_flights,
    read_mike_flights_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Mike's flights CSV into daily Parquet partitions."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input flights CSV (default: {DEFAULT_INPUT})",
    )
    return parser.parse_args()


def main() -> None:
    create_mike_flights(parse_args().input_csv)


if __name__ == "__main__":
    main()
