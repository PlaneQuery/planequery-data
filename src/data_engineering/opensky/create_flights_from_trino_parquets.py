import argparse
from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.openairframes.read import add_latest_icao_info
from data_engineering.utils import OUTPUT_DIR

RAW_DIR = OUTPUT_DIR / "data" / "raw" / "opensky" / "trino-table-flights"
OUTPUT_PATH = OUTPUT_DIR / "data" / "flights" / "opensky" / "v1"


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected date in YYYY-MM-DD format") from exc


def _raw_partition_dir(day: date) -> Path:
    return RAW_DIR / f"year={day.year}" / f"month={day.month:02d}" / f"day={day.day:02d}"


def _output_partition_dir(day: date) -> Path:
    return OUTPUT_PATH / f"year={day.year}" / f"month={day.month:02d}" / f"day={day.day:02d}"


def opensky_flights_output_path(day: date) -> Path:
    return _output_partition_dir(day) / "data.parquet"


def read_opensky_flights(day: date, add_icao_info: bool = False) -> pl.DataFrame:
    df = pl.read_parquet(opensky_flights_output_path(day))
    if add_icao_info:
        return add_latest_icao_info(df)
    return df


def create_flights_for_date(day: date) -> Path:
    raw_dir = _raw_partition_dir(day)
    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    df = (
        pl.read_parquet(parquet_files)
        .select(
            pl.col("icao24").alias("icao"),
            pl.from_epoch("firstSeen", time_unit="s").alias("takeoff_time"),
            pl.from_epoch("lastSeen", time_unit="s").alias("landing_time"),
            pl.col("estDepartureAirport").alias("takeoff_airport_ident"),
            pl.col("estArrivalAirport").alias("landing_airport_ident"),
            pl.lit(False).alias("pia"),
            pl.lit(False).alias("ladd"),
            pl.lit(False).alias("military"),
            pl.lit(False).alias("interesting"),
        )
    )

    output_file = opensky_flights_output_path(day)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_file)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("date", type=_parse_date)
    args = parser.parse_args()
    output_file = create_flights_for_date(args.date)
    print(output_file)


if __name__ == "__main__":
    main()