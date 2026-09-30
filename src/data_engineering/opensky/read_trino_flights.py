"""Read downloaded OpenSky flights_data4 partitions for flight evaluation."""

from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.opensky.paths import (
    OPENSKY_RAW_FLIGHTS_ROOT,
    raw_flights_day_path,
)


def read_opensky_flights(
    day: date,
    *,
    input_dir: Path = OPENSKY_RAW_FLIGHTS_ROOT,
) -> pl.DataFrame:
    """Normalize one downloaded Trino partition to evaluation flight columns."""
    input_path = raw_flights_day_path(day, root=input_dir)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"OpenSky flights_data4 download not found: {input_path}. "
            "Download it with: python -m "
            f"data_engineering.opensky.download_trino_flights {day}"
        )

    return (
        pl.scan_parquet(input_path, hive_partitioning=False)
        .select(
            pl.col("icao24").alias("icao"),
            pl.col("callsign").str.strip_chars(),
            pl.from_epoch("firstseen", time_unit="s").alias("takeoff_time"),
            pl.from_epoch("lastseen", time_unit="s").alias("landing_time"),
            pl.col("estdepartureairport").alias("takeoff_airport_ident"),
            pl.col("estarrivalairport").alias("landing_airport_ident"),
            pl.lit(False).alias("pia"),
            pl.lit(False).alias("ladd"),
            pl.lit(False).alias("military"),
            pl.lit(False).alias("interesting"),
        )
        .collect(engine="streaming")
    )
