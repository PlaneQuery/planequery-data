"""Read downloaded OpenSky flights_data4 partitions for flight evaluation."""

from datetime import date, datetime, time, timedelta, timezone
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
    """Read takeoffs on a UTC day from that day's and the next day's downloads.

    Raw partitions use lastseen, so overnight flights are in the next day's
    file. Both files are required to avoid silently returning an incomplete day.
    """
    input_paths = []
    for partition_day in (day, day + timedelta(days=1)):
        input_path = raw_flights_day_path(partition_day, root=input_dir)
        if not input_path.is_file():
            raise FileNotFoundError(
                f"OpenSky flights_data4 download not found: {input_path}. "
                "Download it with: python -m "
                f"data_engineering.opensky.download_trino_flights {partition_day}"
            )
        input_paths.append(input_path)

    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    return (
        pl.scan_parquet(input_paths, hive_partitioning=False)
        .filter(
            (pl.col("firstseen") >= int(start.timestamp()))
            & (pl.col("firstseen") < int(end.timestamp()))
        )
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
