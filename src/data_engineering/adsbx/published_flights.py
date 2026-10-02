from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.utils import OUTPUT_DIR


ADSBX_PUBLISHED_ARRIVALS_VERSION = 1
PUBLISHED_FLIGHT_SCHEMA = {
    "icao": pl.String,
    "callsign": pl.String,
    "registration": pl.String,
    "takeoff_time": pl.Datetime("ms"),
    "takeoff_airport_ident": pl.String,
    "landing_time": pl.Datetime("ms"),
    "landing_airport_ident": pl.String,
}


def adsbx_published_arrivals_path(arrival_date: date) -> Path:
    """Path to the normalized ADSBX product partitioned by arrival day."""
    return (
        OUTPUT_DIR
        / "data"
        / "intermediate"
        / "adsbx"
        / "published-arrivals"
        / f"version={ADSBX_PUBLISHED_ARRIVALS_VERSION}"
        / f"year={arrival_date.year}"
        / f"month={arrival_date.month:02d}"
        / f"day={arrival_date.day:02d}"
        / "data.parquet"
    )


def legacy_adsbx_published_arrivals_path(arrival_date: date) -> Path:
    """Path used before published arrivals were separated from flight outputs."""
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / "adsbx"
        / "v1"
        / f"year={arrival_date.year}"
        / f"month={arrival_date.month:02d}"
        / f"day={arrival_date.day:02d}"
        / "data.parquet"
    )


def _adsbx_raw_arrivals_path(arrival_date: date) -> Path:
    arrival_date_str = arrival_date.strftime("%Y%m%d")
    return (
        OUTPUT_DIR
        / "data"
        / "raw"
        / "adsbx"
        / "flights-ax-v2"
        / f"flights-ax-v2_ax_arrivals_{arrival_date_str}.csv"
    )


def _select_published_flight_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize either a current or legacy arrivals artifact."""
    missing_columns = [
        pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in PUBLISHED_FLIGHT_SCHEMA.items()
        if column not in df.columns
    ]
    if missing_columns:
        df = df.with_columns(missing_columns)
    return df.select(
        pl.col(column).cast(dtype).alias(column)
        for column, dtype in PUBLISHED_FLIGHT_SCHEMA.items()
    ).with_columns(
        pl.col("icao").str.strip_chars().str.to_lowercase(),
        pl.col("registration").str.strip_chars().str.to_uppercase(),
        pl.col("takeoff_airport_ident").str.strip_chars().str.to_uppercase(),
        pl.col("landing_airport_ident").str.strip_chars().str.to_uppercase(),
    )


def _read_raw_adsbx_published_arrivals(arrival_date: date) -> pl.DataFrame:
    df = pl.read_csv(_adsbx_raw_arrivals_path(arrival_date))
    return _select_published_flight_columns(
        df.rename({
            "hex": "icao",
            "reg": "registration",
            "orig": "takeoff_airport_ident",
            "dest": "landing_airport_ident",
        })
        .with_columns(
            pl.lit(None).cast(pl.String).alias("callsign"),
            pl.col("depTime").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ms").alias("takeoff_time"),
            pl.col("arrTime").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ms").alias("landing_time"),
        )
    )


def read_adsbx_published_arrivals(arrival_date: date) -> pl.DataFrame:
    """Read one arrival-day ADSBX product, normalizing raw data when needed."""
    output_path = adsbx_published_arrivals_path(arrival_date)
    if output_path.exists():
        return _select_published_flight_columns(pl.read_parquet(output_path))

    legacy_path = legacy_adsbx_published_arrivals_path(arrival_date)
    if legacy_path.exists():
        return _select_published_flight_columns(pl.read_parquet(legacy_path))

    raw_path = _adsbx_raw_arrivals_path(arrival_date)
    if not raw_path.exists():
        raise FileNotFoundError(
            "No ADSBX published-arrivals input found for "
            f"{arrival_date}: checked {output_path}, {legacy_path}, and {raw_path}"
        )

    df = _read_raw_adsbx_published_arrivals(arrival_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return df


def read_adsbx_published_same_day_flights(
    flight_date: date,
) -> pl.DataFrame:
    """Return flights that both depart and arrive on one UTC date.

    ADSBX distributes this product by arrival date and only one day is
    available per monthly sample. Restricting that arrival partition to
    flights that also departed on the same date produces a complete cohort
    without requiring the unavailable following day's file.
    """
    return (
        read_adsbx_published_arrivals(flight_date)
        .filter(
            (pl.col("takeoff_time").dt.date() == flight_date)
            & (pl.col("landing_time").dt.date() == flight_date)
        )
        .unique(
            subset=[
                "icao",
                "takeoff_time",
                "landing_time",
                "takeoff_airport_ident",
                "landing_airport_ident",
            ],
            maintain_order=True,
        )
    )
