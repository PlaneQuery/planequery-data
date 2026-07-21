from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.openairframes.read import add_latest_icao_info
from data_engineering.utils import OUTPUT_DIR


ADSBX_FLIGHTS_VERSION = "v1"


def adsbx_flights_output_path(current_date: date) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / "adsbx"
        / ADSBX_FLIGHTS_VERSION
        / f"year={current_date.year}"
        / f"month={current_date.month:02d}"
        / f"day={current_date.day:02d}"
        / "data.parquet"
    )


def _adsbx_raw_flights_path(current_date: date) -> Path:
    current_date_str = current_date.strftime("%Y%m%d")
    return (
        OUTPUT_DIR
        / "data"
        / "raw"
        / "adsb-exchange"
        / "flights-ax-v2"
        / f"flights-ax-v2_ax_arrivals_{current_date_str}.csv"
    )


def _create_adsbx_flights_for_day(current_date: date) -> pl.DataFrame:
    '''
    returns with takeoff_times on previous day. arrivals on current_date
    '''
    df = pl.read_csv(_adsbx_raw_flights_path(current_date))
    df = (
        df
        .rename({
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
    df = add_latest_icao_info(df)
    df = df.select([
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
    ])
    return df


def get_adsbx_flights_for_day(current_date: date) -> pl.DataFrame:
    output_path = adsbx_flights_output_path(current_date)
    if output_path.exists():
        return pl.read_parquet(output_path)

    df = _create_adsbx_flights_for_day(current_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return df
