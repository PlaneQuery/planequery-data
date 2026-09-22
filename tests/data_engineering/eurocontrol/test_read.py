from datetime import date

import polars as pl

import flights.providers.eurocontrol as eurocontrol_read


def test_eurocontrol_processing():
    dt = date(2024, 6, 1)
    raw = pl.read_csv(eurocontrol_read.eurocontrol_raw_flights_parquet_path(dt))
    raw = raw.with_columns(
        eurocontrol_read.datetime_expr("ACTUAL OFF BLOCK TIME", "takeoff_time")
    ).filter(pl.col("takeoff_time").dt.date() == dt)

    flights = eurocontrol_read.get_eurocontrol_flights(dt, use_cache=False)

    assert flights.height <= raw.height
    assert flights.get_column("takeoff_time").dt.date().eq(dt).all()
    assert flights.get_column("registration").is_not_null().all()
    assert flights.get_column("icao").is_not_null().all()
