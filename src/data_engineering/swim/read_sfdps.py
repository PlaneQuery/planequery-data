from datetime import date, datetime
from pathlib import Path

import polars as pl

from data_engineering.swim.read_sfdps_logs import process_day
from data_engineering.utils import OUTPUT_DIR


def _date_partition(d: date) -> str:
    return f"year={d.year}/month={d.month:02d}/day={d.day:02d}/"


def _parquet_path(d: date) -> Path:
    return (
        OUTPUT_DIR
        / "data/intermediate"
        / f"sfdps-logs/year={d.year}/month={d.month:02d}/day={d.day:02d}"
        / f"sfdps-logs_{d.year}_{d.month:02d}_{d.day:02d}.parquet"
    )


def read_sfdps(current_day: datetime) -> pl.DataFrame:
    d = current_day.date()
    if not _parquet_path(d).exists():
        process_day(d)
    columns_needed = [
        "gufi",
        "timestamp",
        "aircraft_identification",
        "flight_status",
        "departure_point",
        "departure_actual_time",
        "departure_estimated_time",
        "arrival_point",
        "arrival_actual_time",
        "arrival_estimated_time",
    ]
    df = pl.read_parquet(_parquet_path(d), columns=columns_needed)
    df = df.rename({
        "aircraft_identification": "callsign",
        "flight_status": "flightStatus",
        "departure_point": "takeoff_airport_icao",
        "departure_actual_time": "takeoff_time",
        "departure_estimated_time": "estimated_takeoff_time",
        "arrival_point": "landing_airport_icao",
        "arrival_actual_time": "landing_time",
        "arrival_estimated_time": "estimated_landing_time",
    })

    time_cols = ["timestamp", "takeoff_time", "estimated_takeoff_time", "landing_time", "estimated_landing_time"]
    df = df.with_columns([
        pl.col(c).str.to_datetime(time_zone="UTC").dt.replace_time_zone(None)
        if df[c].dtype == pl.String else pl.col(c)
        for c in time_cols
    ])

    df = df.filter(pl.col("timestamp").dt.date() == d)
    df = df.filter(pl.col("flightStatus").is_not_null())
    df = df.sort("timestamp")

    state_keys = ["flightStatus", "landing_time", "takeoff_time", "takeoff_airport_icao", "landing_airport_icao"]

    def compress_gufi(group: pl.DataFrame) -> pl.DataFrame:
        """Keep only rows where state_keys change within a single gufi group."""
        return (
            group
            .sort("timestamp")
            .with_columns([
                pl.any_horizontal([
                    (pl.col(col) != pl.col(col).shift(1)).fill_null(True) |
                    (pl.col(col).is_null() != pl.col(col).shift(1).is_null())
                    for col in state_keys
                ]).alias("_changed")
            ])
            .with_columns(pl.col("_changed").cum_sum().alias("_run_id"))
            .group_by("_run_id")
            .last()
            .drop("_changed", "_run_id")
            .sort("timestamp")
        )

    df = (
        df
        .group_by("gufi")
        .map_groups(compress_gufi)
        .sort(["gufi", "timestamp"])
    )
    df = df.select(["gufi", "timestamp", "callsign", "flightStatus", "takeoff_time", "takeoff_airport_icao", "landing_time", "landing_airport_icao", "estimated_landing_time", "estimated_takeoff_time"])
    return df


if __name__ == "__main__":
    df_sfdps = read_sfdps(datetime(2026, 3, 1))
    print(df_sfdps)