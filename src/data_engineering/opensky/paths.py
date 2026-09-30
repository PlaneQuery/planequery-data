"""Filesystem layout for raw OpenSky Trino downloads."""

from datetime import date, datetime
from pathlib import Path

from data_engineering.utils import OUTPUT_DIR


OPENSKY_RAW_STATE_VECTORS_ROOT = (
    OUTPUT_DIR
    / "data"
    / "raw"
    / "opensky"
    / "trino-downloads"
    / "state_vectors_data4"
)
OPENSKY_RAW_FLIGHTS_ROOT = (
    OUTPUT_DIR / "data" / "raw" / "opensky" / "trino-downloads" / "flights_data4"
)


def coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def partitioned_day_dir(root: Path, value: date | datetime | str) -> Path:
    day = coerce_date(value)
    return (
        root
        / f"year={day.year}"
        / f"month={day.month:02d}"
        / f"day={day.day:02d}"
    )


def raw_state_vectors_day_dir(value: date | datetime | str) -> Path:
    return partitioned_day_dir(OPENSKY_RAW_STATE_VECTORS_ROOT, value)


def raw_state_vectors_hour_path(
    start: datetime,
    root: Path = OPENSKY_RAW_STATE_VECTORS_ROOT,
) -> Path:
    return partitioned_day_dir(root, start) / f"hour={start.hour:02d}" / "data.parquet"


def raw_flights_day_path(
    value: date | datetime | str,
    root: Path = OPENSKY_RAW_FLIGHTS_ROOT,
) -> Path:
    return partitioned_day_dir(root, value) / "data.parquet"
