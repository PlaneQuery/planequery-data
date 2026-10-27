import os
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

from data_engineering.swim.read_sfdps import partition_sfdps_rows_by_gufi
from flights.sfdps.models import FlightStatus


def _sfdps_row(gufi: str | None, timestamp: datetime) -> dict:
    return {
        "gufi": gufi,
        "timestamp": timestamp,
        "aircraft_identification": "UAL123",
        "aircraft_address": "ABC123",
        "aircraft_registration": "N12345",
        "aircraft_type": "B738",
        "flight_status": "ACTIVE",
        "departure_point": "KJFK",
        "departure_actual_time": None,
        "departure_estimated_time": None,
        "arrival_point": "KLAX",
        "arrival_actual_time": None,
        "arrival_estimated_time": None,
    }


def test_partition_sfdps_rows_by_gufi_preserves_row_order():
    start = datetime(2026, 3, 1)
    first = _sfdps_row("gufi-1", start)
    second = _sfdps_row("gufi-2", start + timedelta(minutes=1))
    third = _sfdps_row("gufi-1", start + timedelta(minutes=2))
    df = pl.DataFrame(
        [first, second, third],
        schema_overrides={
            "departure_actual_time": pl.Datetime("us"),
            "departure_estimated_time": pl.Datetime("us"),
            "arrival_actual_time": pl.Datetime("us"),
            "arrival_estimated_time": pl.Datetime("us"),
        },
    )

    result = partition_sfdps_rows_by_gufi(df)

    assert list(result) == ["gufi-1", "gufi-2"]
    assert result["gufi-1"] == [first, third]
    assert result["gufi-2"] == [second]
    assert result["gufi-1"][0]["flight_status"] is FlightStatus.ACTIVE


def test_partition_sfdps_rows_by_gufi_rejects_null_gufi():
    df = pl.DataFrame(
        [_sfdps_row(None, datetime(2026, 3, 1))],
        schema_overrides={
            "gufi": pl.String,
            "departure_actual_time": pl.Datetime("us"),
            "departure_estimated_time": pl.Datetime("us"),
            "arrival_actual_time": pl.Datetime("us"),
            "arrival_estimated_time": pl.Datetime("us"),
        },
    )

    with pytest.raises(ValueError, match="null GUFI"):
        partition_sfdps_rows_by_gufi(df)


def test_partition_sfdps_rows_by_gufi_preserves_null_flight_status():
    row = _sfdps_row("gufi-1", datetime(2026, 3, 1))
    row["flight_status"] = None
    df = pl.DataFrame(
        [row],
        schema_overrides={
            "departure_actual_time": pl.Datetime("us"),
            "departure_estimated_time": pl.Datetime("us"),
            "arrival_actual_time": pl.Datetime("us"),
            "arrival_estimated_time": pl.Datetime("us"),
        },
    )

    result = partition_sfdps_rows_by_gufi(df)

    assert result["gufi-1"][0]["flight_status"] is None


def test_partition_sfdps_rows_by_gufi_rejects_unknown_flight_status():
    row = _sfdps_row("gufi-1", datetime(2026, 3, 1))
    row["flight_status"] = "UNKNOWN"
    df = pl.DataFrame(
        [row],
        schema_overrides={
            "departure_actual_time": pl.Datetime("us"),
            "departure_estimated_time": pl.Datetime("us"),
            "arrival_actual_time": pl.Datetime("us"),
            "arrival_estimated_time": pl.Datetime("us"),
        },
    )

    with pytest.raises(
        AssertionError,
        match="Unexpected SFDPS flight status: 'UNKNOWN'",
    ):
        partition_sfdps_rows_by_gufi(df)
