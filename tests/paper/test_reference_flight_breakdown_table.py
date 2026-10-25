"""Protect flight weighting, missing-data denominators, and code-field semantics."""
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/paper"))
import reference_flight_breakdown_table as table


def fixture_data():
    flights = pl.DataFrame({
        "category": [" A3 ", "A3", None, " ", "D2", "A0"],
        "aircraft_type": ["B738"] * 6,
        "takeoff_time": [datetime(2026, 3, 1)] * 6,
        "takeoff_airport_id": [1, 1, 2, 3, None, 999],
        "landing_airport_id": [2, 3, 4, 4, 1, 1],
    })
    airports = {
        1: SimpleNamespace(type="large_airport", iata_code="LAX", icao_code="KLAX"),
        2: SimpleNamespace(type="small_airport", iata_code=None, icao_code=None, ident="KXYZ", gps_code="KXYZ"),
        3: SimpleNamespace(type="medium_airport", iata_code="ABC", icao_code=" "),
        4: SimpleNamespace(type="heliport", iata_code="", icao_code="ABCD"),
    }
    return flights, airports


def test_weighting_unknowns_and_explicit_code_fields():
    flights, airports = fixture_data()
    rows = table.summarize(flights, airports, "bts", "2026-03-01")
    lookup = {(r["dimension"], r["endpoint"], r["category"]): r for r in rows}
    assert lookup["aircraft_category", "flight", "A3"]["count"] == 2
    assert lookup["aircraft_category", "flight", "missing"]["count"] == 2
    assert lookup["aircraft_category", "flight", "D2"]["count"] == 1
    assert lookup["airport_size", "takeoff", "large_airport"]["percent"] == pytest.approx(100/3)
    assert lookup["airport_codes", "takeoff", "neither"]["count"] == 1
    assert lookup["airport_codes", "takeoff", "unknown"]["count"] == 2
    assert lookup["airport_codes", "landing", "icao_only"]["count"] == 2
    for dimension, endpoint in {(r["dimension"], r["endpoint"]) for r in rows}:
        group = [r for r in rows if (r["dimension"], r["endpoint"]) == (dimension, endpoint)]
        assert sum(r["count"] for r in group) == 6
        assert sum(r["percent"] for r in group) == pytest.approx(100)
        assert {r["denominator"] for r in group} == {6}


def test_render_contains_dates_examples_and_zero_categories():
    flights, airports = fixture_data()
    flights = flights.with_columns(pl.Series("category", ["B0", "B1", "D2", "A3", None, "A0"]))
    rows = []
    for source, day in table.REFERENCE_DATES.items():
        sample = flights.with_columns(pl.lit(datetime.fromisoformat(day)).alias("takeoff_time"))
        if source.endswith("-verified"):
            sample = sample.head(3)
        rows.extend(table.summarize(sample, airports, source, day))
    aircraft = table.render_latex(rows, "test", "aircraft", 2)
    assert "Boeing 737" in aircraft
    assert "B0--D2: Other reported categories & 50.00 & 50.00 & 50.00" in aircraft
    assert "B0--D2: Other reported categories & 100.00 & 100.00 & 100.00" in aircraft
    for source in table.VERIFIED_SOURCES:
        assert table.LABELS[source] in aircraft
        assert table.REFERENCE_DATES[source] == table.REFERENCE_DATES[source.removesuffix("-verified")]
    assert "$n=3$" in aircraft and "$n=6$" in aircraft
    assert all(f"\n{code}: " not in aircraft for code in ("B0", "B1", "D2"))
    assert "A1: Light (e.g., Cessna 172) & 0.00 & 0.00 & 0.00" in aircraft
    airport = table.render_latex(rows, "test", "airports", 2)
    assert "No IATA or ICAO & 16.67 / 16.67" in airport
    assert all(f"\n{label} &" not in airport for label in
               ("Heliport", "Seaplane base", "Balloonport", "Closed",
                "Both IATA and ICAO", "IATA only", "ICAO only", "Neither IATA nor ICAO"))
    assert airport.count("\nUnknown / unresolved airport &") == 2
    assert airport.count("\nNo IATA or ICAO &") == 2
    assert "Airport code availability" not in airport
    assert "No IATA or ICAO & 33.33 / 33.33" in airport
    assert "Unknown / unresolved airport & 33.33 / 0.00" in airport
    assert "2024-06-01" in airport and "2026-03-01" in airport


def test_rejects_wrong_day_and_empty_input():
    flights, airports = fixture_data()
    with pytest.raises(ValueError, match="takeoff time"):
        table.summarize(flights, airports, "bts", "2026-03-02")
    with pytest.raises(ValueError, match="empty reference"):
        table.summarize(flights.head(0), airports, "bts", "2026-03-01")


def test_code_snapshot_is_separate_from_historical_airport_size():
    flights, airports = fixture_data()
    code_airports = {1: SimpleNamespace(iata_code=None, icao_code="KLAX")}
    rows = table.summarize(flights, airports, "bts", "2026-03-01", code_airports)
    lookup = {(r["dimension"], r["endpoint"], r["category"]): r["count"] for r in rows}
    assert lookup["airport_size", "takeoff", "large_airport"] == 2
    assert lookup["airport_codes", "takeoff", "icao_only"] == 2
    assert lookup["airport_codes", "takeoff", "unknown"] == 4
