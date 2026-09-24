from dataclasses import dataclass
from datetime import date, datetime

import polars as pl
import pytest

from airports.models import AirportCatalogProvenance
from flights import flight_source as flight_source_module
from flights.flight_source import flight_path, write_flights
from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.providers.mike import MikeFlightSource, create_mike_flights


@dataclass(frozen=True)
class FakeAirport:
    id: int
    ident: str


@dataclass(frozen=True)
class FakeAirportCatalog:
    provenance: AirportCatalogProvenance

    def resolve_code(
        self,
        code: str,
        *,
        code_priority: tuple[str, ...],
    ) -> FakeAirport | None:
        assert code_priority == ("ident",)
        return {
            "OLD-JFK": FakeAirport(101, "KJFK"),
            "OLD-LAX": FakeAirport(202, "KLAX"),
            "KBOS": FakeAirport(303, "KBOS"),
            "KSFO": FakeAirport(404, "KSFO"),
        }.get(code)


def _catalog(target_date: date) -> FakeAirportCatalog:
    return FakeAirportCatalog(
        AirportCatalogProvenance(
            flight_date=target_date,
            observed_commit_sha="observed-commit",
            knowledge_cutoff_date=date(2026, 4, 1),
            knowledge_commit_sha="knowledge-commit",
            catalog_available_date=date(2026, 4, 1),
            policy_version="test",
        )
    )


def _write_input(path, first_icao: str = " ABC123 "):
    pl.DataFrame(
        {
            "icao": [first_icao, "def456"],
            "takeoff_ts": [
                "2026-03-01 10:00:00.000 UTC",
                "2026-03-02 10:00:00.000 UTC",
            ],
            "landing_ts": [
                "2026-03-01 12:00:00.000 UTC",
                "2026-03-02 12:00:00.000 UTC",
            ],
            "takeoff_airport_ident": ["OLD-JFK", "KBOS"],
            "landing_airport_ident": ["OLD-LAX", "KSFO"],
            "owner_operator": ["Example Air", "Other Air"],
        }
    ).write_csv(path)


def test_create_mike_flights_resolves_stable_airport_ids_for_target_date(
    tmp_path,
):
    input_csv = tmp_path / "mike.csv"
    _write_input(input_csv)
    target_date = date(2026, 3, 1)
    airport_catalog = _catalog(target_date)
    result = create_mike_flights(target_date, airport_catalog, input_csv)

    assert result.schema == FLIGHT_POLARS_SCHEMA
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["takeoff_time"] == datetime(2026, 3, 1, 10)
    assert row["landing_time"] == datetime(2026, 3, 1, 12)
    assert row["takeoff_airport_id"] == 101
    assert row["takeoff_airport_ident"] == "KJFK"
    assert row["landing_airport_id"] == 202
    assert row["landing_airport_ident"] == "KLAX"
    assert result["icao"].to_list() == ["abc123"]
    assert result["owner"].to_list() == ["Example Air"]
    assert MikeFlightSource().version == 2


def test_create_mike_flights_propagates_ambiguous_airport_code(
    tmp_path,
):
    input_csv = tmp_path / "mike.csv"
    _write_input(input_csv)

    class AmbiguousAirportCatalog:
        def resolve_code(self, code, *, code_priority):
            raise ValueError(f"ambiguous airport code: {code}")

    with pytest.raises(ValueError, match="ambiguous airport code: OLD-JFK"):
        create_mike_flights(
            date(2026, 3, 1),
            AmbiguousAirportCatalog(),
            input_csv,
        )


def test_create_mike_flights_rejects_unknown_non_null_airport_code(tmp_path):
    input_csv = tmp_path / "mike.csv"
    _write_input(input_csv)

    class UnknownAirportCatalog:
        def resolve_code(self, code, *, code_priority):
            return None

    with pytest.raises(ValueError, match="Unknown airport code 'OLD-JFK'"):
        create_mike_flights(
            date(2026, 3, 1),
            UnknownAirportCatalog(),
            input_csv,
        )


def test_mike_input_contents_change_flight_path(monkeypatch, tmp_path):
    first_csv = tmp_path / "first" / "mike.csv"
    second_csv = tmp_path / "second" / "mike.csv"
    first_csv.parent.mkdir()
    second_csv.parent.mkdir()
    _write_input(first_csv, first_icao="abc123")
    _write_input(second_csv, first_icao="fedcba")
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path / "output")

    first_path = flight_path(
        MikeFlightSource(input_csv=first_csv),
        date(2026, 3, 1),
    )
    second_path = flight_path(
        MikeFlightSource(input_csv=second_csv),
        date(2026, 3, 1),
    )

    assert first_path != second_path


def test_write_mike_flights_uses_canonical_daily_path(monkeypatch, tmp_path):
    input_csv = tmp_path / "mike.csv"
    _write_input(input_csv)
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path / "output")
    source = MikeFlightSource(input_csv=input_csv)
    target_date = date(2026, 3, 1)
    airport_catalog = _catalog(target_date)
    monkeypatch.setattr(
        flight_source_module,
        "airport_catalog_for_date",
        lambda day: airport_catalog,
    )

    result = write_flights(source, target_date)
    path = flight_path(source, target_date)

    assert path.exists()
    assert pl.read_parquet(path).equals(result)
    assert result.height == 1
    assert result.select(
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ).row(0) == (101, "KJFK", 202, "KLAX")
