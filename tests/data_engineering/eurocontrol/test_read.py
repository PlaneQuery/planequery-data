import gzip
from dataclasses import dataclass
from datetime import date
from zipfile import ZipFile

import polars as pl

from airports.models import AirportCatalogProvenance
from data_engineering.eurocontrol import preprocess
import flights.providers.eurocontrol as eurocontrol_read
from flights import flight_source
from flights.flight_source import write_flights


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
        prioritize_faa_airspace: bool,
    ) -> FakeAirport:
        assert code_priority == (
            "icao_code",
            "ident",
            "gps_code",
            "iata_code",
            "local_code",
        )
        assert prioritize_faa_airspace is False
        return {
            "KJFK": FakeAirport(101, "KJFK"),
            "EGLL": FakeAirport(202, "EGLL"),
        }[code]


def test_eurocontrol_zip_to_daily_flight_parquet(monkeypatch, tmp_path):
    target_date = date(2025, 8, 1)
    raw_csv = (
        '"AC Registration","ADEP","ADES","AC Type","AC Operator",'
        '"ACTUAL OFF BLOCK TIME","ACTUAL ARRIVAL TIME"\n'
        '"N123AB","KJFK","EGLL","GLEX","TST",'
        '"01-08-2025 00:20:00","01-08-2025 07:15:00"\n'
    )
    archive_path = tmp_path / "data/raw/eurocontrol/202508.zip"
    archive_path.parent.mkdir(parents=True)
    member_name = "202508/Flights_20250801_20250831.csv.gz"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(member_name, gzip.compress(raw_csv.encode()))

    airport_catalog = FakeAirportCatalog(
        AirportCatalogProvenance(
            flight_date=target_date,
            observed_commit_sha="observed-commit",
            knowledge_cutoff_date=date(2025, 9, 1),
            knowledge_commit_sha="knowledge-commit",
            catalog_available_date=date(2025, 9, 1),
            policy_version="test",
        )
    )

    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        flight_source,
        "airport_catalog_for_date",
        lambda target_date: airport_catalog,
    )
    monkeypatch.setattr(
        eurocontrol_read,
        "get_openairframes_snapshot",
        lambda at_date: pl.DataFrame(
            {"registration": ["N123AB"], "icao": ["abc123"]}
        ),
    )

    result = write_flights(
        eurocontrol_read.EurocontrolFlightSource(),
        target_date,
    )

    output_path = flight_source.flight_path(
        eurocontrol_read.EurocontrolFlightSource(),
        target_date,
    )
    assert output_path == (
        tmp_path
        / "data/flights/source=eurocontrol/version=3"
        / "year=2025/month=08/day=01/data.parquet"
    )
    assert output_path.is_file()
    assert pl.read_parquet(output_path).equals(result)
    assert result.select(
        "icao",
        "registration",
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ).row(0) == ("abc123", "N123AB", 101, "KJFK", 202, "EGLL")
    assert preprocess.eurocontrol_intermediate_path(target_date).is_file()
    assert not preprocess.legacy_eurocontrol_month_dir(target_date).exists()
