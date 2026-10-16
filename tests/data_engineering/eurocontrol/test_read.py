import gzip
from dataclasses import dataclass
from datetime import date, datetime
from zipfile import ZipFile

import polars as pl
import pytest

from airports.models import AirportCatalogProvenance
from data_engineering.adsb import canonical
from data_engineering.adsb.partitions import icao_bucket
from data_engineering.eurocontrol import preprocess
import flights.providers.eurocontrol as eurocontrol_read
from flights import flight_source
from flights.flight_source import write_flights
from flights.flight_manifest import FlightManifest, flight_manifest_path


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
    ) -> FakeAirport | None:
        assert code_priority == eurocontrol_read.EUROCONTROL_AIRPORT_CODE_PRIORITY
        assert prioritize_faa_airspace is False
        return {
            "KJFK": FakeAirport(101, "KJFK"),
            "EGLL": FakeAirport(202, "EGLL"),
        }.get(code)


@pytest.mark.parametrize("callsign_source", [None, canonical.ADSBSource.ADSBX, canonical.ADSBSource.OPENSKY])
@pytest.mark.parametrize("has_coverage", [True, False])
def test_eurocontrol_zip_to_daily_flight_parquet(monkeypatch, tmp_path, callsign_source, has_coverage):
    target_date = date(2025, 8, 2)
    raw_csv = (
        '"ECTRL ID","AC Registration","ADEP","ADES","AC Type","AC Operator",'
        '"ACTUAL OFF BLOCK TIME","ACTUAL ARRIVAL TIME"\n'
        '"123","N123AB","KJFK","EGLL","GLEX","TST",'
        '"01-08-2025 23:55:00","02-08-2025 07:15:00"\n'
        '"456","N123AB","UTAN","EGLL","GLEX","TST",'
        '"02-08-2025 08:20:00","02-08-2025 15:15:00"\n'
        '"789","N123AB","KJFK","EGLL","GLEX","TST",'
        '"02-08-2025 23:55:00","03-08-2025 07:15:00"\n'
    )
    fir_csv = (
        '"ECTRL ID","Sequence Number","FIR ID","Entry Time","Exit Time"\n'
        '"789","0","TAXI_OUT","02-08-2025 23:55:00","03-08-2025 00:10:00"\n'
        '"456","0","TAXI_OUT","02-08-2025 08:20:00","02-08-2025 08:35:00"\n'
        '"123","0","TAXI_OUT","01-08-2025 23:55:00","02-08-2025 00:10:00"\n'
    )
    archive_path = tmp_path / "data/raw/eurocontrol/202508.zip"
    archive_path.parent.mkdir(parents=True)
    member_name = "202508/Flights_20250801_20250831.csv.gz"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(member_name, gzip.compress(raw_csv.encode()))
        archive.writestr(
            member_name.replace("Flights_", "Flight_FIRs_Actual_"),
            gzip.compress(fir_csv.encode()),
        )

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

    monkeypatch.setattr(canonical, "ADSB_PARQUET_ROOT", tmp_path / "adsb")
    # No constructor argument must still run enrichment with the default provider.
    source = (
        eurocontrol_read.EurocontrolFlightSource()
        if callsign_source is None
        else eurocontrol_read.EurocontrolFlightSource(callsign_source=callsign_source)
    )
    callsign_source = source.callsign_source
    if has_coverage:
        day_dir = canonical.adsb_day_dir(callsign_source, target_date)
        adsb_path = day_dir / f"icao_bucket={icao_bucket('abc123')}" / "data.parquet"
        adsb_path.parent.mkdir(parents=True)
        pl.DataFrame({
            "icao": ["abc123"] * 3,
            "time": [datetime(2025, 8, 2, 0, minute) for minute in (11, 12, 13)],
            "callsign": ["TST123"] * 3,
        }).write_parquet(adsb_path)
        canonical.mark_adsb_day_complete(day_dir)
    result = write_flights(source, target_date)

    output_path = flight_source.flight_path(
        source,
        target_date,
    )
    expected_dir = tmp_path / "data/flights/source=eurocontrol"
    expected_dir /= f"adsb_source={callsign_source.value}"
    assert output_path == expected_dir / "version=6/year=2025/month=08/day=02/data.parquet"
    manifest = FlightManifest.read(flight_manifest_path(output_path))
    assert manifest.adsb_source == callsign_source.value
    assert manifest.adsb_parquet_version == canonical.ADSB_PARQUET_VERSIONS[callsign_source]
    assert output_path.is_file()
    assert pl.read_parquet(output_path).equals(result)
    assert result.sort("takeoff_time")["callsign"].to_list() == (
        ["TST123", ""] if has_coverage else ["", ""]
    )
    with monkeypatch.context() as cached:
        def unexpected(*args, **kwargs):
            pytest.fail("Matching callsign-provider manifest should reuse the cache")

        cached.setattr(eurocontrol_read, "create_eurocontrol_flights", unexpected)
        assert flight_source.get_flights_for_catalog(source, target_date, airport_catalog).equals(result)
    assert result["takeoff_time"].sort().to_list() == [
        datetime(2025, 8, 2, 0, 10), datetime(2025, 8, 2, 8, 35)
    ]
    assert result["landing_time"].sort().to_list() == [
        datetime(2025, 8, 2, 7, 15), datetime(2025, 8, 2, 15, 15)
    ]
    assert result.select(
        "icao",
        "registration",
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ).row(0) == ("abc123", "N123AB", 101, "KJFK", 202, "EGLL")
    assert result.filter(pl.col("takeoff_airport_id").is_null()).select(
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ).row(0) == (None, None, 202, "EGLL")
    assert preprocess.eurocontrol_intermediate_path(target_date).is_file()
    assert list(archive_path.parent.iterdir()) == [archive_path]
