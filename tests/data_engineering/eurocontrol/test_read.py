import gzip
from datetime import date
from zipfile import ZipFile

import polars as pl

from data_engineering.eurocontrol import preprocess
import flights.providers.eurocontrol as eurocontrol_read
from flights import flight_source
from flights.flight_source import write_flights


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

    class FakeAirport:
        def __init__(self, ident):
            self.ident = ident

    class FakeAirportLookup:
        def get_airport_from_code(self, airport_id, prioritize_faa_airspace):
            assert prioritize_faa_airspace is False
            return FakeAirport(airport_id)

    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(eurocontrol_read, "AirportLookup", FakeAirportLookup)
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
        / "data/flights/source=eurocontrol/version=1"
        / "year=2025/month=08/day=01/data.parquet"
    )
    assert output_path.is_file()
    assert pl.read_parquet(output_path).equals(result)
    assert result.select(
        "icao",
        "registration",
        "takeoff_airport_ident",
        "landing_airport_ident",
    ).row(0) == ("abc123", "N123AB", "KJFK", "EGLL")
    assert preprocess.eurocontrol_intermediate_path(target_date).is_file()
    assert not preprocess.legacy_eurocontrol_month_dir(target_date).exists()
