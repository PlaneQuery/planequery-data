from datetime import date, datetime, timedelta

import polars as pl
import pytest

from airports.models import AirportCatalogProvenance
from data_engineering.adsb.canonical import ADSBSource
from flights import flight_source, sources
from flights.flight_manifest import FlightManifest, flight_manifest_path
from flights.flight_type import with_flight_schema_columns
from flights.providers.bts import BTSFlightSource
from flights.providers.eurocontrol import EurocontrolFlightSource
from flights.providers.sfdps import SFDPSFlightSource
from flights.verified import source as verified_source
from flights.verified import verification


DAY = date(2024, 6, 1)


class Catalog:
    provenance = AirportCatalogProvenance(
        flight_date=DAY,
        observed_commit_sha="observed",
        knowledge_cutoff_date=DAY,
        knowledge_commit_sha="knowledge",
        catalog_available_date=DAY,
        policy_version="test",
    )


def european_flights():
    return with_flight_schema_columns(pl.DataFrame({
        "icao": ["abc123"] * 4,
        "callsign": [""] * 4,  # Base cache may predate ADS-B availability.
        "takeoff_time": [datetime(2024, 6, 1, h) for h in (6, 12, 18, 23)],
        "landing_time": [datetime(2024, 6, 1, h) for h in (8, 14, 20)] + [datetime(2024, 6, 2, 1)],
        "takeoff_airport_ident": ["EGLL", "LFPG", "EDDF", "EHAM"],
        "landing_airport_ident": ["LFPG", "EDDF", "EHAM", "EGLL"],
    }))


def observed_callsigns():
    # The final flight is observed only after midnight.
    starts = [datetime(2024, 6, 1, h) for h in (6, 12, 18)] + [datetime(2024, 6, 2)]
    return pl.DataFrame({
        "icao": ["abc123"] * 12,
        "time": [start + timedelta(minutes=m) for start in starts for m in range(3)],
        "callsign": [callsign for callsign in ("BAW1", "BAW2", "BAW3", "BAW4") for _ in range(3)],
    })


@pytest.mark.parametrize("provider", [ADSBSource.ADSBLOL, ADSBSource.OPENSKY])
@pytest.mark.parametrize("scenario", ["valid", "broken_airports", "unknown_airports", "too_few_flights", "extra_callsign"])
def test_eurocontrol_verified_checks_global_flight_sequences(monkeypatch, provider, scenario):
    base = european_flights()
    observations = observed_callsigns()
    if scenario == "broken_airports":
        base = base.with_columns(pl.Series("takeoff_airport_ident", ["EGLL", "LIRF", "EDDF", "EHAM"]))
    elif scenario == "unknown_airports":
        base = base.with_columns(
            pl.lit(None, dtype=pl.String).alias("takeoff_airport_ident"),
            pl.lit(None, dtype=pl.String).alias("landing_airport_ident"),
        )
    elif scenario == "too_few_flights":
        base = base.head(3)
    elif scenario == "extra_callsign":
        observations = pl.concat([observations, pl.DataFrame({
            "icao": ["abc123"], "time": [datetime(2024, 6, 1, 10)], "callsign": ["EXTRA"],
        })])
    catalog = Catalog()
    wrapped = EurocontrolFlightSource(callsign_source=provider)
    source = verified_source.VerifiedFlightSource(wrapped)

    def get_base(source_arg, day_arg, catalog_arg):
        assert source_arg is wrapped
        assert day_arg == DAY
        assert catalog_arg is catalog
        return base.reverse()

    def read(day_arg, *, end, columns, icaos, source):
        assert day_arg == DAY
        expected_end = datetime(2024, 6, 2) if scenario == "too_few_flights" else datetime(2024, 6, 2, 1)
        assert end == expected_end
        assert columns == ["icao", "time", "callsign"]
        assert icaos == ["abc123"]
        assert source is provider
        return observations.filter(pl.col("time") < end).reverse()

    monkeypatch.setattr(verified_source, "get_flights_for_catalog", get_base)
    monkeypatch.setattr(verified_source, "read_adsb", read)
    monkeypatch.setattr(verification, "_check_flights_in_us_faa_airspace", lambda *args: pytest.fail("EUROCONTROL must allow European flights"))

    result = source.create_flights_for_day(DAY, catalog)

    if scenario == "valid":
        assert result["callsign"].to_list() == ["BAW1", "BAW2", "BAW3", "BAW4"]
        assert result.drop("callsign").equals(base.drop("callsign"))
    else:
        assert result.is_empty()


def test_overnight_window_does_not_include_other_aircraft_next_day_flights(monkeypatch):
    base = european_flights()
    daytime = base.with_columns(
        pl.lit("def456").alias("icao"),
        pl.col("takeoff_time") - pl.duration(hours=2),
        pl.col("landing_time") - pl.duration(hours=2),
    )
    observations = observed_callsigns()
    daytime_observations = observations.with_columns(
        pl.lit("def456").alias("icao"),
        pl.col("time") - pl.duration(hours=2),
    )
    next_day = pl.DataFrame({
        "icao": ["def456"], "time": [datetime(2024, 6, 2, 0, 30)], "callsign": ["NEWLEG"],
    })
    monkeypatch.setattr(verified_source, "get_flights_for_catalog", lambda *args: pl.concat([base, daytime]))
    monkeypatch.setattr(verified_source, "read_adsb", lambda *args, **kwargs: pl.concat([observations, daytime_observations, next_day]))

    result = sources.EUROCONTROL_VERIFIED_SOURCE.create_flights_for_day(DAY, Catalog())

    assert result.height == 8
    assert result["icao"].n_unique() == 2


def test_eurocontrol_verified_registry_and_cache_provenance(monkeypatch, tmp_path):
    source = sources.FLIGHT_SOURCES_BY_NAME["eurocontrol-verified"]
    assert source is sources.EUROCONTROL_VERIFIED_SOURCE
    assert source.source is sources.EUROCONTROL_SOURCE
    assert source.enrichment_adsb_source() is ADSBSource.ADSBLOL
    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(verified_source, "get_flights_for_catalog", lambda *args: european_flights())
    monkeypatch.setattr(verified_source, "read_adsb", lambda *args, **kwargs: observed_callsigns())

    result = flight_source.write_flights_for_catalog(source, DAY, Catalog())

    path = flight_source.flight_path(source, DAY)
    assert path == tmp_path / "data/flights/source=eurocontrol-verified/adsb_source=adsblol/version=1/year=2024/month=06/day=01/data.parquet"
    assert pl.read_parquet(path).equals(result)
    assert FlightManifest.read(flight_manifest_path(path)).adsb_source == "adsblol"
    assert result.height == 4


def test_empty_eurocontrol_verified_does_not_read_adsb(monkeypatch):
    empty = european_flights().head(0)
    monkeypatch.setattr(verified_source, "get_flights_for_catalog", lambda *args: empty)
    monkeypatch.setattr(verified_source, "read_adsb", lambda *args, **kwargs: pytest.fail("No ADS-B needed for empty input"))
    assert sources.EUROCONTROL_VERIFIED_SOURCE.create_flights_for_day(DAY, Catalog()).equals(empty)


@pytest.mark.parametrize("base_source", [BTSFlightSource(), SFDPSFlightSource()])
def test_existing_verified_sources_keep_us_faa_filter(monkeypatch, base_source):
    base = european_flights().with_columns(pl.Series("callsign", ["BAW1", "BAW2", "BAW3", "BAW4"]))
    calls = []
    monkeypatch.setattr(verified_source, "get_flights_for_catalog", lambda *args: base)
    monkeypatch.setattr(verified_source, "read_adsb", lambda day: observed_callsigns())
    monkeypatch.setattr(verification, "_check_flights_in_us_faa_airspace", lambda *args: calls.append(True) or [])
    result = verified_source.VerifiedFlightSource(base_source).create_flights_for_day(DAY, Catalog())
    assert calls == [True]
    assert result.is_empty()
