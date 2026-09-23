from datetime import date, datetime

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
from flights import flight_source
from flights.flight_source import ADSBFlightSource, flight_path
import flights.ground_truth.source as ground_truth
from flights.ground_truth.source import GroundTruthFlightSource
from flights.sources import FLIGHT_SOURCES_BY_NAME, GROUND_TRUTH_SOURCE


TARGET_DATE = date(2026, 3, 1)


def test_ground_truth_is_registered_adsb_source(monkeypatch, tmp_path):
    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)

    assert isinstance(GROUND_TRUTH_SOURCE, ADSBFlightSource)
    assert GROUND_TRUTH_SOURCE.name == "ground-truth"
    assert GROUND_TRUTH_SOURCE.adsb_columns() == ground_truth.ADSB_MATCHING_COLUMNS
    assert FLIGHT_SOURCES_BY_NAME[GROUND_TRUTH_SOURCE.name] is GROUND_TRUTH_SOURCE
    assert flight_path(
        GROUND_TRUTH_SOURCE,
        TARGET_DATE,
        ADSBSource.ADSBX,
    ) == (
        tmp_path
        / "data/flights/source=ground-truth/adsb_source=adsbx/version=1"
        / "year=2026/month=03/day=01/data.parquet"
    )


def test_ground_truth_source_uses_supplied_adsb(monkeypatch):
    flights = pl.DataFrame(
        {
            "icao": ["abc123"],
            "takeoff_time": [datetime(2026, 3, 1, 1)],
            "landing_time": [datetime(2026, 3, 1, 2)],
            "takeoff_airport_ident": ["KJFK"],
            "landing_airport_ident": ["KLAX"],
        }
    )
    adsb = pl.DataFrame({"icao": ["abc123"]})
    calls = []

    monkeypatch.setattr(
        ground_truth,
        "get_flights",
        lambda *_args, **_kwargs: flights.clone(),
    )
    monkeypatch.setattr(
        ground_truth,
        "continental_usa_to_continental_usa",
        lambda df: df,
    )
    monkeypatch.setattr(
        ground_truth,
        "df_flights_comparison",
        lambda df_bts, *_args, **_kwargs: df_bts.with_columns(
            pl.lit("both").alias("match_status"),
            pl.lit("").alias("category_df1"),
        ),
    )

    def match(flights_to_match, df_adsb, start):
        calls.append((df_adsb, start))
        return flights_to_match

    monkeypatch.setattr(ground_truth, "get_matching_icaos_in_flights", match)

    result = GroundTruthFlightSource().create_flights_for_day(TARGET_DATE, adsb)

    assert result.equals(flights)
    assert calls == [(adsb, datetime(2026, 3, 1))]


def test_get_ground_truth_flights_delegates_to_generic_reader(monkeypatch):
    expected = pl.DataFrame({"icao": ["abc123"]})
    calls = []

    def read(source, day, adsb_source, use_cache):
        calls.append((source, day, adsb_source, use_cache))
        return expected

    monkeypatch.setattr(ground_truth, "get_flights", read)

    result = ground_truth.get_ground_truth_flights_for_day(
        TARGET_DATE,
        ADSBSource.ADSBX,
        use_cache=False,
    )

    assert result is expected
    source, day, adsb_source, use_cache = calls[0]
    assert isinstance(source, GroundTruthFlightSource)
    assert day == TARGET_DATE
    assert adsb_source == ADSBSource.ADSBX
    assert use_cache is False
