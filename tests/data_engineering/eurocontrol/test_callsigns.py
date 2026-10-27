from datetime import date, datetime, timedelta

import polars as pl
import pytest

from data_engineering.adsb import canonical
from data_engineering.adsb.partitions import icao_bucket
from flights.flight_type import with_flight_schema_columns
from flights.providers.eurocontrol_callsigns import (
    assign_callsigns,
    enrich_eurocontrol_callsigns,
)


START = datetime(2024, 6, 1, 23, 50)


def flights(intervals):
    return with_flight_schema_columns(pl.DataFrame(
        {
            "icao": ["abc123"] * len(intervals),
            "takeoff_time": [START + timedelta(minutes=a) for a, _ in intervals],
            "landing_time": [START + timedelta(minutes=b) for _, b in intervals],
        },
        schema={"icao": pl.String, "takeoff_time": pl.Datetime("ms"), "landing_time": pl.Datetime("ms")},
    ))


def observations(rows):
    return pl.DataFrame(
        [(icao, START + timedelta(minutes=minute), callsign) for icao, minute, callsign in rows],
        schema={"icao": pl.String, "time": pl.Datetime("ms"), "callsign": pl.String},
        orient="row",
    )


def test_callsigns_stay_with_their_aircraft_and_flight_including_overnight():
    original = flights([(10, 20), (0, 10), (20, 30)])
    rows = (
        [("ABC123", minute, " ual123 ") for minute in (0, 1, 2)]
        + [("abc123", minute, "BAW456") for minute in (10, 11, 12)]
        + [("abc123", minute, "ON_GROUND") for minute in (-3, -2, -1)]
        + [("def456", minute, "WRONG") for minute in range(30)]
        + [("abc123", minute, None) for minute in (20, 21, 22)]
        + [("abc123", 30, "NEXTLEG")]
    )

    result = assign_callsigns(original, observations(rows).lazy())

    assert result["callsign"].to_list() == ["BAW456", "UAL123", ""]
    assert result.drop("callsign").equals(original.drop("callsign"))
    assert result.schema == original.schema


@pytest.mark.parametrize(
    "values, expected",
    [
        (["UAL123"] * 4 + ["BAW456"], "UAL123"),
        (["UAL123"] * 3 + ["BAW456"], ""),
        (["UAL123"] * 3 + ["BAW456"] * 3, ""),
        (["UAL123"] * 2, ""),
        (["UAL123"] * 3 + [None, "", "BAD!", "TOOLONG123"], "UAL123"),
    ],
)
def test_dominant_callsign_requires_sufficient_consistent_evidence(values, expected):
    result = assign_callsigns(
        flights([(0, 20)]),
        observations([("abc123", minute, value) for minute, value in enumerate(values)]).lazy(),
    )
    assert result["callsign"].to_list() == [expected]


def test_message_bursts_and_duplicates_do_not_dominate_votes():
    rows = [("abc123", second / 60, "BURST") for second in range(60)] * 5
    rows += [("abc123", minute, "UAL123") for minute in (1, 2, 3, 4)]
    result = assign_callsigns(flights([(0, 10)]), observations(rows).lazy())
    assert result["callsign"].to_list() == ["UAL123"]


def test_local_partitions_include_next_day_and_skip_incomplete_days(monkeypatch, tmp_path):
    monkeypatch.setattr(canonical, "ADSB_PARQUET_ROOT", tmp_path)
    source = canonical.ADSBSource.ADSBLOL
    original = flights([(0, 20), (20, 30)])
    # Only the next day is complete. June 1's partial data must be ignored.
    for day, rows, complete in (
        (date(2024, 6, 1), [("abc123", m, "PARTIAL") for m in range(10)], False),
        (date(2024, 6, 2), [("abc123", m, "UAL123") for m in (10, 11, 12)], True),
    ):
        day_dir = canonical.adsb_day_dir(source, day)
        path = day_dir / f"icao_bucket={icao_bucket('abc123')}" / "data.parquet"
        path.parent.mkdir(parents=True)
        observations(rows).write_parquet(path)
        if complete:
            canonical.mark_adsb_day_complete(day_dir)

    result = enrich_eurocontrol_callsigns(original, source)

    assert result["callsign"].to_list() == ["UAL123", ""]
    assert result.drop("callsign").equals(original.drop("callsign"))


def test_no_local_coverage_keeps_all_flights(monkeypatch, tmp_path):
    monkeypatch.setattr(canonical, "ADSB_PARQUET_ROOT", tmp_path)
    original = flights([(0, 20), (20, 30)])
    result = enrich_eurocontrol_callsigns(original, canonical.ADSBSource.ADSBX)
    assert result.equals(original)
    assert list(tmp_path.iterdir()) == []


def test_empty_or_invalid_flight_windows_do_not_read_adsb(monkeypatch):
    from flights.providers import eurocontrol_callsigns

    def unexpected(*args, **kwargs):
        pytest.fail("No valid flight windows should require ADS-B")

    monkeypatch.setattr(eurocontrol_callsigns, "scan_adsb_parquet", unexpected)
    for original in (flights([]), flights([(10, 0), (0, 0)])):
        assert enrich_eurocontrol_callsigns(original, canonical.ADSBSource.ADSBLOL).equals(original)
