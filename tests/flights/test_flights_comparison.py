#use  ac9113. Do conversion to text 

# flight takeoff_time, landing_time fixed, 

from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
import flights.flights_comparison as comparison_module
from flights.flight_type import Flight
from flights.flights_comparison import (
    df_flights_comparison,
    df_flights_comparison_stats,
    flights_comparison,
)
import polars as pl
import pytest

def create_flight(takeoff_time: datetime):
    landing_time = takeoff_time + timedelta(hours=1)
    flight = Flight(
        "a",
        takeoff_time,
        landing_time,
        "A",
        "B",
        takeoff_airport_id=1,
        landing_airport_id=2,
    )
    return flight

flights_0 = [
   create_flight(datetime(2026,1,1,0,28))
]
flights_1 = [
    create_flight(datetime(2026,1,1,0,28)),
    create_flight(datetime(2026,1,1,2,0)),
    create_flight(datetime(2026,1,1,3,0)),
    create_flight(datetime(2026,1,1,4,0)),
]

flights_2 = [
    create_flight(datetime(2026,1,1,3)),
    create_flight(datetime(2026,1,1,6)),
]
flights_3 = [
    create_flight(datetime(2026,1,1,0,28)),
    create_flight(datetime(2026,1,1,3)),
    create_flight(datetime(2026,1,1,6)),
]

flights_4 = [
    create_flight(datetime(2026,1,1,3)),
    create_flight(datetime(2026,1,1,6)),
    create_flight(datetime(2026,1,1,23,58))
]

flights_5 = [
    create_flight(datetime(2026,1,1,0,28)),
    create_flight(datetime(2026,1,1,3)),
    create_flight(datetime(2026,1,1,6)),
    create_flight(datetime(2026,1,1,23,58))
]

start_dt = datetime.combine(date(2026,1,1), time.min)
end_dt = datetime.combine(date(2026,1,2), time.min)
airport_catalog = SimpleNamespace(
    airports=(
        SimpleNamespace(id=1, lat=0.0, lon=0.0),
        SimpleNamespace(id=2, lat=1.0, lon=1.0),
    )
)
flights_comparison(flights_0,flights_0, start_dt, end_dt)

def test_flights_comparison():
    f0_match, f1_match = flights_comparison(flights_0, flights_0, start_dt, end_dt)
    assert f0_match == [0]
    assert f1_match == [0]

    f0_match, f1_match = flights_comparison(flights_1, flights_1, start_dt, end_dt)
    matches = [0, 1, 2, 3]
    assert f0_match == matches
    assert f1_match == matches

    f0_match, f1_match = flights_comparison(flights_0, flights_1, start_dt, end_dt)
    assert f0_match == [0]
    assert f1_match == [0] + [None] * (len(f1_match) - 1)

    # check both ways 
    f0_match, f1_match = flights_comparison(flights_2, flights_3, start_dt, end_dt)
    assert f0_match == [1,2]
    assert f1_match == ['DISCARD',0,1]

    f0_match, f1_match = flights_comparison(flights_2, flights_4, start_dt, end_dt)
    assert f0_match == [0,1]
    assert f1_match == [0,1, 'DISCARD']

    f0_match, f1_match = flights_comparison(flights_2, flights_5, start_dt, end_dt)
    assert f0_match == [1,2]
    assert f1_match == ["DISCARD",0,1, 'DISCARD']

    f0_match, f1_match = flights_comparison(flights_5, flights_2 , start_dt, end_dt)
    assert f0_match == ["DISCARD",0,1, 'DISCARD']
    assert f1_match == [1,2]


def test_flight_landing_after_day_end_is_not_automatically_discarded():
    flight = create_flight(datetime(2026, 1, 1, 22))
    flight.landing_time = datetime(2026, 1, 2, 1)

    _, unmatched = flights_comparison([], [flight], start_dt, end_dt)

    assert unmatched == [None]

# I can have a fixed dataset here. 

def test_df_flights_comparision_with_icao_only_in_one_df():
    df_0 = pl.DataFrame(
        {
            "icao": ["a"],
            "takeoff_time": [datetime(2026, 1, 1, 3)],
            "landing_time": [datetime(2026, 1, 1, 4)],
            "takeoff_airport_id": [1],
            "takeoff_airport_ident": ["A"],
            "landing_airport_id": [2],
            "landing_airport_ident": ["B"],
        }
    )
    df_1 = pl.DataFrame(
        {
            "icao": ["b"],
            "takeoff_time": [datetime(2026, 1, 1, 3)],
            "landing_time": [datetime(2026, 1, 1, 4)],
            "takeoff_airport_id": [1],
            "takeoff_airport_ident": ["A"],
            "landing_airport_id": [2],
            "landing_airport_ident": ["B"],
        }
    )

    comparison = df_flights_comparison(
        df_0,
        df_1,
        start_dt,
        end_dt,
        airport_catalog,
    )

    assert sorted(comparison.get_column("match_status").to_list()) == ["df0_only", "df1_only"]


def test_airport_coordinates_are_added_once_before_icao_partitioning(monkeypatch):
    df = pl.DataFrame(
        {
            "icao": ["a", "b"],
            "takeoff_time": [datetime(2026, 1, 1, 3)] * 2,
            "landing_time": [datetime(2026, 1, 1, 4)] * 2,
            "takeoff_airport_id": [1, 1],
            "takeoff_airport_ident": ["A", "A"],
            "landing_airport_id": [2, 2],
            "landing_airport_ident": ["B", "B"],
        }
    )
    calls = []
    original = comparison_module.add_lat_lon_to_airport_columns

    def add_coordinates(flights, catalog):
        calls.append(flights.height)
        return original(flights, catalog)

    monkeypatch.setattr(
        comparison_module,
        "add_lat_lon_to_airport_columns",
        add_coordinates,
    )

    comparison_module.df_flights_comparison(
        df,
        df,
        start_dt,
        end_dt,
        airport_catalog,
    )

    assert calls == [2]

def test_df_flights_comparison_stats():
    comparison = pl.DataFrame(
        {
            "match_status": [
                "both",
                "both",
                "both",
                "both",
                "df0_only",
                "df1_only",
            ],
            "takeoff_airport_id": [1, 3, None, 1, 1, None],
            "takeoff_airport_id_df1": [1, 5, 1, None, None, 1],
            "landing_airport_id": [2, 4, 2, 2, 2, None],
            "landing_airport_id_df1": [2, 6, 2, 2, None, 2],
            "same_takeoff_airport_id": [True, False, False, False, False, False],
            "same_landing_airport_id": [True, False, True, True, False, False],
            "same_airport_id": [True, False, False, False, False, False],
        }
    )

    stats = df_flights_comparison_stats(comparison)

    assert stats["true_positive"] == 4
    assert stats["false_positive"] == 1
    assert stats["false_negative"] == 1
    assert stats["precision"] == pytest.approx(4 / 5)
    assert stats["recall"] == pytest.approx(4 / 5)
    assert stats["f1"] == pytest.approx(4 / 5)
    assert stats["takeoff_airport_id_total_count"] == 4
    assert stats["takeoff_airport_id_match_count"] == 1
    assert stats["takeoff_airport_id_incorrect_match_count"] == 3
    assert stats["takeoff_airport_id_match_pct"] == pytest.approx(1 / 4)
    assert stats["landing_airport_id_total_count"] == 4
    assert stats["landing_airport_id_match_count"] == 3
    assert stats["landing_airport_id_incorrect_match_count"] == 1
    assert stats["landing_airport_id_match_pct"] == pytest.approx(3 / 4)
    assert stats["airport_id_total_count"] == 4
    assert stats["airport_id_match_count"] == 1
    assert stats["airport_id_incorrect_match_count"] == 3
    assert stats["airport_id_match_pct"] == pytest.approx(1 / 4)
