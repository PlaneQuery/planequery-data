from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import polars as pl

import flights.flights_comparison as comparison_module
from flights.flight_type import Flight
from flights.flights_comparison import (
    compare_flight,
    df_flights_comparison,
    flights_comparison,
)


def create_flight(takeoff_time: datetime) -> Flight:
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

flights_0 = [create_flight(datetime(2026, 1, 1, 0, 28))]
flights_1 = [
    create_flight(datetime(2026, 1, 1, 0, 28)),
    create_flight(datetime(2026, 1, 1, 2)),
    create_flight(datetime(2026, 1, 1, 3)),
    create_flight(datetime(2026, 1, 1, 4)),
]

flights_2 = [
    create_flight(datetime(2026, 1, 1, 3)),
    create_flight(datetime(2026, 1, 1, 6)),
]
flights_3 = [
    create_flight(datetime(2026, 1, 1, 0, 28)),
    create_flight(datetime(2026, 1, 1, 3)),
    create_flight(datetime(2026, 1, 1, 6)),
]

flights_4 = [
    create_flight(datetime(2026, 1, 1, 3)),
    create_flight(datetime(2026, 1, 1, 6)),
    create_flight(datetime(2026, 1, 1, 23, 58)),
]

flights_5 = [
    create_flight(datetime(2026, 1, 1, 0, 28)),
    create_flight(datetime(2026, 1, 1, 3)),
    create_flight(datetime(2026, 1, 1, 6)),
    create_flight(datetime(2026, 1, 1, 23, 58)),
]

start_dt = datetime.combine(date(2026, 1, 1), time.min)
end_dt = datetime.combine(date(2026, 1, 2), time.min)
airport_catalog = SimpleNamespace(
    airports=(
        SimpleNamespace(id=1, lat=0.0, lon=0.0),
        SimpleNamespace(id=2, lat=1.0, lon=1.0),
    )
)


def test_flights_comparison():
    assert flights_comparison(flights_0, flights_0, start_dt, end_dt) == [0]

    matches = [0, 1, 2, 3]
    assert flights_comparison(flights_1, flights_1, start_dt, end_dt) == matches

    assert flights_comparison(flights_0, flights_1, start_dt, end_dt) == [0]

    assert flights_comparison(flights_2, flights_3, start_dt, end_dt) == [1, 2]

    assert flights_comparison(flights_2, flights_4, start_dt, end_dt) == [0, 1]

    assert flights_comparison(flights_2, flights_5, start_dt, end_dt) == [1, 2]

    assert flights_comparison(flights_5, flights_2, start_dt, end_dt) == [
        None,
        0,
        1,
        None,
    ]


def test_compare_flight_accepts_custom_time_delta():
    flight_0 = create_flight(datetime(2026, 1, 1, 3))
    flight_1 = create_flight(datetime(2026, 1, 1, 3, 30))

    assert compare_flight(flight_0, flight_1)
    assert not compare_flight(
        flight_0,
        flight_1,
        max_flight_match_time_delta=timedelta(minutes=15),
    )


def test_flight_landing_after_day_end_is_not_automatically_discarded():
    flight = create_flight(datetime(2026, 1, 1, 22))
    flight.landing_time = datetime(2026, 1, 2, 1)

    unmatched = flights_comparison([flight], [], start_dt, end_dt)

    assert unmatched == [None]


def test_unmatched_test_flights_outside_comparison_window_are_retained():
    flights = [
        create_flight(start_dt - timedelta(minutes=1)),
        create_flight(end_dt + timedelta(minutes=1)),
    ]

    matches = flights_comparison(flights, [], start_dt, end_dt)

    assert matches == [None, None]


def test_flights_with_different_airports_match_on_time():
    flight_0 = create_flight(datetime(2026, 1, 1, 3))
    flight_1 = create_flight(datetime(2026, 1, 1, 3))
    flight_1.takeoff_airport_id = 3

    matches = flights_comparison(
        [flight_0],
        [flight_1],
        start_dt,
        end_dt,
    )

    assert matches == [0]


def test_df_flights_comparison_with_icao_only_in_one_df():
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


def test_df_flights_comparison_keeps_every_gold_row():
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
    gold_takeoff_times = [
        datetime(2026, 1, 1, 0),
        datetime(2026, 1, 1, 3),
        datetime(2026, 1, 1, 23),
    ]
    df_1 = pl.DataFrame(
        {
            "icao": ["a"] * 3,
            "takeoff_time": gold_takeoff_times,
            "landing_time": [dt + timedelta(hours=1) for dt in gold_takeoff_times],
            "takeoff_airport_id": [1] * 3,
            "takeoff_airport_ident": ["A"] * 3,
            "landing_airport_id": [2] * 3,
            "landing_airport_ident": ["B"] * 3,
        }
    )

    comparison = df_flights_comparison(
        df_0,
        df_1,
        start_dt,
        end_dt,
        airport_catalog,
    )

    assert comparison.height == df_1.height
    assert sorted(comparison.get_column("match_status").to_list()) == [
        "both",
        "df1_only",
        "df1_only",
    ]
    assert sorted(
        comparison.get_column("takeoff_time_df1").drop_nulls().to_list()
    ) == gold_takeoff_times


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
