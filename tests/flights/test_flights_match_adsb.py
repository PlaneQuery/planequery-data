from datetime import date, datetime
import os
import unittest

import polars as pl

os.environ.setdefault("OUTPUT_DIR", "/tmp")

from data_engineering.adsb.read_adsb import read_icaos_from_adsb
from flights.flights_match_adsb import flights_match_adsb, get_matching_icaos_in_flights
from data_engineering.flights.flight_type import get_flights
from data_engineering.flights.flight_type import flights_to_flights_df, Flight


class TestFlightsMatchAdsb(unittest.TestCase):
    def test_flights_match_adsb(self):
        target_date = date(2026, 3, 1)

        icao = "a63b3c"
        df_flights = get_flights(target_date).filter(pl.col("icao") == icao)
        df_adsb = read_icaos_from_adsb(target_date, [icao]).filter(pl.col("icao") == icao)
        self.assertTrue(flights_match_adsb(df_flights, df_adsb))

        icao = "a4ed4f" # The airports are wrong for this. 
        df_flights = get_flights(target_date).filter(pl.col("icao") == icao)
        print(df_flights.select("takeoff_time", "landing_time", "takeoff_airport_ident", "landing_airport_ident"))
        df_adsb = read_icaos_from_adsb(target_date, [icao]).filter(pl.col("icao") == icao)
        self.assertTrue(flights_match_adsb(df_flights, df_adsb))

    # adsb.lol readsb historical is missing data for a54cd9 on 2026-03-01


def test_get_matching_icaos_in_flights_can_read_readsb_parquet(tmp_path):
    target_date = date(2026, 3, 1)
    day_dir = tmp_path / "year=2026" / "month=03" / "day=01"
    day_dir.mkdir(parents=True)
    takeoff_time = datetime(2026, 3, 1, 12, 5)
    landing_time = datetime(2026, 3, 1, 12, 20)

    df_flights = flights_to_flights_df(
        [
            Flight(
                icao="abc123",
                takeoff_time=takeoff_time,
                landing_time=landing_time,
                takeoff_airport_ident="",
                landing_airport_ident="",
            ),
            Flight(
                icao="def456",
                takeoff_time=takeoff_time,
                landing_time=landing_time,
                takeoff_airport_ident="",
                landing_airport_ident="",
            ),
        ]
    )

    pl.DataFrame(
        [
            {"time": datetime(2026, 3, 1, 12, 0), "icao": "abc123", "lat": 0.0, "lon": 0.0},
            {"time": takeoff_time, "icao": "abc123", "lat": 0.0, "lon": 0.0},
            {"time": datetime(2026, 3, 1, 12, 12), "icao": "abc123", "lat": 0.0, "lon": 1.0},
            {"time": landing_time, "icao": "abc123", "lat": 0.0, "lon": 2.0},
            {"time": datetime(2026, 3, 1, 12, 25), "icao": "abc123", "lat": 0.0, "lon": 2.0},
            {"time": datetime(2026, 3, 1, 12, 0), "icao": "def456", "lat": 0.0, "lon": 0.0},
            {"time": takeoff_time, "icao": "def456", "lat": 0.0, "lon": 1.0},
            {"time": landing_time, "icao": "def456", "lat": 0.0, "lon": 2.0},
            {"time": datetime(2026, 3, 1, 12, 25), "icao": "def456", "lat": 0.0, "lon": 2.0},
        ]
    ).write_parquet(day_dir / "part-00000.parquet")

    df = get_matching_icaos_in_flights(
        df_flights,
        use_readsb_parquet=True,
        target_date=target_date,
        readsb_parquet_root=tmp_path,
    )

    assert df.get_column("icao").unique().to_list() == ["abc123"]


def test_flights_match_adsb_rejects_stationary_adsb_during_flight_window():
    takeoff_time = datetime(2026, 3, 1, 12, 5)
    landing_time = datetime(2026, 3, 1, 12, 20)
    df_flights = flights_to_flights_df(
        [
            Flight(
                icao="abc123",
                takeoff_time=takeoff_time,
                landing_time=landing_time,
                takeoff_airport_ident="",
                landing_airport_ident="",
            )
        ]
    )
    df_adsb = pl.DataFrame(
        [
            {"time": datetime(2026, 3, 1, 12, 0), "icao": "abc123", "lat": 0.0, "lon": 0.0},
            {"time": takeoff_time, "icao": "abc123", "lat": 0.0, "lon": 0.0},
            {"time": datetime(2026, 3, 1, 12, 12), "icao": "abc123", "lat": 0.0, "lon": 0.0},
            {"time": landing_time, "icao": "abc123", "lat": 0.0, "lon": 0.0},
            {"time": datetime(2026, 3, 1, 12, 25), "icao": "abc123", "lat": 0.0, "lon": 0.0},
        ]
    )

    assert not flights_match_adsb(df_flights, df_adsb)


def test_flights_match_adsb_rejects_long_interior_stop_during_declared_flight():
    takeoff_time = datetime(2026, 3, 1, 14, 42)
    landing_time = datetime(2026, 3, 1, 21, 0)
    df_flights = flights_to_flights_df(
        [
            Flight(
                icao="0d0cf3",
                takeoff_time=takeoff_time,
                landing_time=landing_time,
                takeoff_airport_ident="MDPC",
                landing_airport_ident="KFLL",
            )
        ]
    )
    df_adsb = pl.DataFrame(
        [
            {"time": takeoff_time, "icao": "0d0cf3", "lat": 18.5, "lon": -68.4},
            {"time": datetime(2026, 3, 1, 15, 30), "icao": "0d0cf3", "lat": 20.0, "lon": -70.0},
            {"time": datetime(2026, 3, 1, 16, 43), "icao": "0d0cf3", "lat": 21.0, "lon": -71.0},
            {"time": datetime(2026, 3, 1, 17, 0), "icao": "0d0cf3", "lat": 21.0, "lon": -71.0},
            {"time": datetime(2026, 3, 1, 17, 45), "icao": "0d0cf3", "lat": 21.01, "lon": -71.01},
            {"time": datetime(2026, 3, 1, 18, 32), "icao": "0d0cf3", "lat": 21.02, "lon": -71.02},
            {"time": datetime(2026, 3, 1, 19, 30), "icao": "0d0cf3", "lat": 24.0, "lon": -75.0},
            {"time": landing_time, "icao": "0d0cf3", "lat": 26.1, "lon": -80.1},
        ]
    )

    assert not flights_match_adsb(df_flights, df_adsb)


def test_flights_match_adsb_rejects_slow_interior_gap_during_declared_flight():
    takeoff_time = datetime(2026, 3, 1, 14, 27)
    landing_time = datetime(2026, 3, 1, 21, 43)
    df_flights = flights_to_flights_df(
        [
            Flight(
                icao="a00125",
                takeoff_time=takeoff_time,
                landing_time=landing_time,
                takeoff_airport_ident="MMSL",
                landing_airport_ident="KADS",
            )
        ]
    )
    df_adsb = pl.DataFrame(
        [
            {"time": takeoff_time, "icao": "a00125", "lat": 23.0, "lon": -110.0},
            {"time": datetime(2026, 3, 1, 15, 30), "icao": "a00125", "lat": 23.8, "lon": -109.2},
            {"time": datetime(2026, 3, 1, 17, 6), "icao": "a00125", "lat": 22.9, "lon": -109.9},
            {"time": datetime(2026, 3, 1, 19, 1), "icao": "a00125", "lat": 23.0, "lon": -110.1},
            {"time": datetime(2026, 3, 1, 20, 15), "icao": "a00125", "lat": 27.0, "lon": -103.0},
            {"time": landing_time, "icao": "a00125", "lat": 32.9, "lon": -96.8},
        ]
    )

    assert not flights_match_adsb(df_flights, df_adsb)


if __name__ == "__main__":
    unittest.main()

# TODO: Get data on how many pass this test.
