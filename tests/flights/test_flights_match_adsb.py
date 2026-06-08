from datetime import date
import os
from pathlib import Path
import unittest

import polars as pl

from data_engineering.adsb.read_adsb import read_icaos_from_adsb
from flights.flights_match_adsb import flights_match_adsb
from data_engineering.flights.flight_type import get_flights


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
if __name__ == "__main__":
    unittest.main()

# TODO: Get data on how many pass this test.