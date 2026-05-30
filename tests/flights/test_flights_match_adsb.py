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

        icao = "a4ed4f"
        df_flights = get_flights(target_date).filter(pl.col("icao") == icao)
        df_adsb = read_icaos_from_adsb(target_date, [icao]).filter(pl.col("icao") == icao)
        self.assertFalse(flights_match_adsb(df_flights, df_adsb))


if __name__ == "__main__":
    unittest.main()
