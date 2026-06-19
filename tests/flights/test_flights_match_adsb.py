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

    #a7f63d on 2026-03-01 in BTS is accurate and very hard to verify. Alaska flights. Also a6ce36 on 2026-03-01. Similiarly an Alaska flight. BTS is accurate.
    # a7f63d this should have been gotten by the cutoff time logic.https://globe.adsbexchange.com/?icao=ac9113&lat=41.393&lon=-107.783&zoom=6.0&showTrace=2026-03-01&leg=1&timestamp=1772325663 
    # https://globe.adsbexchange.com/?icao=046c56&lat=16.887&lon=-61.471&zoom=6.3&showTrace=2026-03-01&leg=1&timestamp=1772382996 make this good. for SFDPS

    

if __name__ == "__main__":
    unittest.main()

# TODO: Get data on how many pass this test.