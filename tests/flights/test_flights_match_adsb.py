from datetime import date, datetime
import os
import unittest

import polars as pl

os.environ.setdefault("OUTPUT_DIR", "/tmp")

from data_engineering.adsb.read_adsb import read_icaos_from_adsb
from flights.flights_match_adsb import flights_match_adsb, get_matching_icaos_in_flights
from data_engineering.flights.flight_type import get_flights
from data_engineering.flights.flight_type import flights_to_flights_df, Flight


AIRPORTS_CSV = """ident,type,name,latitude_deg,longitude_deg,elevation_ft,continent,iso_country,iso_region,municipality,scheduled_service,gps_code,iata_code,local_code,home_link,wikipedia_link,keywords
TESTA,small_airport,Test Airport A,0.0,0.0,0,NA,US,US-TS,Test,no,TESTA,,TESTA,,,
TESTB,small_airport,Test Airport B,0.0,2.0,0,NA,US,US-TS,Test,no,TESTB,,TESTB,,,
"""


def _write_test_airports_csv(output_dir):
    airport_dir = output_dir / "data" / "interim"
    airport_dir.mkdir(parents=True)
    (airport_dir / "airports.csv").write_text(AIRPORTS_CSV)


def _reset_airport_lookup_cache(monkeypatch, output_dir):
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    import flights.flights_match_adsb as flights_match_adsb_module

    flights_match_adsb_module._airport_lookup = None
    flights_match_adsb_module._airport_lookup_loaded = False
    try:
        from airports.airportLookup import AirportLookup

        AirportLookup._instance = None
    except (ImportError, KeyError):
        pass


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
        self.assertFalse(flights_match_adsb(df_flights, df_adsb))

    # adsb.lol readsb historical is missing data for a54cd9 on 2026-03-01

    #a7f63d on 2026-03-01 in BTS is accurate and very hard to verify. Alaska flights. Also a6ce36 on 2026-03-01. Similiarly an Alaska flight. BTS is accurate.
    # a7f63d this should have been gotten by the cutoff time logic.https://globe.adsbexchange.com/?icao=ac9113&lat=41.393&lon=-107.783&zoom=6.0&showTrace=2026-03-01&leg=1&timestamp=1772325663 
    # https://globe.adsbexchange.com/?icao=046c56&lat=16.887&lon=-61.471&zoom=6.3&showTrace=2026-03-01&leg=1&timestamp=1772382996 make this good. for SFDPS

    

if __name__ == "__main__":
    unittest.main()

# TODO: Get data on how many pass this test.
