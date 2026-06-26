from datetime import date, datetime
import os
from pathlib import Path

import polars as pl
import pytest


os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[2]))

from data_engineering.adsb.read_adsb import read_adsb  # noqa: E402
from data_engineering.flights.flight_type import Flight, flights_to_flights_df  # noqa: E402
from flights.flights_match_adsb import flights_match_adsb  # noqa: E402


TARGET_DATE = date(2026, 3, 1)

FLIGHT_MATCH_CASES = [
    pytest.param(
        "a63b3c",
        True,
        [
            (datetime(2026, 3, 1, 11, 38), "KTEB", datetime(2026, 3, 1, 13, 15), "KCHS"),
            (datetime(2026, 3, 1, 15, 19), "KCHS", datetime(2026, 3, 1, 16, 26), "KSUA"),
            (datetime(2026, 3, 1, 17, 2), "KSUA", datetime(2026, 3, 1, 19, 23), "KTEB"),
            (datetime(2026, 3, 1, 19, 48), "KTEB", datetime(2026, 3, 1, 20, 21), "KFOK"),
            (datetime(2026, 3, 1, 21, 3), "KFOK", datetime(2026, 3, 1, 23, 36), "KSTL"),
        ],
        id="a63b3c-matches",
    ),
    pytest.param(
        "a23406",
        True,
        [
            (datetime(2026, 3, 1, 16, 32), "KCOS", datetime(2026, 3, 1, 18, 15), "KGGG"),
            (datetime(2026, 3, 1, 18, 38), "KGGG", datetime(2026, 3, 1, 20, 54), "KISM"),
        ],
        id="a23406-matches",
    ),
    pytest.param(
        "a4ed4f",
        False,
        [
            (datetime(2026, 3, 1, 21, 17), "KCSV", datetime(2026, 3, 1, 21, 54), "KTYS"),
        ],
        id="a4ed4f-airports-wrong",
    ),
    pytest.param(
        "a1e5b5",
        False,
        [
            (datetime(2026, 3, 1, 15, 37), "KCIU", datetime(2026, 3, 1, 18, 13), "KUGN"),
        ],
        id="a1e5b5-airports-wrong",
    ),
]


def flights_df(icao: str, flight_rows) -> pl.DataFrame:
    return flights_to_flights_df(
        [
            Flight(
                icao=icao,
                takeoff_time=takeoff_time,
                takeoff_airport_ident=takeoff_airport_ident,
                landing_time=landing_time,
                landing_airport_ident=landing_airport_ident,
            )
            for (
                takeoff_time,
                takeoff_airport_ident,
                landing_time,
                landing_airport_ident,
            ) in flight_rows
        ]
    )


@pytest.mark.parametrize(("icao", "expected_match", "flight_rows"), FLIGHT_MATCH_CASES)
def test_flights_match_adsb(icao: str, expected_match: bool, flight_rows):
    df_flights = flights_df(icao, flight_rows)
    df_adsb = read_adsb(TARGET_DATE, icaos=[icao]).filter(pl.col("icao") == icao)

    assert flights_match_adsb(df_flights, df_adsb) is expected_match


'''
# adsb.lol readsb historical is missing data for a54cd9 on 2026-03-01

#a7f63d on 2026-03-01 in BTS is accurate and very hard to verify. Alaska flights. Also a6ce36 on 2026-03-01. Similiarly an Alaska flight. BTS is accurate.
# a7f63d this should have been gotten by the cutoff time logic.https://globe.adsbexchange.com/?icao=ac9113&lat=41.393&lon=-107.783&zoom=6.0&showTrace=2026-03-01&leg=1&timestamp=1772325663 
# https://globe.adsbexchange.com/?icao=046c56&lat=16.887&lon=-61.471&zoom=6.3&showTrace=2026-03-01&leg=1&timestamp=1772382996 make this good. for SFDPS
'''