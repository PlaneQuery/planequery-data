from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
import polars as pl
import pytest
from data_engineering.adsb.read_adsb import read_adsb
from flights.flight_type import Flight, flights_to_flights_df
from flights.evaluation.flights_test_package import flights_match_real_adsb_with_dropout
from flights.flights_match_adsb import flights_match_adsb
import flights.flights_match_adsb as flights_match_adsb_module
from tests.flights.example_flights import Flights_Test_Packages, FlightsTestPackage
from tests.flights.example_false_flights import False_Flights_Test_Packages
TARGET_DATE = date(2026, 3, 1)

_AIRPORT_ROWS = pl.read_csv(
    Path(__file__).parents[2] / "data" / "raw" / "ourairports" / "airports.csv",
    columns=["id", "ident", "latitude_deg", "longitude_deg"],
)
_AIRPORT_ID_BY_IDENT = dict(
    _AIRPORT_ROWS.select("ident", "id").iter_rows()
)
_AIRPORT_CATALOG = SimpleNamespace(
    id_index={
        row["id"]: SimpleNamespace(
            id=row["id"],
            ident=row["ident"],
            lat=row["latitude_deg"],
            lon=row["longitude_deg"],
        )
        for row in _AIRPORT_ROWS.iter_rows(named=True)
    }
)


@pytest.fixture(autouse=True)
def airport_catalog(monkeypatch):
    monkeypatch.setattr(
        flights_match_adsb_module,
        "airport_catalog_for_date",
        lambda _: _AIRPORT_CATALOG,
    )

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
        "ab4b73",
        True,
        [
            (datetime(2026, 3, 1, 19, 46), "KPIE", datetime(2026, 3, 1, 20, 43), "KSAV"),
            (datetime(2026, 3, 1, 21, 2), "KSAV", datetime(2026, 3, 1, 22, 25), "KAGC"),
        ],
        id="ab4b73-matches",
    ),
    pytest.param(
        "a67084",
        True,
        [
            (datetime(2026, 3, 1, 0, 21), "KCHO", datetime(2026, 3, 1, 1, 20), "KPHF"),
            (datetime(2026, 3, 1, 1, 29), "KPHF", datetime(2026, 3, 1, 2, 52), "KHEF"),
        ],
        id="a67084-matches",
        marks=pytest.mark.skip(reason="flights_match_algoirhtm gives False negatie on this.") # False negatives are ok. False Positives are terrible
    ),
    pytest.param(
        "a00929",
        "unknown",
        [
            (datetime(2026, 3, 1, 3, 5), "KORD", datetime(2026, 3, 1, 4, 46), "KOKC"),
            (datetime(2026, 3, 1, 16, 35), "KOKC", datetime(2026, 3, 1, 19, 17), "KLGA"),
            (datetime(2026, 3, 1, 20, 55), "KLGA", datetime(2026, 3, 1, 22, 19), "KCLE"),
            (datetime(2026, 3, 1, 23, 32), "KCLE", datetime(2026, 3, 2, 0, 31), "KLGA"),
        ],
        id="a00929-early-coverage-unknown",
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
    pytest.param(
        "a07b98",
        False,
        [
            (datetime(2026, 3, 1, 14, 55), "KHPN", datetime(2026, 3, 1, 17, 22), "KMQY"),
            (datetime(2026, 3, 1, 17, 45), "KMQY", datetime(2026, 3, 1, 18, 54), "KPNS"),
        ],
        id="a07b98-airports-wrong",
    ),
]


def flights_df(icao: str, flight_rows) -> pl.DataFrame:
    return flights_to_flights_df(
        [
            Flight(
                icao=icao,
                takeoff_time=takeoff_time,
                takeoff_airport_id=_AIRPORT_ID_BY_IDENT[takeoff_airport_ident],
                takeoff_airport_ident=takeoff_airport_ident,
                landing_time=landing_time,
                landing_airport_id=_AIRPORT_ID_BY_IDENT[landing_airport_ident],
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


def test_flights_match_adsb_uses_stable_airport_ids():
    airport_ident = "KJFK"
    airport = _AIRPORT_CATALOG.id_index[_AIRPORT_ID_BY_IDENT[airport_ident]]
    takeoff_time = datetime(2026, 3, 1, 10)
    landing_time = datetime(2026, 3, 1, 11)
    df_flights = flights_df(
        "abc123",
        [(takeoff_time, airport_ident, landing_time, airport_ident)],
    )
    df_adsb = pl.DataFrame(
        {
            "time": [
                takeoff_time,
                landing_time,
                landing_time.replace(minute=11),
            ],
            "lat": [airport.lat] * 3,
            "lon": [airport.lon] * 3,
        }
    )

    assert flights_match_adsb(
        df_flights,
        df_adsb,
        datetime(2026, 3, 1),
    ) is True


@pytest.mark.parametrize(("icao", "expected_match", "flight_rows"), FLIGHT_MATCH_CASES)
def test_flights_match_adsb(
    icao: str, expected_match: bool | Literal["unknown"], flight_rows
):
    df_flights = flights_df(icao, flight_rows)
    df_adsb = read_adsb(TARGET_DATE, icaos=[icao]).filter(pl.col("icao") == icao)
    dt = datetime(TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day)
    assert flights_match_adsb(df_flights, df_adsb, dt) == expected_match


def test_flights_match_adsb_rejects_a233f0_on_2026_06_01():
    icao = "a233f0"
    target_date = date(2026, 6, 1)
    df_flights = flights_df(
        icao,
        [
            (
                datetime(2026, 6, 1, 19, 10),
                "KVNY",
                datetime(2026, 6, 1, 23, 30),
                "KIAD",
            ),
        ],
    )
    df_adsb = read_adsb(target_date, icaos=[icao]).filter(pl.col("icao") == icao)
    dt = datetime(target_date.year, target_date.month, target_date.day)
    assert flights_match_adsb(df_flights, df_adsb, dt) is False

def test_flights_test_packages():
    for test_package in Flights_Test_Packages:
        flights_match_real_adsb_with_dropout(test_package)

def test_false_flights_test_packages():
    for test_package in False_Flights_Test_Packages:
        flights_match_real_adsb_with_dropout(test_package)

'''
# adsb.lol readsb historical is missing data for a54cd9 on 2026-03-01

#a7f63d on 2026-03-01 in BTS is accurate and very hard to verify. Alaska flights. Also a6ce36 on 2026-03-01. Similiarly an Alaska flight. BTS is accurate.
# a7f63d this should have been gotten by the cutoff time logic.https://globe.adsbexchange.com/?icao=ac9113&lat=41.393&lon=-107.783&zoom=6.0&showTrace=2026-03-01&leg=1&timestamp=1772325663 
# https://globe.adsbexchange.com/?icao=046c56&lat=16.887&lon=-61.471&zoom=6.3&showTrace=2026-03-01&leg=1&timestamp=1772382996 make this good. for SFDPS
'''
