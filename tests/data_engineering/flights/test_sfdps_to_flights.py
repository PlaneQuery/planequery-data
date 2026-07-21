from datetime import datetime

import polars as pl

from data_engineering.flights import sfdps_to_flights as sfdps


def test_sfdps_to_flights_only_treats_valid_registrations_as_registrations(
    monkeypatch,
):
    callsigns = ["NKS108", "N12345", "N123AB", "N123A4", "C-GABC"]
    source = pl.DataFrame(
        {
            "gufi": [f"gufi-{index}" for index in range(len(callsigns))],
            "timestamp": [datetime(2026, 3, 1, 2, 0)] * len(callsigns),
            "callsign": callsigns,
            "flightStatus": ["COMPLETED"] * len(callsigns),
            "takeoff_time": [datetime(2026, 3, 1, 1, 0)] * len(callsigns),
            "takeoff_airport_icao": ["KJFK"] * len(callsigns),
            "landing_time": [datetime(2026, 3, 1, 2, 0)] * len(callsigns),
            "landing_airport_icao": ["KBOS"] * len(callsigns),
            "estimated_landing_time": [None] * len(callsigns),
            "estimated_takeoff_time": [None] * len(callsigns),
        },
        schema_overrides={
            "estimated_landing_time": pl.Datetime("us"),
            "estimated_takeoff_time": pl.Datetime("us"),
        },
    )
    monkeypatch.setattr(sfdps, "_read_sfdps_for_day", lambda _current_day: source)

    result = (
        sfdps.sfdps_to_flights(datetime(2026, 3, 1))
        .select("callsign", "registration")
        .sort("callsign")
    )

    assert result.to_dicts() == [
        {"callsign": "C-GABC", "registration": "C-GABC"},
        {"callsign": "N12345", "registration": "N12345"},
        {"callsign": "N123A4", "registration": None},
        {"callsign": "N123AB", "registration": "N123AB"},
        {"callsign": "NKS108", "registration": None},
    ]


def test_airline_callsign_starting_with_n_uses_callsign_icao_matching(monkeypatch):
    source = pl.DataFrame(
        {
            "gufi": ["gufi-1"],
            "timestamp": [datetime(2026, 3, 1, 2, 0)],
            "callsign": ["NKS108"],
            "flightStatus": ["COMPLETED"],
            "takeoff_time": [datetime(2026, 3, 1, 1, 0)],
            "takeoff_airport_icao": ["KJFK"],
            "landing_time": [datetime(2026, 3, 1, 2, 0)],
            "landing_airport_icao": ["KBOS"],
            "estimated_landing_time": [None],
            "estimated_takeoff_time": [None],
        },
        schema_overrides={
            "estimated_landing_time": pl.Datetime("us"),
            "estimated_takeoff_time": pl.Datetime("us"),
        },
    )
    monkeypatch.setattr(sfdps, "_read_sfdps_for_day", lambda _current_day: source)
    monkeypatch.setattr(
        sfdps,
        "_adsb_registration_map",
        lambda _current_day, _registrations: pl.DataFrame(
            schema={
                "icao": pl.String,
                "registration": pl.String,
                "pia": pl.Boolean,
                "ladd": pl.Boolean,
            }
        ),
    )
    monkeypatch.setattr(
        sfdps,
        "_adsb_callsign_segments",
        lambda _current_day, _callsigns: pl.DataFrame(
            {
                "icao": ["a91349"],
                "callsign": ["NKS108"],
                "start_time": [datetime(2026, 3, 1, 0, 55)],
                "end_time": [datetime(2026, 3, 1, 2, 5)],
                "pia": [False],
                "ladd": [False],
            }
        ),
    )
    monkeypatch.setattr(
        sfdps,
        "add_adsb_icao_info",
        lambda flights, _current_day: flights,
    )

    flights = sfdps.sfdps_to_flights(datetime(2026, 3, 1))
    result = sfdps.sfdps_flights_derive_icao(flights, datetime(2026, 3, 1))

    assert result.select("icao", "callsign").to_dicts() == [
        {"icao": "a91349", "callsign": "NKS108"}
    ]
