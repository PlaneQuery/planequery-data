import os
from datetime import datetime
from pathlib import Path

import polars as pl


os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

from data_engineering.flights import sfdps_to_flights


def test_adsb_registration_map_keeps_pia_ladd_flags(monkeypatch):
    df_adsb = pl.DataFrame(
        {
            "icao": ["a00001", "a00001", "a00002", "a00003"],
            "registration": ["N1", "N1", "N1", "N2"],
            "pia": [False, True, False, False],
            "ladd": [False, False, True, True],
            "time": [
                datetime(2026, 3, 1, 1, 0),
                datetime(2026, 3, 1, 1, 1),
                datetime(2026, 3, 1, 1, 2),
                datetime(2026, 3, 1, 1, 3),
            ],
        }
    )

    monkeypatch.setattr(
        sfdps_to_flights,
        "scan_adsb",
        lambda *_args, **_kwargs: df_adsb.lazy(),
    )

    result = sfdps_to_flights._adsb_registration_map(
        datetime(2026, 3, 1),
        ["N1", "N2"],
    )

    assert result.sort("registration").to_dicts() == [
        {"icao": "a00001", "registration": "N1", "pia": True, "ladd": False},
        {"icao": "a00003", "registration": "N2", "pia": False, "ladd": True},
    ]


def test_sfdps_flights_derive_icao_concats_registration_and_callsign_matches(monkeypatch):
    takeoff_time = datetime(2026, 3, 1, 12, 0)
    landing_time = datetime(2026, 3, 1, 13, 0)
    df_sfdps_flights = sfdps_to_flights.with_flight_schema_columns(
        pl.DataFrame(
            {
                "icao": [None, None],
                "callsign": ["N1", "TEST123"],
                "registration": ["N1", None],
                "takeoff_time": [takeoff_time, takeoff_time],
                "takeoff_airport_ident": ["KJFK", "KLAX"],
                "landing_time": [landing_time, landing_time],
                "landing_airport_ident": ["KBOS", "KSFO"],
                "pia": [False, False],
                "ladd": [False, False],
            }
        )
    )

    monkeypatch.setattr(
        sfdps_to_flights,
        "_adsb_registration_map",
        lambda *_args, **_kwargs: pl.DataFrame(
            {
                "icao": ["a00001"],
                "registration": ["N1"],
                "pia": [True],
                "ladd": [False],
            }
        ),
    )
    monkeypatch.setattr(
        sfdps_to_flights,
        "_adsb_callsign_segments",
        lambda *_args, **_kwargs: pl.DataFrame(
            {
                "icao": ["a00002"],
                "callsign": ["TEST123"],
                "start_time": [datetime(2026, 3, 1, 11, 45)],
                "end_time": [datetime(2026, 3, 1, 13, 15)],
                "pia": [False],
                "ladd": [True],
            }
        ),
    )
    monkeypatch.setattr(
        sfdps_to_flights,
        "add_adsb_icao_info",
        lambda df_flights, _current_day: sfdps_to_flights.with_flight_schema_columns(df_flights),
    )

    result = sfdps_to_flights.sfdps_flights_derive_icao(
        df_sfdps_flights,
        datetime(2026, 3, 1),
    )

    assert sorted(result.get_column("icao").to_list()) == ["a00001", "a00002"]
