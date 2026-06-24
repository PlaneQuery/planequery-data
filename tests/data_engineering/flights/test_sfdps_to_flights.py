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
