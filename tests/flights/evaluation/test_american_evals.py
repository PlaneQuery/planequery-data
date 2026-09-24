import os
from pathlib import Path
from types import SimpleNamespace

import polars as pl


os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

from flights.evaluation.american_evals import (  # noqa: E402
    add_airspace_columns,
    continental_usa_to_continental_usa,
)


AIRPORT_CATALOG = SimpleNamespace(
    id_index={
        1: SimpleNamespace(
            is_in_us_faa_airspace=True,
            is_in_us_faa_island_airspace=False,
        ),
        2: SimpleNamespace(
            is_in_us_faa_airspace=True,
            is_in_us_faa_island_airspace=False,
        ),
        3: SimpleNamespace(
            is_in_us_faa_airspace=True,
            is_in_us_faa_island_airspace=True,
        ),
        4: SimpleNamespace(
            is_in_us_faa_airspace=False,
            is_in_us_faa_island_airspace=False,
        ),
    }
)


def _flight_rows() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "icao": "mainland",
                "takeoff_airport_id": 1,
                "takeoff_airport_ident": "KJFK",
                "landing_airport_id": 2,
                "landing_airport_ident": "KLAX",
            },
            {
                "icao": "island",
                "takeoff_airport_id": 1,
                "takeoff_airport_ident": "KJFK",
                "landing_airport_id": 3,
                "landing_airport_ident": "PHNL",
            },
            {
                "icao": "outside",
                "takeoff_airport_id": 4,
                "takeoff_airport_ident": "CYVR",
                "landing_airport_id": 1,
                "landing_airport_ident": "KJFK",
            },
            {
                "icao": "unresolved",
                "takeoff_airport_id": None,
                "takeoff_airport_ident": None,
                "landing_airport_id": 999,
                "landing_airport_ident": None,
            },
        ]
    )


def test_add_airspace_columns_marks_us_faa_and_island_airspace():
    df = add_airspace_columns(_flight_rows(), AIRPORT_CATALOG)

    assert df.select(
        [
            "icao",
            "takeoff_airport_in_us_faa_airspace",
            "landing_airport_in_us_faa_airspace",
            "takeoff_airport_in_us_faa_island_airspace",
            "landing_airport_in_us_faa_island_airspace",
        ]
    ).to_dicts() == [
        {
            "icao": "mainland",
            "takeoff_airport_in_us_faa_airspace": True,
            "landing_airport_in_us_faa_airspace": True,
            "takeoff_airport_in_us_faa_island_airspace": False,
            "landing_airport_in_us_faa_island_airspace": False,
        },
        {
            "icao": "island",
            "takeoff_airport_in_us_faa_airspace": True,
            "landing_airport_in_us_faa_airspace": True,
            "takeoff_airport_in_us_faa_island_airspace": False,
            "landing_airport_in_us_faa_island_airspace": True,
        },
        {
            "icao": "outside",
            "takeoff_airport_in_us_faa_airspace": False,
            "landing_airport_in_us_faa_airspace": True,
            "takeoff_airport_in_us_faa_island_airspace": False,
            "landing_airport_in_us_faa_island_airspace": False,
        },
        {
            "icao": "unresolved",
            "takeoff_airport_in_us_faa_airspace": False,
            "landing_airport_in_us_faa_airspace": False,
            "takeoff_airport_in_us_faa_island_airspace": False,
            "landing_airport_in_us_faa_island_airspace": False,
        },
    ]


def test_continental_usa_to_continental_usa_keeps_only_mainland_rows():
    df = continental_usa_to_continental_usa(
        _flight_rows(),
        AIRPORT_CATALOG,
    )

    assert df.get_column("icao").to_list() == ["mainland"]
    assert df.height == 1
