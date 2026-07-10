import os
from pathlib import Path

import polars as pl


os.environ["OUTPUT_DIR"] = str(Path(__file__).resolve().parents[3])

from flights.evaluation.american_evals import (  # noqa: E402
    add_airspace_columns,
    continental_usa_to_continental_usa,
)


def _flight_rows() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "icao": "mainland",
                "takeoff_airport_ident": "KJFK",
                "landing_airport_ident": "KLAX",
            },
            {
                "icao": "island",
                "takeoff_airport_ident": "KJFK",
                "landing_airport_ident": "PHNL",
            },
            {
                "icao": "outside",
                "takeoff_airport_ident": "CYVR",
                "landing_airport_ident": "KJFK",
            },
        ]
    )


def test_add_airspace_columns_marks_us_faa_and_island_airspace():
    df = add_airspace_columns(_flight_rows())

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
    ]


def test_continental_usa_to_continental_usa_keeps_only_mainland_rows():
    df = continental_usa_to_continental_usa(_flight_rows())

    assert df.get_column("icao").to_list() == ["mainland"]
    assert df.height == 1
