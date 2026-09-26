import os
from pathlib import Path
from types import SimpleNamespace

import polars as pl

os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[2]))

from airports.polars_exprs import (
    airport_has_iata_code_expr,
    airport_has_icao_code_expr,
    airport_size_expr,
)


def test_airport_attribute_expressions():
    airport_catalog = SimpleNamespace(
        airports=(
            SimpleNamespace(
                id=1,
                type=SimpleNamespace(value="large_airport"),
                icao_code="KJFK",
                iata_code="JFK",
            ),
            SimpleNamespace(
                id=2,
                type=SimpleNamespace(value="small_airport"),
                icao_code=None,
                iata_code=None,
            ),
        )
    )
    airports = pl.DataFrame(
        {"airport_id": [1, 2, 999, None]},
        schema={"airport_id": pl.Int64},
    )

    result = airports.with_columns(
        airport_size_expr(airport_catalog),
        airport_has_icao_code_expr(airport_catalog),
        airport_has_iata_code_expr(airport_catalog),
    )

    assert result.to_dicts() == [
        {
            "airport_id": 1,
            "airport_size": "large_airport",
            "airport_has_icao_code": True,
            "airport_has_iata_code": True,
        },
        {
            "airport_id": 2,
            "airport_size": "small_airport",
            "airport_has_icao_code": False,
            "airport_has_iata_code": False,
        },
        {
            "airport_id": 999,
            "airport_size": None,
            "airport_has_icao_code": False,
            "airport_has_iata_code": False,
        },
        {
            "airport_id": None,
            "airport_size": None,
            "airport_has_icao_code": False,
            "airport_has_iata_code": False,
        },
    ]
