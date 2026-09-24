from collections.abc import Sequence

import polars as pl

from airports.catalog import AirportCatalogView
from airports.models import Airport


def airport_from_reference(
    airport_catalog: AirportCatalogView,
    airport_id: int,
) -> Airport:
    """Resolve a flight endpoint from its stable OurAirports ID."""
    return airport_catalog.id_index[airport_id]


def resolve_flight_airport_codes(
    df: pl.DataFrame,
    airport_catalog: AirportCatalogView,
    *,
    takeoff_code_column: str,
    landing_code_column: str,
    code_priority: Sequence[str],
) -> pl.DataFrame:
    """Replace provider airport codes with stable IDs and canonical identifiers."""
    code_columns = {
        "takeoff": takeoff_code_column,
        "landing": landing_code_column,
    }
    codes = sorted(
        {
            code
            for column in code_columns.values()
            for code in df.get_column(column).drop_nulls().unique().to_list()
        }
    )
    resolved = {}
    for code in codes:
        airport = airport_catalog.resolve_code(
            code,
            code_priority=code_priority,
        )
        if airport is None:
            raise ValueError(f"Unknown airport code {code!r}")
        resolved[code] = airport
    code_to_id = {code: airport.id for code, airport in resolved.items()}
    code_to_ident = {
        code: airport.ident for code, airport in resolved.items()
    }

    return df.with_columns(
        *[
            expression
            for endpoint, code_column in code_columns.items()
            for expression in (
                pl.col(code_column)
                .replace_strict(
                    code_to_id,
                    default=None,
                    return_dtype=pl.Int64,
                )
                .alias(f"{endpoint}_airport_id"),
                pl.col(code_column)
                .replace_strict(
                    code_to_ident,
                    default=None,
                    return_dtype=pl.String,
                )
                .alias(f"{endpoint}_airport_ident"),
            )
        ]
    )
