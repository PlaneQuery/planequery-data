import polars as pl

from airports.airport_lookup import AirportLookup


def _airport_attribute_col(airport_ident_col: str, attribute: str) -> str:
    prefix = airport_ident_col.removesuffix("_ident")
    return f"{prefix}_{attribute}"


def airport_size_expr(
    airport_ident_col: str = "airport_ident",
) -> pl.Expr:
    airport_lookup = AirportLookup()
    size_by_ident = {
        ident: airport.type.value
        for ident, airport in airport_lookup.ident_index.items()
    }

    return pl.col(airport_ident_col).replace_strict(
        size_by_ident,
        default=None,
        return_dtype=pl.String,
    ).alias(_airport_attribute_col(airport_ident_col, "size"))


def airport_has_icao_code_expr(
    airport_ident_col: str = "airport_ident",
) -> pl.Expr:
    airport_lookup = AirportLookup()
    has_icao_code_by_ident = {
        ident: airport.icao_code is not None
        for ident, airport in airport_lookup.ident_index.items()
    }

    return pl.col(airport_ident_col).replace_strict(
        has_icao_code_by_ident,
        default=False,
        return_dtype=pl.Boolean,
    ).alias(_airport_attribute_col(airport_ident_col, "has_icao_code"))


def airport_has_iata_code_expr(
    airport_ident_col: str = "airport_ident",
) -> pl.Expr:
    airport_lookup = AirportLookup()
    has_iata_code_by_ident = {
        ident: airport.iata_code is not None
        for ident, airport in airport_lookup.ident_index.items()
    }

    return pl.col(airport_ident_col).replace_strict(
        has_iata_code_by_ident,
        default=False,
        return_dtype=pl.Boolean,
    ).alias(_airport_attribute_col(airport_ident_col, "has_iata_code"))
