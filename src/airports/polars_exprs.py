import polars as pl

from airports.catalog import AirportCatalogView


def _airport_attribute_col(airport_id_col: str, attribute: str) -> str:
    return f"{airport_id_col.removesuffix('_id')}_{attribute}"


def _airport_has_code_expr(
    airport_catalog: AirportCatalogView,
    airport_id_col: str,
    code_attribute: str,
) -> pl.Expr:
    values = {
        airport.id: getattr(airport, code_attribute) is not None
        for airport in airport_catalog.airports
    }
    return pl.col(airport_id_col).replace_strict(
        values,
        default=False,
        return_dtype=pl.Boolean,
    ).alias(_airport_attribute_col(airport_id_col, f"has_{code_attribute}"))


def airport_size_expr(
    airport_catalog: AirportCatalogView,
    airport_id_col: str = "airport_id",
) -> pl.Expr:
    values = {airport.id: airport.type.value for airport in airport_catalog.airports}
    return pl.col(airport_id_col).replace_strict(
        values,
        default=None,
        return_dtype=pl.String,
    ).alias(_airport_attribute_col(airport_id_col, "size"))


def airport_has_icao_code_expr(
    airport_catalog: AirportCatalogView,
    airport_id_col: str = "airport_id",
) -> pl.Expr:
    return _airport_has_code_expr(
        airport_catalog,
        airport_id_col,
        "icao_code",
    )


def airport_has_iata_code_expr(
    airport_catalog: AirportCatalogView,
    airport_id_col: str = "airport_id",
) -> pl.Expr:
    return _airport_has_code_expr(
        airport_catalog,
        airport_id_col,
        "iata_code",
    )
