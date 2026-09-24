import polars as pl

from airports.catalog import AirportCatalogView


def add_airspace_columns(
    df: pl.DataFrame,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    def in_us_faa_airspace(airport_id: int) -> bool:
        return airport_catalog.id_index[airport_id].is_in_us_faa_airspace

    def in_us_faa_island_airspace(airport_id: int) -> bool:
        return airport_catalog.id_index[airport_id].is_in_us_faa_island_airspace

    return df.with_columns(
        pl.col("takeoff_airport_id")
        .map_elements(in_us_faa_airspace, return_dtype=pl.Boolean)
        .alias("takeoff_airport_in_us_faa_airspace"),
        pl.col("landing_airport_id")
        .map_elements(in_us_faa_airspace, return_dtype=pl.Boolean)
        .alias("landing_airport_in_us_faa_airspace"),
        pl.col("takeoff_airport_id")
        .map_elements(in_us_faa_island_airspace, return_dtype=pl.Boolean)
        .alias("takeoff_airport_in_us_faa_island_airspace"),
        pl.col("landing_airport_id")
        .map_elements(in_us_faa_island_airspace, return_dtype=pl.Boolean)
        .alias("landing_airport_in_us_faa_island_airspace"),
    )


def us_faa_airspace_to_us_faa_airspace_expr() -> pl.Expr:
    return (
        pl.col("takeoff_airport_in_us_faa_airspace")
        & pl.col("landing_airport_in_us_faa_airspace")
    )


def any_us_faa_airspace_expr() -> pl.Expr:
    return (
        pl.col("takeoff_airport_in_us_faa_airspace")
        | pl.col("landing_airport_in_us_faa_airspace")
    )


def continental_usa_to_continental_usa(
    df: pl.DataFrame,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    df = add_airspace_columns(df, airport_catalog)
    condition = (
        pl.col("takeoff_airport_in_us_faa_airspace")
        & ~pl.col("takeoff_airport_in_us_faa_island_airspace")
        & pl.col("landing_airport_in_us_faa_airspace")
        & ~pl.col("landing_airport_in_us_faa_island_airspace")
    )
    return df.filter(condition)
