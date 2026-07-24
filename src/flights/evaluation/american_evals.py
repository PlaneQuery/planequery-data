import polars as pl
from airports.airport_lookup import AirportLookup

airport_lookup = AirportLookup()

def add_airspace_columns(df: pl.DataFrame):
    df = df.with_columns(
        pl.col("takeoff_airport_ident")
        .map_elements(airport_lookup.airport_ident_is_in_us_faa_airspace, return_dtype=pl.Boolean)
        .alias("takeoff_airport_in_us_faa_airspace"),
        pl.col("landing_airport_ident")
        .map_elements(airport_lookup.airport_ident_is_in_us_faa_airspace, return_dtype=pl.Boolean)
        .alias("landing_airport_in_us_faa_airspace"),
        pl.col("takeoff_airport_ident")
        .map_elements(airport_lookup.airport_ident_is_in_us_faa_island_airspace, return_dtype=pl.Boolean)
        .alias("takeoff_airport_in_us_faa_island_airspace"),
        pl.col("landing_airport_ident")
        .map_elements(airport_lookup.airport_ident_is_in_us_faa_island_airspace, return_dtype=pl.Boolean)
        .alias("landing_airport_in_us_faa_island_airspace"),
    )
    return df

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

# Flight Sequencing Continetal USA -> Continetal USA. With SFDPS I can use the idents
def continental_usa_to_continental_usa(df: pl.DataFrame) -> pl.DataFrame:  # except Alaska?
    df = add_airspace_columns(df)
    condition = pl.col("takeoff_airport_in_us_faa_airspace") & ~pl.col("takeoff_airport_in_us_faa_island_airspace") & pl.col("landing_airport_in_us_faa_airspace") & ~pl.col("landing_airport_in_us_faa_island_airspace")
    return df.filter(condition)

# Flight Sequencing Continetal USA -> Outside
# Flight Sequencing Outside -> Continetal USA
