from dataclasses import dataclass
from datetime import date

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.eurocontrol.preprocess import prepare_eurocontrol_month_parquet
from data_engineering.openairframes.read import get_openairframes_snapshot
from flights.flight_source import NonADSBFlightSource
from flights.flight_type import with_flight_schema_columns


def create_eurocontrol_flights(
    target_date: date,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    df = (
        pl.scan_parquet(prepare_eurocontrol_month_parquet(target_date))
        .filter(pl.col("takeoff_time").dt.date() == target_date)
        .collect()
    )
    df_openairframes = (
        get_openairframes_snapshot(at_date=target_date)
        .select("registration", "icao")
        .with_columns(
            pl.col("registration")
            .str.strip_chars()
            .str.to_uppercase()
            .str.replace_all("-", "")
            .alias("registration_norm")
        )
        .with_columns(pl.len().over("registration_norm").alias("_match_count"))
        .filter(pl.col("_match_count") == 1)
        .drop("_match_count")
    )
    df = df.join(
        df_openairframes,
        left_on="registration",
        right_on="registration_norm",
        how="inner",
        validate="m:1",
    )

    codes = (
        pl.concat(
            [
                df.select(pl.col("takeoff_airport_code").alias("code")),
                df.select(pl.col("landing_airport_code").alias("code")),
            ]
        )
        .get_column("code")
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    airports = []
    for code in codes:
        airport = airport_catalog.resolve_code(
            code,
            code_priority=(
                "icao_code",
                "ident",
                "gps_code",
                "iata_code",
                "local_code",
            ),
            prioritize_faa_airspace=False,
        )
        if airport is None:
            raise ValueError(f"Unknown Eurocontrol airport code {code!r}")
        airports.append(airport)
    mapping = pl.DataFrame(
        {
            "code": codes,
            "airport_id": [airport.id for airport in airports],
            "airport_ident": [airport.ident for airport in airports],
        },
        schema={
            "code": pl.String,
            "airport_id": pl.Int64,
            "airport_ident": pl.String,
        },
    )
    df = df.join(
        mapping,
        left_on="takeoff_airport_code",
        right_on="code",
        how="left",
    ).rename(
        {
            "airport_id": "takeoff_airport_id",
            "airport_ident": "takeoff_airport_ident",
        }
    )
    df = df.join(
        mapping,
        left_on="landing_airport_code",
        right_on="code",
        how="left",
    ).rename(
        {
            "airport_id": "landing_airport_id",
            "airport_ident": "landing_airport_ident",
        }
    )
    df = df.select(
        "icao",
        "registration",
        "takeoff_time",
        "landing_time",
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
        "aircraft_type",
        "owner",
    )
    return with_flight_schema_columns(df)


@dataclass(frozen=True)
class EurocontrolFlightSource(NonADSBFlightSource):
    name: str = "eurocontrol"
    version: int = 3

    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return create_eurocontrol_flights(dt, airport_catalog)
