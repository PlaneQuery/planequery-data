from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

import polars as pl

from data_engineering.utils import OUTPUT_DIR


FLIGHT_POLARS_SCHEMA = {
    "icao": pl.String,
    "callsign": pl.String,
    "registration": pl.String,
    "takeoff_time": pl.Datetime("ms"),
    "takeoff_airport_ident": pl.String,
    "landing_time": pl.Datetime("ms"),
    "landing_airport_ident": pl.String,
    "first_message_time": pl.Datetime("ms"),
    "first_lat": pl.Float64,
    "first_lon": pl.Float64,
    "first_baro_altitude_ft": pl.Int64,
    "first_geom_altitude_ft": pl.Int64,
    "last_message_time": pl.Datetime("ms"),
    "last_lat": pl.Float64,
    "last_lon": pl.Float64,
    "last_baro_altitude_ft": pl.Int64,
    "last_geom_altitude_ft": pl.Int64,
    "pia": pl.Boolean,
    "ladd": pl.Boolean,
    "military": pl.Boolean,
    "interesting": pl.Boolean,
    "aircraft_type": pl.String,
    "owner": pl.String,
    "aircraft_description": pl.String,
    "category": pl.String,
}

FLIGHT_COLUMN_DEFAULTS = {
    "callsign": "",
    "registration": "",
    "pia": False,
    "ladd": False,
    "military": False,
    "interesting": False,
    "aircraft_type": "",
    "owner": "",
    "aircraft_description": "",
    "category": "",
}

def with_flight_schema_columns(df: pl.DataFrame) -> pl.DataFrame:
    missing_columns = [
        pl.lit(FLIGHT_COLUMN_DEFAULTS.get(column_name), dtype=dtype).alias(column_name)
        for column_name, dtype in FLIGHT_POLARS_SCHEMA.items()
        if column_name not in df.columns
    ]
    if missing_columns:
        df = df.with_columns(missing_columns)
    return df.select(
        pl.col(column_name).cast(dtype).alias(column_name)
        for column_name, dtype in FLIGHT_POLARS_SCHEMA.items()
    )

def add_flight_duration_col(df: pl.DataFrame):
    df = df.with_columns((pl.col("landing_time") - pl.col("takeoff_time")).alias("flight_duration"))
    return df

@dataclass(eq=False)
class Flight:
    icao: str
    takeoff_time: datetime
    landing_time: datetime
    takeoff_airport_ident: str
    landing_airport_ident: str
    first_message_time: datetime | None = None
    first_lat: float | None = None
    first_lon: float | None = None
    first_baro_altitude_ft: int | None = None
    first_geom_altitude_ft: int | None = None
    last_message_time: datetime | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    last_baro_altitude_ft: int | None = None
    last_geom_altitude_ft: int | None = None
    callsign: str = ""
    registration: str = ""
    pia: bool = False
    ladd: bool = False
    military: bool = False
    interesting: bool = False
    aircraft_type: str = ""
    owner: str = ""
    aircraft_description: str = ""
    category: str = ""


    @property
    def flight_id(self) -> str:
        return self.icao + "_" + self.takeoff_time.strftime("%Y-%m-%d_%H-%M")

    def __hash__(self) -> int:
        return hash(self.flight_id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Flight):
            return NotImplemented
        return self.flight_id == other.flight_id

def flights_df_to_flights(df: pl.DataFrame) -> list[Flight]:
    df = df.select([col for col in FLIGHT_POLARS_SCHEMA if col in df.columns])
    return [
        Flight(**row)
        for row in df.iter_rows(named=True)
    ]

def flights_to_flights_df(flights: list[Flight]) -> pl.DataFrame:
    df = pl.DataFrame([asdict(flight) for flight in flights], schema=FLIGHT_POLARS_SCHEMA)
    return df.select(list(FLIGHT_POLARS_SCHEMA.keys()))

FLIGHT_NUMBER_REGEX = r"^[A-Z]{3}\d{1,4}[A-Z]?$"
def flight_number_validation(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(
    pl.col("flight")
    .str.contains(FLIGHT_NUMBER_REGEX)
    )
    return df

def add_flight_id_col(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
    (
        pl.col("icao").cast(pl.Utf8)
        + "_"
        + pl.col("takeoff_time").dt.strftime("%Y-%m-%d_%H-%M")
    ).alias("flight_id")
    )
    return df


def flights_algorithm_output_path(target_date: date, source: str = "adsblol", no_airports_model: bool = False):
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / ("algorithm-no-airports" if no_airports_model else "algorithm")
        / "v1"
        / source
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / f"day={target_date.day:02d}"
        / "flights.parquet"
    )


def _date_range(start_date: date, end_date: date):
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += timedelta(days=1)


def _get_flights_for_day(target_date: date, no_airports_model: bool = False):
    path = flights_algorithm_output_path(target_date, no_airports_model=no_airports_model)
    df = pl.read_parquet(path)
    return with_flight_schema_columns(df)


def get_flights(start_date: date, end_date: date | None = None, no_airports_model: bool = False):
    if end_date is None:
        return _get_flights_for_day(start_date, no_airports_model=no_airports_model)
    if end_date <= start_date:
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    return pl.concat([
        _get_flights_for_day(target_date, no_airports_model=no_airports_model)
        for target_date in _date_range(start_date, end_date)
    ])

def add_lat_lon_to_airport_ident_columns(df: pl.DataFrame) -> pl.DataFrame:
    from airports.airport_lookup import AirportLookup

    airport_lookup = AirportLookup()

    def add_airport_coords(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        ident_col = f"{prefix}_airport_ident"
        lat_col = f"{prefix}_airport_lat"
        lon_col = f"{prefix}_airport_lon"

        def get_coords(airport_ident: str | None) -> dict[str, float | None]:
            coords = airport_lookup.get_airport_coordinates(airport_ident)
            if coords is None:
                return {lat_col: None, lon_col: None}

            lat, lon = coords
            return {lat_col: lat, lon_col: lon}

        return (
            df.with_columns(
                pl.col(ident_col)
                .map_elements(
                    get_coords,
                    return_dtype=pl.Struct({
                        lat_col: pl.Float64,
                        lon_col: pl.Float64,
                    }),
                )
                .alias("_airport_coords")
            )
            .unnest("_airport_coords")
        )

    df = add_airport_coords(df, "takeoff")
    return add_airport_coords(df, "landing")

def _is_american_icao_expr() -> pl.Expr:
    icao = pl.col("icao").str.to_lowercase()
    return (icao >= "a00000") & (icao <= "afffff")


def _pia_or_american_ladd_filter() -> pl.Expr:
    return (
        pl.col("pia").fill_null(False)
        | (_is_american_icao_expr() & pl.col("ladd").fill_null(False))
    )


def pia_or_american_ladd_icao_filter() -> pl.Expr:
    return _pia_or_american_ladd_filter().any().over("icao")
