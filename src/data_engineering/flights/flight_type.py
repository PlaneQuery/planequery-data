import polars as pl
import os
import hashlib
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from data_engineering.utils import FLIGHTS_VERSION, OUTPUT_DIR
FLIGHT_POLARS_SCHEMA = {
    "icao": pl.String,
    "callsign": pl.String,
    "registration": pl.String,
    "takeoff_time": pl.Datetime("ms"),
    "takeoff_airport_ident": pl.String,
    "landing_time": pl.Datetime("ms"),
    "landing_airport_ident": pl.String,
    "pia": pl.Boolean,
    "ladd": pl.Boolean,
    "military": pl.Boolean,
    "interesting": pl.Boolean,
    "aircraft_type": pl.String,
    "owner": pl.String,
    "aircraft_description": pl.String,
    "category": pl.String,
    "takeoff_airport_lat": pl.Float64,
    "takeoff_airport_lon": pl.Float64,
    "landing_airport_lat": pl.Float64,
    "landing_airport_lon": pl.Float64,
}

def add_flight_duration_col(df: pl.DataFrame):
    df = df.with_columns((pl.col("landing_time") - pl.col("takeoff_time")).alias("flight_duration"))
    return df

@dataclass
class Flight:
    icao: str
    takeoff_time: datetime
    landing_time: datetime
    takeoff_airport_ident: str
    landing_airport_ident: str
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
    takeoff_airport_lat: float | None = None
    takeoff_airport_lon: float | None = None
    landing_airport_lat: float | None = None
    landing_airport_lon: float | None = None

    @property
    def flight_id(self) -> str:
        return self.icao + "_" + self.takeoff_time.strftime("%Y-%m-%d_%H-%M")

@dataclass
class TestFlight():
    icao: str
    takeoff_time: datetime
    landing_time: datetime
    takeoff_airport_ident: str
    landing_airport_ident: str
    category: str = ""
    @property
    def flight_id(self) -> str:
        return self.icao + "_" + self.takeoff_time.strftime("%Y-%m-%d_%H-%M")
    @property
    def pia(self) -> bool:
        return self.icao.startswith("pia")
    @property
    def callsign(self) -> str:
        return hashlib.sha1(self.flight_id.encode()).hexdigest()[:8].upper()

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

def flights_parquet_path(target_date: date):
    FLIGHTS_OUTPUT_DIR = OUTPUT_DIR / "data" / "flights" / FLIGHTS_VERSION
    flights_path = FLIGHTS_OUTPUT_DIR / f"year={target_date.year}" / f"month={target_date.month:02d}" / f"day={target_date.day:02d}" / "part-0.parquet"
    return flights_path


def _date_range(start_date: date, end_date: date):
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += timedelta(days=1)


def _get_flights_for_day(target_date: date):
    path = flights_parquet_path(target_date)
    if path.exists():
        df = pl.read_parquet(path)
        return df
    df = get_flights_from_iceberg(target_date)
    os.makedirs(path.parent, exist_ok=True)
    df.write_parquet(path)
    return df


def get_flights(start_date: date, end_date: date | None = None):
    if end_date is None:
        return _get_flights_for_day(start_date)
    if end_date <= start_date:
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    return pl.concat([_get_flights_for_day(target_date) for target_date in _date_range(start_date, end_date)])


def get_flights_from_iceberg(target_date: date):
    import awswrangler as wr
    from data_engineering.utils import ATHENA_DATABASE, ATHENA_WORKGROUP
    ICEBERG_TABLE = "planequery_lake_prod.flights"
    next_day = target_date + timedelta(days=1)
    sql = f"""
        SELECT *
        FROM {ICEBERG_TABLE}
        WHERE takeoff_time >= TIMESTAMP '{target_date:%Y-%m-%d} 00:00:00'
          AND takeoff_time <  TIMESTAMP '{next_day:%Y-%m-%d} 00:00:00'
    """
    # see datatypes I get back
    pdf = wr.athena.read_sql_query(
        sql=sql,
        database=ATHENA_DATABASE,
        workgroup=ATHENA_WORKGROUP,
        ctas_approach=False,
    )
    df = pl.from_pandas(pdf)
    for col_name in ["takeoff_time", "landing_time"]:
        df = df.with_columns(
            pl.col(col_name)
              .dt.replace_time_zone(None)   # remove UTC → becomes naive
              .dt.cast_time_unit("ms")      # ns → ms
              .alias(col_name)
        )
    return df
