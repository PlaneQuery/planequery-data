import polars as pl
import os
from dataclasses import dataclass
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

def flights_df_to_flights(df: pl.DataFrame) -> list[Flight]:
    return [
        Flight(**row)
        for row in df.select(list(FLIGHT_POLARS_SCHEMA.keys())).iter_rows(named=True)
    ]

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
def get_flights(target_date: date):
    path = flights_parquet_path(target_date)
    if path.exists():
        df = pl.read_parquet(path)
        return df
    df = get_flights_from_iceberg(target_date)
    os.makedirs(path.parent, exist_ok=True)
    df.write_parquet(path)
    return df
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
    print(df.dtypes)
    for col_name in ["takeoff_time", "landing_time"]:
        df = df.with_columns(
            pl.col(col_name)
              .dt.replace_time_zone(None)   # remove UTC → becomes naive
              .dt.cast_time_unit("ms")      # ns → ms
              .alias(col_name)
        )
    return df
