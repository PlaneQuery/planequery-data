import polars as pl
from datetime import datetime,date, timedelta
from pathlib import Path
import os

from data_engineering.utils import OUTPUT_DIR
ATHENA_DATABASE = "planequery_lake_dev"
ATHENA_TABLE = "adsb_messages"
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "planequery-workgroup-dev")


def _read_adsb_from_iceberg(current_day, columns):
    """Query `planequery_lake_dev.adsb_messages` via Athena for `current_day`."""
    import awswrangler as wr

    select_clause = ", ".join(columns)
    next_day = current_day + timedelta(days=1)
    sql = f"""
        SELECT {select_clause}
        FROM {ATHENA_TABLE}
        WHERE time >= TIMESTAMP '{current_day:%Y-%m-%d} 00:00:00'
          AND time <  TIMESTAMP '{next_day:%Y-%m-%d} 00:00:00'
    """
    # see datatypes I get back
    pdf = wr.athena.read_sql_query(
        sql=sql,
        database=ATHENA_DATABASE,
        workgroup=ATHENA_WORKGROUP,
        ctas_approach=False,
    )
    df = pl.from_pandas(pdf)
    df = df.with_columns(
    pl.col("time")
      .dt.replace_time_zone(None)   # remove UTC → becomes naive
      .dt.cast_time_unit("ms")      # ns → ms
    )
    return df


DEFAULT_COLUMNS_SET = columns=["time", "icao", "registration", "callsign", "lat", "lon", "baro_altitude_ft", "on_ground", "ground_speed_kt", "track_deg", "aircraft_type", "aircraft_description", "owner", "category", "pia", "ladd", "military", "interesting"]


def _day_partition_dir(root: Path, current_day: date) -> Path:
    return root / f"year={current_day.year}" / f"month={current_day.month:02d}" / f"day={current_day.day:02d}"


def _v6_parquet_partition_dir(current_day: date) -> Path:
    parquet_dir = OUTPUT_DIR / "data/raw/adsblol/parquet_output/v6"
    return _day_partition_dir(parquet_dir, current_day)


def _parquet_partition_has_data(partition_dir: Path) -> bool:
    return any(partition_dir.glob("icao_bucket=*/data.parquet"))


def _get_parquet_file_path(current_day: date):
    return _v6_parquet_partition_dir(current_day)


def _get_parquet_read_path(current_day: date) -> str | Path:
    return str(_get_parquet_file_path(current_day) / "icao_bucket=*" / "data.parquet")


def post_process_adsb_df(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("callsign").str.replace_all(" ",""))
    return df
def create_adsb_parquet_for_day(current_day: date):
    parquet_file_path = _get_parquet_file_path(current_day)
    if _parquet_partition_has_data(parquet_file_path):
        return

    from data_engineering.adsb.download_adsb_to_parquet import create_parquet_for_day
    create_parquet_for_day(current_day)
def read_adsb(current_day: date, columns=DEFAULT_COLUMNS_SET, additional_columns=[],):
    create_adsb_parquet_for_day(current_day)
    columns = list(set(columns + additional_columns))
    parquet_file_path = _get_parquet_read_path(current_day)
    df = pl.read_parquet(parquet_file_path, columns=columns)
    df = df.filter(pl.col("time").dt.date() == current_day)
    return post_process_adsb_df(df)

def scan_adsb(current_day: date, columns=DEFAULT_COLUMNS_SET, additional_columns=[]):
    create_adsb_parquet_for_day(current_day)
    columns = list(dict.fromkeys(columns + additional_columns))
    scan_columns = list(dict.fromkeys(columns + ["time"]))
    parquet_file_path = _get_parquet_read_path(current_day)
    lf = pl.scan_parquet(parquet_file_path).select(scan_columns)
    lf = lf.filter(pl.col("time").dt.date() == current_day)
    if "callsign" in columns:
        lf = lf.with_columns(pl.col("callsign").str.replace_all(" ","").alias("callsign"))
    return lf.select(columns)

def read_icaos_from_adsb(current_day: date, icaos: list[str], columns=DEFAULT_COLUMNS_SET):
    create_adsb_parquet_for_day(current_day)
    parquet_file_path = _get_parquet_read_path(current_day)
    df = pl.scan_parquet(parquet_file_path).filter(pl.col("icao").is_in(icaos)).select(columns).collect()
    return post_process_adsb_df(df)
def get_icaos_in_adsb(current_day: date) -> list[str]:
    '''
    Filters invalid icaos
    '''
    create_adsb_parquet_for_day(current_day)
    parquet_file_path = _get_parquet_read_path(current_day)
    df = pl.scan_parquet(parquet_file_path).select("icao").filter(exclude_invalid_icaos()).unique().collect()
    return df["icao"].to_list()

def is_american_icao(col: str = "icao") -> pl.Expr:
    return (pl.col(col) >= "a00000") & (pl.col(col) <= "afffff")

def exclude_invalid_icaos(col: str = "icao") -> pl.Expr:
    return (pl.col(col) >= "010000") & (pl.col(col) != "ffffff") & ~pl.col(col).str.starts_with("~")

def filter_ladd_icaos(col: str = "is_ladd") -> pl.Expr:
    return pl.col(col).any().over("icao")

def filter_fixed_wing_icaos(col: str = "emitter_category") -> pl.Expr:
    return pl.col(col).is_in(["A1", "A2", "A3", "A4", "A5"]).any().over("icao")

def most_common_category(col: str = "emitter_category") -> pl.Expr:
    return (
        pl.col(col)
        .filter(pl.col(col) != "")
        .mode()
        .first()
        .over("icao")
    )

def extract_db_flags(col: str = "db_flags") -> list[pl.Expr]:
    return [
        (pl.col(col).cast(pl.Int64) & 1).cast(pl.Boolean).alias("military"),
        (pl.col(col).cast(pl.Int64) & 2).cast(pl.Boolean).alias("interesting"),
        (pl.col(col).cast(pl.Int64) & 4).cast(pl.Boolean).alias("pia"),
        (pl.col(col).cast(pl.Int64) & 8).cast(pl.Boolean).alias("ladd"),
    ]

# we could fill in the dbflags column with most common to get whether it is ladd. 
if __name__ == "__main__":
    current_day = datetime(2025, 8, 1)
    df_adsb = read_adsb(current_day)
    print(df_adsb)
