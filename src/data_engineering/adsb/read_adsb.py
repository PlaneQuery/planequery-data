from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import polars as pl

from data_engineering.utils import OUTPUT_DIR

ADSBLOL_PARQUET_ROOT = OUTPUT_DIR / "data/raw/adsblol/parquet_output/v6"
ICAO_BUCKET_COUNT = 8
ICAO_BUCKET_SIZE = 0x1000000 // ICAO_BUCKET_COUNT
AMERICAN_ICAO_BUCKET = 5
EMPTY_COLUMN_DTYPES = {
    "time": pl.Datetime("ms"),
    "icao": pl.String,
    "callsign": pl.String,
    "registration": pl.String,
    "aircraft_type": pl.String,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "baro_altitude_ft": pl.Int32,
    "geom_altitude_ft": pl.Int32,
    "ground_speed_kt": pl.Float32,
    "track_deg": pl.Float32,
    "on_ground": pl.Boolean,
    "pia": pl.Boolean,
    "ladd": pl.Boolean,
    "military": pl.Boolean,
    "interesting": pl.Boolean,
    "owner": pl.String,
    "aircraft_description": pl.String,
    "category": pl.String,
}


DEFAULT_COLUMNS_SET = [
    "time",
    "icao",
    "registration",
    "callsign",
    "lat",
    "lon",
    "baro_altitude_ft",
    "on_ground",
    "ground_speed_kt",
    "track_deg",
    "aircraft_type",
    "aircraft_description",
    "owner",
    "category",
    "pia",
    "ladd",
    "military",
    "interesting",
]


def _day_partition_dir(root: Path, current_day: date) -> Path:
    return root / f"year={current_day.year}" / f"month={current_day.month:02d}" / f"day={current_day.day:02d}"


def _parquet_day_dir(current_day: date) -> Path:
    return _day_partition_dir(ADSBLOL_PARQUET_ROOT, current_day)


def _parquet_partition_has_data(partition_dir: Path) -> bool:
    return any(partition_dir.glob("icao_bucket=*/data.parquet"))


def _icao_bucket(icao: str) -> int:
    if icao.startswith("~"):
        return ICAO_BUCKET_COUNT - 1
    icao_int = int(icao.lower(), 16)
    return min(icao_int // ICAO_BUCKET_SIZE, ICAO_BUCKET_COUNT - 1)


def _is_american_icao_value(icao: str) -> bool:
    icao = icao.lower()
    return "a00000" <= icao <= "afffff"


def _parquet_read_paths(
    current_day: date,
    icao_buckets: Sequence[int] | None = None,
) -> str | list[str]:
    day_dir = _parquet_day_dir(current_day)
    if icao_buckets is None:
        return str(day_dir / "icao_bucket=*" / "data.parquet")
    paths = [
        day_dir / f"icao_bucket={bucket}" / "data.parquet"
        for bucket in sorted(set(icao_buckets))
    ]
    return [str(path) for path in paths if path.exists()]


def _query_buckets(
    *,
    icaos: Sequence[str] | None = None,
    american_icaos_only: bool = False,
) -> list[int] | None:
    if icaos is not None:
        query_icaos = list(icaos)
        if american_icaos_only:
            query_icaos = [icao for icao in query_icaos if _is_american_icao_value(icao)]
        return sorted({_icao_bucket(icao) for icao in query_icaos})
    if american_icaos_only:
        return [AMERICAN_ICAO_BUCKET]
    return None


def _column_list(
    columns: Sequence[str],
    additional_columns: Sequence[str] | None = None,
) -> list[str]:
    return list(dict.fromkeys(list(columns) + list(additional_columns or [])))


def _empty_scan(columns: Sequence[str]) -> pl.LazyFrame:
    return pl.DataFrame(
        schema={column: EMPTY_COLUMN_DTYPES.get(column, pl.Null) for column in columns}
    ).lazy()


def create_adsb_parquet_for_day(current_day: date):
    parquet_day_dir = _parquet_day_dir(current_day)
    if _parquet_partition_has_data(parquet_day_dir):
        return

    from data_engineering.adsb.download_adsb_to_parquet import create_parquet_for_day
    create_parquet_for_day(current_day)


def read_adsb(
    current_day: date,
    columns=DEFAULT_COLUMNS_SET,
    additional_columns=None,
    *,
    icaos: Sequence[str] | None = None,
    american_icaos_only: bool = False,
):
    return scan_adsb(
        current_day,
        columns=columns,
        additional_columns=additional_columns,
        icaos=icaos,
        american_icaos_only=american_icaos_only,
    ).collect()


def scan_adsb(
    current_day: date,
    columns=DEFAULT_COLUMNS_SET,
    additional_columns=None,
    *,
    icaos: Sequence[str] | None = None,
    american_icaos_only: bool = False,
):
    create_adsb_parquet_for_day(current_day)
    columns = _column_list(columns, additional_columns)
    scan_columns = list(dict.fromkeys(columns + ["time", "icao"]))
    parquet_paths = _parquet_read_paths(
        current_day,
        _query_buckets(icaos=icaos, american_icaos_only=american_icaos_only),
    )
    if not parquet_paths:
        return _empty_scan(columns)

    lf = pl.scan_parquet(parquet_paths).select(scan_columns)
    lf = lf.filter(pl.col("time").dt.date() == current_day)
    if icaos is not None:
        lf = lf.filter(pl.col("icao").is_in(list(icaos)))
    if american_icaos_only:
        lf = lf.filter(is_american_icao())
    if "callsign" in columns:
        lf = lf.with_columns(pl.col("callsign").str.replace_all(" ","").alias("callsign"))
    return lf.select(columns)


def read_icaos_from_adsb(current_day: date, icaos: list[str], columns=DEFAULT_COLUMNS_SET):
    return read_adsb(current_day, columns=columns, icaos=icaos)


def get_icaos_in_adsb(current_day: date) -> list[str]:
    '''
    Filters invalid icaos
    '''
    df = scan_adsb(current_day, columns=["icao"]).filter(exclude_invalid_icaos()).unique().collect()
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
