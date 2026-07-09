from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from data_engineering.utils import OUTPUT_DIR

ADSBLOL_PARQUET_ROOT = OUTPUT_DIR / "data/raw/adsblol/parquet_output/v6"
ADSBLOL_LEGACY_PARQUET_ROOT = OUTPUT_DIR / "data/raw/adsblol/parquet_output/v5"
ADSBX_PARQUET_ROOT = OUTPUT_DIR / "data/raw/adsb-exchange/traces_parquet"
ICAO_BUCKET_COUNT = 8
ICAO_BUCKET_SIZE = 0x1000000 // ICAO_BUCKET_COUNT
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


def _adsbx_parquet_day_dir(current_day: date) -> Path:
    return _day_partition_dir(ADSBX_PARQUET_ROOT, current_day)


def _adsbx_parquet_path(current_day: date) -> str:
    return str(_adsbx_parquet_day_dir(current_day) / "icao_bucket=*" / "data.parquet")


def _legacy_parquet_day_dir(current_day: date) -> Path:
    return _day_partition_dir(ADSBLOL_LEGACY_PARQUET_ROOT, current_day)


def _legacy_parquet_path(current_day: date) -> Path:
    return _legacy_parquet_day_dir(current_day) / "part-00000.parquet"


def _parquet_partition_has_data(partition_dir: Path) -> bool:
    return any(partition_dir.glob("icao_bucket=*/data.parquet"))


def _parquet_day_has_data(current_day: date) -> bool:
    return (
        _parquet_partition_has_data(_parquet_day_dir(current_day))
        or _legacy_parquet_path(current_day).exists()
    )


def _icao_bucket(icao: str) -> int:
    if icao.startswith("~"):
        return ICAO_BUCKET_COUNT - 1
    icao_int = int(icao.lower(), 16)
    return min(icao_int // ICAO_BUCKET_SIZE, ICAO_BUCKET_COUNT - 1)


def _parquet_read_paths(
    current_day: date,
    icao_buckets: Sequence[int] | None = None,
) -> str | list[str]:
    day_dir = _parquet_day_dir(current_day)
    if _parquet_partition_has_data(day_dir):
        if icao_buckets is None:
            return str(day_dir / "icao_bucket=*" / "data.parquet")
        paths = [
            day_dir / f"icao_bucket={bucket}" / "data.parquet"
            for bucket in sorted(set(icao_buckets))
        ]
        return [str(path) for path in paths if path.exists()]

    legacy_path = _legacy_parquet_path(current_day)
    if legacy_path.exists():
        return str(legacy_path)

    return []


def _query_buckets(
    *,
    icaos: Sequence[str] | None = None,
) -> list[int] | None:
    if icaos is not None:
        return sorted({_icao_bucket(icao) for icao in icaos})
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


def create_adsb_parquet_for_day(
    current_day: date,
    *,
    pia_or_american_ladd_only: bool = False,
):
    if _parquet_day_has_data(current_day):
        return

    from data_engineering.adsb.download_adsb_to_parquet import create_parquet_for_day
    create_parquet_for_day(
        current_day,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
    )


def read_adsb(
    current_day: date,
    columns=DEFAULT_COLUMNS_SET,
    additional_columns=None,
    *,
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
):
    return scan_adsb(
        current_day,
        columns=columns,
        additional_columns=additional_columns,
        icaos=icaos,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
    ).collect()


def read_adsbx(
    current_day: date,
    columns=DEFAULT_COLUMNS_SET,
    additional_columns=None,
    *,
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
):
    columns = _column_list(columns, additional_columns)
    filter_columns = ["pia", "ladd"] if pia_or_american_ladd_only else []
    read_columns = list(dict.fromkeys(columns + ["time", "icao", *filter_columns]))
    if not _parquet_partition_has_data(_adsbx_parquet_day_dir(current_day)):
        return pl.DataFrame(
            schema={column: EMPTY_COLUMN_DTYPES.get(column, pl.Null) for column in columns}
        )

    df = pl.read_parquet(_adsbx_parquet_path(current_day), columns=read_columns)
    df = df.filter(pl.col("time").dt.date() == current_day)
    if icaos is not None:
        df = df.filter(pl.col("icao").is_in(list(icaos)))
    if pia_or_american_ladd_only:
        df = df.filter(pia_or_american_ladd_icao())
    if "callsign" in columns:
        df = df.with_columns(pl.col("callsign").str.replace_all(" ","").alias("callsign"))
    return df.select(columns)


def scan_adsb(
    current_day: date,
    columns=DEFAULT_COLUMNS_SET,
    additional_columns=None,
    *,
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
):
    create_adsb_parquet_for_day(
        current_day,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
    )
    columns = _column_list(columns, additional_columns)
    filter_columns = ["pia", "ladd"] if pia_or_american_ladd_only else []
    scan_columns = list(dict.fromkeys(columns + ["time", "icao", *filter_columns]))
    parquet_paths = _parquet_read_paths(
        current_day,
        _query_buckets(icaos=icaos),
    )
    if not parquet_paths:
        return _empty_scan(columns)

    lf = pl.scan_parquet(parquet_paths).select(scan_columns)
    lf = lf.filter(pl.col("time").dt.date() == current_day)
    if icaos is not None:
        lf = lf.filter(pl.col("icao").is_in(list(icaos)))
    if pia_or_american_ladd_only:
        lf = lf.filter(pia_or_american_ladd_icao())
    if "callsign" in columns:
        lf = lf.with_columns(pl.col("callsign").str.replace_all(" ","").alias("callsign"))
    return lf.select(columns)


def read_icaos_from_adsb(current_day: date, icaos: list[str], columns=DEFAULT_COLUMNS_SET):
    return read_adsb(current_day, columns=columns, icaos=icaos)

def read_adsb_lz(start_dt: datetime, end_dt: datetime, icaos: list[str], columns=DEFAULT_COLUMNS_SET):
    '''icaos is sorted'''
    current_day = start_dt.date()
    end_day = end_dt.date()
    dates = [current_day + timedelta(days=i) for i in range((end_day - current_day).days + 1)]
    for day in dates:
        create_adsb_parquet_for_day(day)
    first_bucket = _icao_bucket(icaos[0])
    last_bucket = _icao_bucket(icaos[-1])
    icao_buckets = list(range(first_bucket, last_bucket + 1))
    paths = []
    for day in dates:
        paths.extend(_parquet_read_paths(day, icao_buckets=icao_buckets))
    df_lz = pl.scan_parquet(paths).filter(pl.col("icao").is_in(icaos)).filter((pl.col("time") >= start_dt) & (pl.col("time") < end_dt)).select(columns)
    return df_lz
def get_icaos_in_adsb(current_day: date) -> list[str]:
    '''
    Filters invalid icaos
    '''
    df = scan_adsb(current_day, columns=["icao"]).filter(exclude_invalid_icaos()).unique().collect()
    return df["icao"].to_list()

def is_american_icao(col: str = "icao") -> pl.Expr:
    icao = pl.col(col).str.to_lowercase()
    return (icao >= "a00000") & (icao <= "afffff")


def pia_or_american_ladd() -> pl.Expr:
    return (
        pl.col("pia").fill_null(False)
        | (is_american_icao() & pl.col("ladd").fill_null(False))
    )


def pia_or_american_ladd_icao() -> pl.Expr:
    return pia_or_american_ladd().any().over("icao")


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
