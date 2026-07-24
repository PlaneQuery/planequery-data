# functions to read csv and convert columns and types to mine

# funciton to get all icao information up to certain date.
from pathlib import Path
import urllib.request

import polars as pl

from data_engineering.utils import OUTPUT_DIR

OPENAIRFRAMES_DIR = OUTPUT_DIR / "data" / "raw" / "openairframes"
DEFAULT_OPENAIRFRAMES_ADSB_PATH = OPENAIRFRAMES_DIR / "openairframes_adsb_2024-01-01_2026-07-05.csv.gz"
DEFAULT_OPENAIRFRAMES_ADSB_URL = (
    "https://github.com/PlaneQuery/OpenAirframes/releases/download/"
    "openairframes-2026-07-10-main/"
    "openairframes_adsb_2024-01-01_2026-07-05.csv.gz"
)
DEFAULT_OPENAIRFRAMES_FAA_PATH = OPENAIRFRAMES_DIR / "openairframes_faa_2023-08-16_2026-07-10.csv"
DEFAULT_OPENAIRFRAMES_FAA_URL = (
    "https://github.com/PlaneQuery/OpenAirframes/releases/download/"
    "openairframes-2026-07-10-main/openairframes_faa_2023-08-16_2026-07-10.csv"
)

COLUMN_MAP = {
    "r": "registration",
    "t": "aircraft_type",
    "ownOp": "owner",
    "desc": "aircraft_description",
    "dbFlags": "db_flags",
    "aircraft_category": "category",
}

ICAO_INFO_COLUMNS = [
    "registration",
    "aircraft_type",
    "owner",
    "aircraft_description",
    "category",
    "pia",
    "ladd",
    "military",
    "interesting",
]


def _ensure_default_openairframes_adsb_path(path: str | Path) -> Path:
    resolved_path = Path(path)
    if resolved_path != DEFAULT_OPENAIRFRAMES_ADSB_PATH or resolved_path.exists():
        return resolved_path

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OpenAirframes ADS-B snapshot to {resolved_path}")
    urllib.request.urlretrieve(DEFAULT_OPENAIRFRAMES_ADSB_URL, resolved_path)
    return resolved_path


def _ensure_default_openairframes_faa_path(path: str | Path) -> Path:
    resolved_path = Path(path)
    if resolved_path != DEFAULT_OPENAIRFRAMES_FAA_PATH or resolved_path.exists():
        return resolved_path

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OpenAirframes FAA snapshot to {resolved_path}")
    urllib.request.urlretrieve(DEFAULT_OPENAIRFRAMES_FAA_URL, resolved_path)
    return resolved_path


def extract_db_flags(col: str = "db_flags") -> list[pl.Expr]:
    return [
        (pl.col(col).cast(pl.Int64) & 1).cast(pl.Boolean).alias("military"),
        (pl.col(col).cast(pl.Int64) & 2).cast(pl.Boolean).alias("interesting"),
        (pl.col(col).cast(pl.Int64) & 4).cast(pl.Boolean).alias("pia"),
        (pl.col(col).cast(pl.Int64) & 8).cast(pl.Boolean).alias("ladd"),
    ]


def scan_icao_info(path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH) -> pl.LazyFrame:
    path = _ensure_default_openairframes_adsb_path(path)
    df = pl.scan_csv(
        path,
        schema_overrides={
            "icao": pl.Utf8,
            "dbFlags": pl.Int64,
        },
        low_memory=True,
    )
    df = df.rename(COLUMN_MAP)
    df = df.with_columns(
        pl.col("time")
        .str.strptime(pl.Datetime("ms"), strict=False)
        .dt.replace_time_zone(None)
        .alias("time")
    )
    df = df.with_columns(extract_db_flags(col="db_flags"))
    df = df.drop("db_flags")
    return df


def scan_latest_icao_info_for_join(
    path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH,
) -> pl.LazyFrame:
    """Scan one latest OpenAirframes row per ICAO for ADS-B enrichment joins.

    The OpenAirframes CSV is chronological, so keeping the last row per ICAO
    avoids a multi-GB time sort before joining.
    """
    path = _ensure_default_openairframes_adsb_path(path)
    source_columns = ["icao", "r", "t", "ownOp", "desc", "aircraft_category", "dbFlags"]
    df = pl.scan_csv(
        path,
        schema_overrides={
            "icao": pl.Utf8,
            "dbFlags": pl.Int64,
        },
        low_memory=True,
    )
    df = (
        df
        .select(source_columns)
        .rename(COLUMN_MAP)
        .unique(subset=["icao"], keep="last", maintain_order=False)
        .with_columns(extract_db_flags(col="db_flags"))
        .drop("db_flags")
        .select(["icao", *ICAO_INFO_COLUMNS])
    )
    return df


def add_latest_icao_info(
    df: pl.DataFrame,
    path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH,
) -> pl.DataFrame:
    info_columns = ["pia", "ladd", "military", "interesting"]
    df = df.drop([col for col in info_columns if col in df.columns])
    df_openairframes = (
        scan_latest_icao_info_for_join(path)
        .select(["icao", *info_columns])
        .with_columns(pl.col("icao").str.to_lowercase())
    )
    return (
        df.lazy()
        .with_columns(pl.col("icao").str.to_lowercase())
        .join(df_openairframes, on="icao", how="left")
        .with_columns(pl.col(info_columns).fill_null(False))
        .collect()
    )


def scan_latest_registration_icao_for_join(
    path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH,
    real_icao_column: str = "real_icao",
) -> pl.LazyFrame:
    """Scan one latest OpenAirframes ICAO per registration for tail-number joins."""
    path = _ensure_default_openairframes_adsb_path(path)
    df = pl.scan_csv(
        path,
        schema_overrides={
            "icao": pl.Utf8,
            "r": pl.Utf8,
        },
        low_memory=True,
    )
    df = (
        df.select(["icao", "r"])
        .rename({"r": "registration", "icao": real_icao_column})
        .with_columns(
            pl.col("registration").str.strip_chars().str.to_uppercase(),
            pl.col(real_icao_column).str.strip_chars().str.to_lowercase(),
        )
        .filter(pl.col("registration").is_not_null())
        .unique(subset=["registration"], keep="last", maintain_order=False)
    )
    return df


def scan_latest_faa_registration_icao_for_join(
    path: str | Path = DEFAULT_OPENAIRFRAMES_FAA_PATH,
    real_icao_column: str = "real_icao",
) -> pl.LazyFrame:
    """Scan one latest FAA OpenAirframes ICAO per registration for tail-number joins."""
    path = _ensure_default_openairframes_faa_path(path)
    df = pl.scan_csv(
        path,
        schema_overrides={
            "transponder_code_hex": pl.Utf8,
            "registration_number": pl.Utf8,
        },
        low_memory=True,
    )
    df = (
        df.select(["transponder_code_hex", "registration_number"])
        .rename(
            {
                "registration_number": "registration",
                "transponder_code_hex": real_icao_column,
            }
        )
        .with_columns(
            pl.col("registration").str.strip_chars().str.to_uppercase(),
            pl.col(real_icao_column).str.strip_chars().str.to_lowercase(),
        )
        .filter(pl.col("registration").is_not_null())
        .unique(subset=["registration"], keep="last", maintain_order=False)
    )
    return df


def get_icao_info(path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH) -> pl.DataFrame:
    path = _ensure_default_openairframes_adsb_path(path)
    df = scan_icao_info(path)
    df = df.sort("time").group_by("icao").tail(1)
    return df.collect()
