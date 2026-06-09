# functions to read csv and convert columns and types to mine

# funciton to get all icao information up to certain date.
from pathlib import Path

import polars as pl

DEFAULT_OPENAIRFRAMES_ADSB_PATH = Path(
    "/Volumes/T2-SSD/planequery/data/raw/openairframes/openairframes_adsb_2024-01-01_2026-05-15.csv"
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


def extract_db_flags(col: str = "db_flags") -> list[pl.Expr]:
    return [
        (pl.col(col).cast(pl.Int64) & 1).cast(pl.Boolean).alias("military"),
        (pl.col(col).cast(pl.Int64) & 2).cast(pl.Boolean).alias("interesting"),
        (pl.col(col).cast(pl.Int64) & 4).cast(pl.Boolean).alias("pia"),
        (pl.col(col).cast(pl.Int64) & 8).cast(pl.Boolean).alias("ladd"),
    ]


def scan_icao_info(path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH) -> pl.LazyFrame:
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


def get_icao_info(path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH) -> pl.DataFrame:
    df = scan_icao_info(path)
    df = df.sort("time").group_by("icao").tail(1)
    return df.collect()
