# functions to read csv and convert columns and types to mine

# funciton to get all icao information up to certain date.
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import warnings

import polars as pl

from data_engineering.utils import OUTPUT_DIR

OPENAIRFRAMES_DIR = OUTPUT_DIR / "data" / "raw" / "openairframes"
OPENAIRFRAMES_RELEASE_DATE = datetime.now(timezone.utc).date() - timedelta(days=2)
OPENAIRFRAMES_FAA_SNAPSHOT_END_DATE = OPENAIRFRAMES_RELEASE_DATE
OPENAIRFRAMES_ADSB_SNAPSHOT_END_DATE = OPENAIRFRAMES_RELEASE_DATE - timedelta(days=1)
DEFAULT_OPENAIRFRAMES_ADSB_PATH = (
    OPENAIRFRAMES_DIR
    / f"openairframes_adsb_2024-01-01_{OPENAIRFRAMES_ADSB_SNAPSHOT_END_DATE.isoformat()}.csv.gz"
)
DEFAULT_OPENAIRFRAMES_ADSB_URL = (
    "https://github.com/PlaneQuery/OpenAirframes/releases/download/"
    f"openairframes-{OPENAIRFRAMES_RELEASE_DATE.isoformat()}-main/"
    f"openairframes_adsb_2024-01-01_{OPENAIRFRAMES_ADSB_SNAPSHOT_END_DATE.isoformat()}.csv.gz"
)
DEFAULT_OPENAIRFRAMES_FAA_PATH = (
    OPENAIRFRAMES_DIR
    / f"openairframes_faa_2023-08-16_{OPENAIRFRAMES_FAA_SNAPSHOT_END_DATE.isoformat()}.csv"
)
DEFAULT_OPENAIRFRAMES_FAA_URL = (
    "https://github.com/PlaneQuery/OpenAirframes/releases/download/"
    f"openairframes-{OPENAIRFRAMES_RELEASE_DATE.isoformat()}-main/"
    f"openairframes_faa_2023-08-16_{OPENAIRFRAMES_FAA_SNAPSHOT_END_DATE.isoformat()}.csv"
)
OPENAIRFRAMES_RELEASES_API_URL = (
    "https://api.github.com/repos/PlaneQuery/OpenAirframes/releases?per_page=100"
)


@dataclass(frozen=True)
class _SnapshotAsset:
    release_date: date
    snapshot_date: date
    name: str
    download_url: str


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


def _latest_snapshot_path(dataset: str) -> Path | None:
    """Return the locally available snapshot with the newest trailing date."""
    filename_pattern = _snapshot_filename_pattern(dataset)
    snapshots = []
    for candidate in OPENAIRFRAMES_DIR.glob(f"openairframes_{dataset}_*.csv*"):
        match = filename_pattern.match(candidate.name)
        if match and candidate.is_file() and candidate.stat().st_size > 0:
            snapshots.append((match.group(1), candidate))

    return max(snapshots, key=lambda snapshot: snapshot[0], default=(None, None))[1]


def _snapshot_filename_pattern(dataset: str) -> re.Pattern[str]:
    return re.compile(
        rf"^openairframes_{re.escape(dataset)}_[^/]+_(\d{{4}}-\d{{2}}-\d{{2}})\.csv(?:\.gz)?$"
    )


@lru_cache(maxsize=None)
def _remote_snapshot_assets(dataset: str) -> tuple[_SnapshotAsset, ...]:
    """Return matching assets from newest eligible releases first."""
    request = urllib.request.Request(
        OPENAIRFRAMES_RELEASES_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "planequery-data",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        releases = json.load(response)
    if not isinstance(releases, list):
        raise ValueError("GitHub releases API did not return a release list")

    filename_pattern = _snapshot_filename_pattern(dataset)
    assets: list[_SnapshotAsset] = []
    seen_names: set[str] = set()
    for release in releases:
        if release.get("draft"):
            continue
        tag_match = re.fullmatch(
            r"openairframes-(\d{4}-\d{2}-\d{2})-main",
            release.get("tag_name", ""),
        )
        if not tag_match:
            continue
        release_date = date.fromisoformat(tag_match.group(1))
        if release_date > OPENAIRFRAMES_RELEASE_DATE:
            continue
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            match = filename_pattern.fullmatch(name)
            download_url = asset.get("browser_download_url")
            if not match or not download_url or name in seen_names:
                continue
            assets.append(
                _SnapshotAsset(
                    release_date=release_date,
                    snapshot_date=date.fromisoformat(match.group(1)),
                    name=name,
                    download_url=download_url,
                )
            )
            seen_names.add(name)

    return tuple(
        sorted(
            assets,
            key=lambda asset: (asset.release_date, asset.snapshot_date),
            reverse=True,
        )
    )


def _download_snapshot(asset: _SnapshotAsset, destination: Path) -> None:
    """Download an asset atomically so failed transfers are never used as data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        request = urllib.request.Request(
            asset.download_url,
            headers={"User-Agent": "planequery-data"},
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary_path.stat().st_size == 0:
            raise OSError(f"Downloaded empty OpenAirframes asset: {asset.name}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


@lru_cache(maxsize=None)
def _resolve_default_snapshot(dataset: str) -> Path:
    local_fallback = _latest_snapshot_path(dataset)
    try:
        assets = _remote_snapshot_assets(dataset)
    except (OSError, ValueError) as exc:
        if local_fallback is not None:
            warnings.warn(
                f"Could not check OpenAirframes releases ({exc}); "
                f"using local snapshot {local_fallback}",
                stacklevel=2,
            )
            return local_fallback
        raise RuntimeError(
            f"Could not discover an OpenAirframes {dataset.upper()} snapshot"
        ) from exc

    for asset in assets:
        destination = OPENAIRFRAMES_DIR / asset.name
        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        print(f"Downloading OpenAirframes {dataset.upper()} snapshot to {destination}")
        try:
            _download_snapshot(asset, destination)
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 410}:
                continue
            if local_fallback is not None:
                warnings.warn(
                    f"Could not download {asset.name} ({exc}); "
                    f"using local snapshot {local_fallback}",
                    stacklevel=2,
                )
                return local_fallback
            raise
        except OSError as exc:
            if local_fallback is not None:
                warnings.warn(
                    f"Could not download {asset.name} ({exc}); "
                    f"using local snapshot {local_fallback}",
                    stacklevel=2,
                )
                return local_fallback
            raise
        return destination

    if local_fallback is not None:
        warnings.warn(
            f"No downloadable OpenAirframes {dataset.upper()} asset was found; "
            f"using local snapshot {local_fallback}",
            stacklevel=2,
        )
        return local_fallback
    raise FileNotFoundError(
        f"No downloadable OpenAirframes {dataset.upper()} snapshot was found"
    )


def _ensure_default_openairframes_adsb_path(path: str | Path) -> Path:
    resolved_path = Path(path)
    if resolved_path != DEFAULT_OPENAIRFRAMES_ADSB_PATH:
        return resolved_path
    return _resolve_default_snapshot("adsb")


def _ensure_default_openairframes_faa_path(path: str | Path) -> Path:
    resolved_path = Path(path)
    if resolved_path != DEFAULT_OPENAIRFRAMES_FAA_PATH:
        return resolved_path
    return _resolve_default_snapshot("faa")


def _extract_db_flags(col: str = "db_flags") -> list[pl.Expr]:
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
    df = df.with_columns(_extract_db_flags(col="db_flags"))
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
        .with_columns(_extract_db_flags(col="db_flags"))
        .drop("db_flags")
        .select(["icao", *ICAO_INFO_COLUMNS])
    )
    return df


def scan_icao_info_as_of_for_join(
    at_date: date,
    path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH,
) -> pl.LazyFrame:
    """Return the latest OpenAirframes row per ICAO available by a date."""
    end = datetime.combine(at_date + timedelta(days=1), datetime.min.time())
    return (
        scan_icao_info(path)
        .filter(pl.col("time") < end)
        .sort(["icao", "time"])
        .group_by("icao", maintain_order=True)
        .last()
        .select(["icao", *ICAO_INFO_COLUMNS])
    )


def add_icao_info_as_of(
    df: pl.DataFrame,
    at_date: date,
    path: str | Path = DEFAULT_OPENAIRFRAMES_ADSB_PATH,
) -> pl.DataFrame:
    info_columns = ["pia", "ladd", "military", "interesting"]
    df = df.drop([col for col in info_columns if col in df.columns])
    df_openairframes = (
        scan_icao_info_as_of_for_join(at_date, path)
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


def get_openairframes_faa_df(
    path: str | Path = DEFAULT_OPENAIRFRAMES_FAA_PATH,
) -> pl.DataFrame:
    """Return the full FAA OpenAirframes snapshot as a Polars DataFrame."""
    path = _ensure_default_openairframes_faa_path(path)
    return pl.read_csv(
        path,
        schema_overrides={
            "transponder_code_hex": pl.Utf8,
            "registration_number": pl.Utf8,
        },
        ignore_errors=True
    )

# TODO: Maybe have this whole thing be directly using readsb database instead of this derived ADS-B thing. 
def get_openairframes_snapshot(
    at_date: date | None = None,
) -> pl.DataFrame:
    """Return one preferred OpenAirframes row per ICAO near ``at_date``.

    Rows with a registration are preferred over rows without one. Within that
    preference, the row whose timestamp is closest to ``at_date`` is selected.
    """
    if at_date is None:
        at_date = datetime.now(timezone.utc).date()
    at_datetime = datetime.combine(at_date, datetime.min.time())

    df =  (
        scan_icao_info()
        .with_columns(
            (pl.col("registration").is_null() | pl.col("registration").eq("")).alias("_registration_is_null"),
            (pl.col("time") - at_datetime).abs().alias("_distance_from_at_date"),
        )
        .sort(
            ["icao", "_registration_is_null", "_distance_from_at_date", "time"],
            descending=[False, False, False, False],
            nulls_last=True,
        )
        .group_by("icao", maintain_order=True)
        .first()
        .filter(~pl.col("_registration_is_null"))
        .drop("_registration_is_null", "_distance_from_at_date")
        .collect()
    )
    df = df.with_columns(pl.col("registration").str.strip_chars().str.to_uppercase())
    duplicates = df.group_by("registration").agg(pl.col("icao").n_unique().alias("icao_count")).filter(pl.col("icao_count") > 1)
    if not duplicates.is_empty():
        # TODO: Use frequency of appearnces to determine which one to keep. 
        df = df.filter(~pl.col("registration").is_in(duplicates.select("registration").to_series().implode()))
    return df
