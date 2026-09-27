import fcntl
import os
import re
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import urllib.request
import time
from data_engineering.utils import OUTPUT_DIR
ADSBLOL_DIR = OUTPUT_DIR / "data" / "raw" /"adsblol"
PARQUET_DIR = ADSBLOL_DIR / "parquet_output" / "v6"
CACHE_MAX_AGE_SECONDS = 60 * 60 * 12
LATEST_PROD_ADSB_DATE_CACHE = ADSBLOL_DIR / "LATEST_PROD_ADSB_DATE.txt"
DEFAULT_LATEST_PROD_ADSB_DATE = date(2026, 6, 4)
MIN_DAILY_ADSB_ROWS = 1_000
PREPARED_RELEASE_YEARS_ENV = "PLANEQUERY_PREPARED_RELEASE_YEARS"
_RUN_RELEASES_CACHE: dict[str, str] = {}
RELEASE_TAG_RE = re.compile(
    r"v\d{4}\.\d{2}\.\d{2}-planes-readsb-(?:prod|staging|mlatonly)-\d+(?:tmp)?"
)


def _cache_is_fresh(cache_path: str | Path) -> bool:
    return time.time() - Path(cache_path).stat().st_mtime <= CACHE_MAX_AGE_SECONDS


@contextmanager
def _cache_lock(cache_path: Path):
    """Serialize refreshes so parallel day workers do not race on the cache."""
    lock_path = cache_path.with_name(f".{cache_path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete cache file without exposing a partially written one."""
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _request_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _prepared_release_years() -> set[str]:
    return {
        year
        for year in os.getenv(PREPARED_RELEASE_YEARS_ENV, "").split(",")
        if year
    }


def prepare_releases_for_run(years: Iterable[str | int]) -> None:
    """Refresh each release catalog once, before spawning any day workers."""
    prepared_years = _prepared_release_years()
    for year in sorted({str(year) for year in years}):
        load_releases_txt(year)
        prepared_years.add(year)
    os.environ[PREPARED_RELEASE_YEARS_ENV] = ",".join(sorted(prepared_years))


def _extract_release_tags_from_atom(content: str) -> list[str]:
    return list(dict.fromkeys(RELEASE_TAG_RE.findall(content)))


def _release_line_from_tag(year: str, tag: str) -> str:
    base_url = f"https://github.com/adsblol/globe_history_{year}/releases/download/{tag}/{tag}.tar"
    if "-mlatonly-" in tag:
        return base_url
    return f"{base_url}.aa,{base_url}.ab"


def _merge_atom_releases_into_cache(year: str, cache_path: str, content: str) -> str:
    atom_url = f"https://github.com/adsblol/globe_history_{year}/releases.atom"
    missing_tags = [
        tag
        for tag in _extract_release_tags_from_atom(_request_text(atom_url))
        if tag not in content
    ]
    if not missing_tags:
        return content

    new_content = "\n".join(_release_line_from_tag(year, tag) for tag in missing_tags)
    updated_content = f"{new_content}\n{content.rstrip()}\n"

    _atomic_write_text(cache_path, updated_content)
    print(f"[CACHE] Added {len(missing_tags)} releases from releases.atom to ALL_RELEASES_{year}.txt")
    return updated_content


def load_releases_txt(year: str) -> str:
    """Return the content of ALL_RELEASES.txt for the given year.

    Refreshes a local cache at {OUTPUT_DIR}/ALL_RELEASES_{year}.txt and
    augments it with releases.atom entries that are newer than ALL_RELEASES.txt.
    """
    ADSBLOL_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = ADSBLOL_DIR / f"ALL_RELEASES_{year}.txt"

    if year in _RUN_RELEASES_CACHE:
        return _RUN_RELEASES_CACHE[year]

    # Spawned workers inherit this run-scoped marker from their parent. The
    # parent already refreshed and augmented the snapshot before starting them.
    if year in _prepared_release_years() and cache_path.exists():
        content = cache_path.read_text(encoding="utf-8")
        _RUN_RELEASES_CACHE[year] = content
        return content

    with _cache_lock(cache_path):
        # Recheck after taking the lock: another worker may just have refreshed it.
        cache_exists = cache_path.exists()
        if cache_exists and _cache_is_fresh(cache_path):
            print(f"[CACHE] Using cached ALL_RELEASES_{year}.txt")
            content = cache_path.read_text(encoding="utf-8")
        else:
            if cache_exists:
                print(f"[CACHE] Cached ALL_RELEASES_{year}.txt is older than 12 hours; redownloading")
            txt_url = f"https://raw.githubusercontent.com/adsblol/globe_history_{year}/refs/heads/main/ALL_RELEASES.txt"
            content = _request_text(txt_url)
            _atomic_write_text(cache_path, content)
            print(f"Downloaded and cached ALL_RELEASES_{year}.txt")

        content = _merge_atom_releases_into_cache(year, cache_path, content)
        _RUN_RELEASES_CACHE[year] = content
        return content


def _extract_prod_version_dates_from_text(content: str) -> set[str]:
    """Return a set of version strings like `vYYYY.MM.DD` found in the
    provided ALL_RELEASES.txt content for prod releases.
    """
    # Find full release tags (with optional trailing 'tmp') and then
    # extract the date portion `vYYYY.MM.DD`.
    tags = set(re.findall(r"v\d{4}\.\d{2}\.\d{2}-planes-readsb-prod-\d+(?:tmp)?", content))
    versions: set[str] = set()
    for tag in tags:
        m = re.match(r"(v\d{4}\.\d{2}\.\d{2})", tag)
        if m:
            versions.add(m.group(1))
    return versions


def _parse_version_to_date(v: str) -> date | None:
    try:
        return datetime.strptime(v.lstrip("v"), "%Y.%m.%d").date()
    except Exception:
        return None


def get_latest_prod_release_date(latest_date: date) -> list[date]:
    if latest_date.month == 1:
        years_to_check = [latest_date.year, latest_date.year - 1]
    elif latest_date.month == 12:
        years_to_check = [latest_date.year + 1, latest_date.year]
    else:
        years_to_check = [latest_date.year]
    
    found_dates = set()

    for y in years_to_check:
        content = load_releases_txt(str(int(y)))
        if not content:
            continue
        versions = _extract_prod_version_dates_from_text(content)
        for v in versions:
            d = _parse_version_to_date(v)
            if d and d > latest_date:
                found_dates.add(d)

    found_dates = list(found_dates)
    found_dates.sort()
    return found_dates


def _read_latest_table_date_cache() -> date | None:
    if not LATEST_PROD_ADSB_DATE_CACHE.exists():
        return None
    return datetime.strptime(
        LATEST_PROD_ADSB_DATE_CACHE.read_text(encoding="utf-8").strip(),
        "%Y-%m-%d",
    ).date()


def _write_latest_table_date_cache(latest_date: date) -> None:
    ADSBLOL_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PROD_ADSB_DATE_CACHE.write_text(f"{latest_date:%Y-%m-%d}\n", encoding="utf-8")


def _cached_latest_table_date_or_default() -> date:
    latest_date = _read_latest_table_date_cache()
    if latest_date is None:
        latest_date = DEFAULT_LATEST_PROD_ADSB_DATE
        _write_latest_table_date_cache(latest_date)
        print("Seeded latest prod ADS-B date cache with", latest_date)
    elif latest_date < DEFAULT_LATEST_PROD_ADSB_DATE:
        print(
            "Cached latest prod ADS-B date is",
            latest_date,
            "which is older than default",
            DEFAULT_LATEST_PROD_ADSB_DATE,
        )
        latest_date = DEFAULT_LATEST_PROD_ADSB_DATE
        _write_latest_table_date_cache(latest_date)
    else:
        print("Cached latest prod ADS-B date is", latest_date)
    return latest_date


def _prod_adsb_row_count_for_day(spark, day: date) -> int:
    next_day = day + timedelta(days=1)
    sql = f"""
    SELECT COUNT(*) AS row_count
    FROM glue.planequery_lake_prod.adsb_messages
    WHERE time >= TIMESTAMP '{day:%Y-%m-%d} 00:00:00'
      AND time < TIMESTAMP '{next_day:%Y-%m-%d} 00:00:00'
    """
    return spark.sql(sql).collect()[0]["row_count"]


def latest_table_data_date(latest_date: date | None = None):
    if latest_date is None:
        latest_date = _cached_latest_table_date_or_default()
    probe_date = latest_date + timedelta(days=1)
    if probe_date >= datetime.now(timezone.utc).date():
        print("Latest data in prod is from", latest_date)
        return latest_date

    from data_engineering.spark import get_spark_session
    spark = get_spark_session(app_name="check_adsblol_new_release")
    try:
        while probe_date < datetime.now(timezone.utc).date():
            row_count = _prod_adsb_row_count_for_day(spark, probe_date)
            print(f"Rows in prod ADS-B table on {probe_date}: {row_count}")
            if row_count <= MIN_DAILY_ADSB_ROWS:
                break
            latest_date = probe_date
            _write_latest_table_date_cache(latest_date)
            probe_date += timedelta(days=1)
    finally:
        spark.stop()

    print("Latest data in prod is from", latest_date)
    return latest_date

def get_adsblol_releases_to_process():
    cached_latest_table_date = _cached_latest_table_date_or_default()
    release_dates = get_latest_prod_release_date(cached_latest_table_date)
    if not release_dates:
        print("No prod adsb.lol releases newer than cached latest prod ADS-B date.")
        return []

    latest_table_date = latest_table_data_date(cached_latest_table_date)
    return [day for day in release_dates if day > latest_table_date]


if __name__ == "__main__":
    latest = get_adsblol_releases_to_process()
    print(latest)
# we could return dates that are past. 
