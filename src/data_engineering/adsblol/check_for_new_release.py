import re
from datetime import datetime, date, timedelta, timezone

import urllib.request
import time
from data_engineering.utils import OUTPUT_DIR
ADSBLOL_DIR = OUTPUT_DIR / "data" / "raw" /"adsblol"
PARQUET_DIR = ADSBLOL_DIR / "parquet_output" / "v5"
CACHE_MAX_AGE_SECONDS = 60 * 60 * 12
LATEST_PROD_ADSB_DATE_CACHE = ADSBLOL_DIR / "LATEST_PROD_ADSB_DATE.txt"
DEFAULT_LATEST_PROD_ADSB_DATE = date(2026, 6, 4)
MIN_DAILY_ADSB_ROWS = 1_000
RELEASE_TAG_RE = re.compile(
    r"v\d{4}\.\d{2}\.\d{2}-planes-readsb-(?:prod|staging|mlatonly)-\d+(?:tmp)?"
)


def _cache_created_at(cache_path) -> float:
    stat = cache_path.stat()
    return getattr(stat, "st_birthtime", stat.st_ctime)


def _cache_is_fresh(cache_path: str) -> bool:
    return time.time() - _cache_created_at(cache_path) <= CACHE_MAX_AGE_SECONDS


def _request_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


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

    with cache_path.open("w", encoding="utf-8") as f:
        f.write(updated_content)
    print(f"[CACHE] Added {len(missing_tags)} releases from releases.atom to ALL_RELEASES_{year}.txt")
    return updated_content


def load_releases_txt(year: str) -> str:
    """Return the content of ALL_RELEASES.txt for the given year.

    Refreshes a local cache at {OUTPUT_DIR}/ALL_RELEASES_{year}.txt and
    augments it with releases.atom entries that are newer than ALL_RELEASES.txt.
    """
    ADSBLOL_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = ADSBLOL_DIR / f"ALL_RELEASES_{year}.txt"

    cache_exists = cache_path.exists()
    if cache_exists and _cache_is_fresh(cache_path):
        print(f"[CACHE] Using cached ALL_RELEASES_{year}.txt")
        with cache_path.open("r", encoding="utf-8") as f:
            content = f.read()
    else:
        if cache_exists:
            print(f"[CACHE] Cached ALL_RELEASES_{year}.txt is older than 12 hours; redownloading")
        txt_url = f"https://raw.githubusercontent.com/adsblol/globe_history_{year}/refs/heads/main/ALL_RELEASES.txt"
        content = _request_text(txt_url)
        with cache_path.open("w", encoding="utf-8") as f:
            f.write(content)
        print(f"Downloaded and cached ALL_RELEASES_{year}.txt")

    return _merge_atom_releases_into_cache(year, cache_path, content)


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


def _prod_adsb_row_count_for_day(spark, day: date) -> int:
    next_day = day + timedelta(days=1)
    sql = f"""
    SELECT COUNT(*) AS row_count
    FROM glue.planequery_lake_prod.adsb_messages
    WHERE time >= TIMESTAMP '{day:%Y-%m-%d} 00:00:00'
      AND time < TIMESTAMP '{next_day:%Y-%m-%d} 00:00:00'
    """
    return spark.sql(sql).collect()[0]["row_count"]


def latest_table_data_date():
    from data_engineering.spark import get_spark_session
    spark = get_spark_session(app_name="check_adsblol_new_release")
    try:
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

        probe_date = latest_date + timedelta(days=1)
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
    latest_table_date = latest_table_data_date()
    release_dates = get_latest_prod_release_date(latest_table_date)
    return release_dates
if __name__ == "__main__":
    latest_table_date = latest_table_data_date()
    latest = get_latest_prod_release_date(latest_table_date)
    print(latest)
# we could return dates that are past. 
