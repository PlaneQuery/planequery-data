import re
from datetime import datetime, date, timedelta
from typing import Iterable, Optional

# Reuse the existing cached downloader from the adsb module
import urllib.request
import urllib.error
import time
import os
from data_engineering.utils import OUTPUT_DIR
ADSBLOL_DIR = OUTPUT_DIR / "data" / "raw" /"adsblol"
PARQUET_DIR = ADSBLOL_DIR / "parquet_output" / "v5"
CACHE_MAX_AGE_SECONDS = 60 * 60 * 12


def _cache_created_at(cache_path: str) -> float:
    stat = os.stat(cache_path)
    return getattr(stat, "st_birthtime", stat.st_ctime)


def _cache_is_fresh(cache_path: str) -> bool:
    return time.time() - _cache_created_at(cache_path) <= CACHE_MAX_AGE_SECONDS
    
def load_releases_txt(year: str) -> str | None:
    """Return the content of ALL_RELEASES.txt for the given year.

    Checks a local cache at {OUTPUT_DIR}/ALL_RELEASES_{year}.txt first.
    If absent or older than 12 hours, downloads from GitHub and saves to the cache.
    """
    cache_path = os.path.join(ADSBLOL_DIR, f"ALL_RELEASES_{year}.txt")

    if os.path.exists(cache_path) and _cache_is_fresh(cache_path):
        print(f"[CACHE] Using cached ALL_RELEASES_{year}.txt")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    elif os.path.exists(cache_path):
        print(f"[CACHE] Cached ALL_RELEASES_{year}.txt is older than 12 hours; redownloading")

    txt_url = f"https://raw.githubusercontent.com/adsblol/globe_history_{year}/refs/heads/main/ALL_RELEASES.txt"
    max_retries = 10
    retry_delay = 60 * 5

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(txt_url)
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    content = response.read().decode('utf-8')
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"Downloaded and cached ALL_RELEASES_{year}.txt")
                    return content
                else:
                    print(f"Failed to fetch ALL_RELEASES.txt (attempt {attempt}/{max_retries}): {response.status}")
                    if attempt < max_retries:
                        print(f"Waiting {retry_delay} seconds before retry")
                        time.sleep(retry_delay)
                    else:
                        print(f"Giving up after {max_retries} attempts")
                        return None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"ALL_RELEASES.txt not found for year {year} repo")
                return None
            print(f"HTTP error (attempt {attempt}/{max_retries}): {e.code} {e.reason}")
            if attempt < max_retries:
                print(f"Waiting {retry_delay} seconds before retry")
                time.sleep(retry_delay)
            else:
                return None
        except Exception as e:
            print(f"Request exception (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"Waiting {retry_delay} seconds before retry")
                time.sleep(retry_delay)
            else:
                return None

    return None



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


def _parse_version_to_date(v: str) -> Optional[date]:
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
def latest_table_data_date():
    sql = '''
    SELECT max(time) AS latest_time
    FROM glue.planequery_lake_dev.adsb_messages
    WHERE time > TIMESTAMP '2026-05-03 00:00:00'
    '''
    from data_engineering.spark import get_spark_session
    spark = get_spark_session(app_name="check_adsblol_new_release")
    try:
        val: datetime = spark.sql(sql).collect()[0]["latest_time"]
    finally:
        spark.stop()

    if val.hour < 1:
        print("There is a new release with data up to", val)
        latest_date = date(val.year, val.month, val.day) - timedelta(days=1)
    else:
        latest_date = date(val.year, val.month, val.day)
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
