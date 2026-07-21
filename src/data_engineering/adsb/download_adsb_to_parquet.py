"""
Downloads adsb.lol data and writes to Parquet files.

This file contains utility functions for downloading and processing adsb.lol trace data.
Used by the historical ADS-B processing pipeline.
"""
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
import time
from pathlib import Path
from data_engineering.adsblol.check_for_new_release import get_adsblol_releases_to_process, load_releases_txt
from data_engineering.adsb.trace_files_to_parquet import process_trace_files


# ============================================================================
# Configuration
# ============================================================================
from data_engineering.utils import S3_BUCKET, OUTPUT_DIR
ADSBLOL_DIR = OUTPUT_DIR / "data" / "raw" /"adsblol"
RELEASES_DIR = ADSBLOL_DIR / "releases"
EXTRACT_DIR = ADSBLOL_DIR / "extract"
PARQUET_DIR = ADSBLOL_DIR / "parquet_output" / "v6"
for _dir in (RELEASES_DIR, EXTRACT_DIR, PARQUET_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get('GITHUB_TOKEN')  # Optional: kept for authenticated asset downloads
HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}


def _date_parts_from_version_date(version_date: str) -> tuple[str, str, str]:
    return version_date[1:5], version_date[6:8], version_date[9:11]


def _date_partition_dir(root: Path, year: str, month: str, day: str) -> Path:
    return root / f"year={year}" / f"month={month}" / f"day={day}"


def _release_dir(version_date: str) -> Path:
    return _date_partition_dir(RELEASES_DIR, *_date_parts_from_version_date(version_date))


def _extract_dir(version_date: str) -> Path:
    return _date_partition_dir(EXTRACT_DIR, *_date_parts_from_version_date(version_date))


def _parquet_partition_dir(day: date | datetime) -> Path:
    return PARQUET_DIR / f"year={day.year}" / f"month={day.month:02d}" / f"day={day.day:02d}"


def _parquet_partition_has_data(partition_dir: Path) -> bool:
    return any(partition_dir.glob("icao_bucket=*/data.parquet"))


# ============================================================================
# Release URL Fetching from ALL_RELEASES.txt
# ============================================================================

class DownloadTimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise DownloadTimeoutException("Download timed out after 40 seconds")

def _fetch_release_urls_from_txt(year: str, version_date: str) -> list[str]:
    """Fetch download URLs from ALL_RELEASES.txt for a given year's adsblol repo."""
    PATTERN = re.compile(rf"{re.escape(version_date)}-planes-readsb-prod-\d+(tmp)?/")

    content = load_releases_txt(year)
    if content is None:
        return []

    seen = set()
    urls = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        line_urls = [u.strip() for u in line.split(',')]
        if any(PATTERN.search(u) for u in line_urls):
            for u in line_urls:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
    return urls


def fetch_release_urls(version_date: str) -> list[str]:
    """Fetch download URLs for a given version date from ALL_RELEASES.txt.

    For Dec 31 dates, if no URLs are found in the current year's repo,
    also checks the next year's repo (adsblol sometimes publishes Dec 31
    data in the following year's repository).
    """
    year = version_date.split('.')[0][1:]
    urls = _fetch_release_urls_from_txt(year, version_date)

    if not urls and version_date.endswith(".12.31"):
        next_year = str(int(year) + 1)
        print(f"No releases found for {version_date} in {year} repo, checking {next_year} repo")
        urls = _fetch_release_urls_from_txt(next_year, version_date)

    return urls


def download_asset(asset_url: str, file_path: str | Path, expected_size: int | None = None) -> bool:
    """Download a single release asset with size verification.
    
    Args:
        asset_url: URL to download from
        file_path: Local path to save to
        expected_size: Expected file size in bytes (for verification)
    
    Returns:
        True if download succeeded and size matches (if provided), False otherwise
    """
    os.makedirs(os.path.dirname(file_path) or ADSBLOL_DIR, exist_ok=True)
    
    # Check if file exists and has correct size
    if os.path.exists(file_path):
        if expected_size is not None:
            actual_size = os.path.getsize(file_path)
            if actual_size == expected_size:
                print(f"[SKIP] {file_path} already downloaded and verified ({actual_size} bytes).")
                return True
            else:
                print(f"[WARN] {file_path} exists but size mismatch (expected {expected_size}, got {actual_size}). Re-downloading.")
                os.remove(file_path)
        else:
            print(f"[SKIP] {file_path} already downloaded.")
            return True
    
    max_retries = 2
    retry_delay = 30
    timeout_seconds = 140
    
    for attempt in range(1, max_retries + 1):
        print(f"Downloading {asset_url} (attempt {attempt}/{max_retries})")
        try:
            req = urllib.request.Request(asset_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                if response.status == 200:
                    with open(file_path, "wb") as file:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            file.write(chunk)
                    
                    # Verify file size if expected_size was provided
                    if expected_size is not None:
                        actual_size = os.path.getsize(file_path)
                        if actual_size != expected_size:
                            print(f"[ERROR] Size mismatch for {file_path}: expected {expected_size} bytes, got {actual_size} bytes")
                            os.remove(file_path)
                            if attempt < max_retries:
                                print(f"Waiting {retry_delay} seconds before retry")
                                time.sleep(retry_delay)
                                continue
                            return False
                        print(f"Saved {file_path} ({actual_size} bytes, verified)")
                    else:
                        print(f"Saved {file_path}")
                    return True
                else:
                    print(f"Failed to download {asset_url}: {response.status} {response.msg}")
                    if attempt < max_retries:
                        print(f"Waiting {retry_delay} seconds before retry")
                        time.sleep(retry_delay)
                    else:
                        return False
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"404 Not Found: {asset_url}")
                raise Exception(f"Asset not found (404): {asset_url}")
            else:
                print(f"HTTP error occurred (attempt {attempt}/{max_retries}): {e.code} {e.reason}")
                if attempt < max_retries:
                    print(f"Waiting {retry_delay} seconds before retry")
                    time.sleep(retry_delay)
                else:
                    return False
        except urllib.error.URLError as e:
            print(f"URL/Timeout error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"Waiting {retry_delay} seconds before retry")
                time.sleep(retry_delay)
            else:
                return False
        except Exception as e:
            print(f"An error occurred (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"Waiting {retry_delay} seconds before retry")
                time.sleep(retry_delay)
            else:
                return False
    
    return False


def extract_split_archive(file_paths: list, extract_dir: str) -> bool:
    """
    Extracts a split archive by concatenating the parts using 'cat'
    and then extracting with 'tar' in one pipeline.
    """
    if os.path.isdir(extract_dir):
        print(f"[SKIP] Extraction directory already exists: {extract_dir}")
        return True
    
    def sort_key(path: str):
        base = os.path.basename(path)
        parts = base.rsplit('.', maxsplit=1)
        if len(parts) == 2:
            suffix = parts[1]
            if suffix.isdigit():
                return (0, int(suffix))
            if re.fullmatch(r'[a-zA-Z]+', suffix):
                return (1, suffix)
        return (2, base)
    
    file_paths = sorted(file_paths, key=sort_key)
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        cat_proc = subprocess.Popen(
            ["cat"] + file_paths,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        tar_cmd = ["tar", "xf", "-", "-C", extract_dir, "--strip-components=1"]
        result = subprocess.run(
            tar_cmd,
            stdin=cat_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if cat_proc.stdout is not None:
            cat_proc.stdout.close()
        cat_stderr = cat_proc.stderr.read().decode() if cat_proc.stderr else ""
        cat_proc.wait()
        
        if cat_stderr:
            print(f"cat stderr: {cat_stderr}")
        
        tar_stderr = result.stderr.decode() if result.stderr else ""
        if result.returncode != 0:
            # GNU tar exits non-zero for format issues that BSD tar silently
            # tolerates (e.g. trailing junk after the last valid entry).
            # Check whether files were actually extracted before giving up.
            extracted_items = os.listdir(extract_dir)
            if extracted_items:
                print(f"[WARN] tar exited {result.returncode} but extracted "
                      f"{len(extracted_items)} items — treating as success")
                if tar_stderr:
                    print(f"tar stderr: {tar_stderr}")
            else:
                print(f"Failed to extract split archive (tar exit {result.returncode})")
                if tar_stderr:
                    print(f"tar stderr: {tar_stderr}")
                shutil.rmtree(extract_dir, ignore_errors=True)
                return False
        
        print(f"Successfully extracted archive to {extract_dir}")
        
        disk = shutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        print(f"Disk space after extraction: {free_gb:.1f}GB free")
        
        return True
    except Exception as e:
        print(f"Failed to extract split archive: {e}")
        shutil.rmtree(extract_dir, ignore_errors=True)
        return False


OS_CPU_COUNT = os.cpu_count() or 1
MAX_WORKERS = min(OS_CPU_COUNT, 8)
SKIP_DAYS = {date(2026, 5, 5)}


def _day_date(day: date | datetime) -> date:
    if isinstance(day, datetime):
        return day.date()
    return day


def download_and_extract(version_date: str) -> str | None:
    """Download and extract tar files, return extract directory path."""
    release_dir = _release_dir(version_date)
    extract_dir = _extract_dir(version_date)
    release_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    if os.path.isdir(extract_dir):
        print(f"[SKIP] Already extracted: {extract_dir}")
        return str(extract_dir)
    
    # Check for existing tar files
    pattern = str(release_dir / f"{version_date}-planes-readsb-prod-*")
    matches = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    
    if matches:
        print(f"Found existing tar files for {version_date}")
        normal_matches = [
            p for p in matches
            if "tmp" not in os.path.basename(p)
        ]
        downloaded_files = normal_matches if normal_matches else matches
    else:
        # Download via ALL_RELEASES.txt
        print(f"Fetching release URLs for {version_date}...")
        all_urls = fetch_release_urls(version_date)
        if not all_urls:
            print(f"No releases found for {version_date}")
            return None

        # Prefer non-tmp URLs; only use tmp if no normal URLs exist
        normal_urls = [u for u in all_urls if "tmp" not in u]
        tmp_urls = [u for u in all_urls if "tmp" in u]
        use_urls = normal_urls if normal_urls else tmp_urls
        print(f"Using {'normal' if normal_urls else 'tmp'} releases ({len(use_urls)} URLs)")

        downloaded_files = []
        for url in use_urls:
            asset_name = url.rstrip('/').split('/')[-1]
            file_path = release_dir / asset_name
            if download_asset(url, file_path):
                downloaded_files.append(str(file_path))
    
    if not downloaded_files:
        print(f"No files downloaded for {version_date}")
        return None
    
    # Extract
    if extract_split_archive(downloaded_files, str(extract_dir)):
        return str(extract_dir)
    return None


def process_version_date(
    version_date: str,
    keep_folders: bool = True,
    pia_or_american_ladd_only: bool = False,
) -> int:
    """Process a version date: download, extract, and write to parquet.
    
    Args:
        version_date: Format like 'v2026.02.01'
        keep_folders: Whether to keep extracted folders after processing
    
    Returns:
        Number of rows processed
    """
    print(f"\n{'='*80}")
    print(f"Processing {version_date}")
    print(f"{'='*80}")
    
    # Download and extract
    extract_dir = download_and_extract(version_date)
    if not extract_dir:
        print(f"Failed to download/extract data for {version_date}")
        return 0
    
    target_day = datetime.strptime(version_date, "v%Y.%m.%d").date()
    partition_dir = _parquet_partition_dir(target_day)

    print(f"Processing trace files and writing parquet under {partition_dir}")
    total_rows = process_trace_files(
        folder_path=extract_dir,
        output_root=PARQUET_DIR,
        max_workers=MAX_WORKERS,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
    )
    
    print(f"\nTotal rows written: {total_rows}")
    print(f"Partition dir: {partition_dir}")
    # Cleanup extracted directory
    if not keep_folders and os.path.exists(extract_dir):
        print(f"Removing {extract_dir}")
        shutil.rmtree(extract_dir)
    
    return total_rows


def ingest_parquet_to_iceberg(parquet_path: Path, target_date: date):
    from data_engineering.spark import get_spark_session
    spark = get_spark_session()
    table_name = "glue.planequery_lake_prod.adsb_messages"
    df = spark.read.parquet(str(parquet_path))
    df = df.select(*spark.table(table_name).columns)
    df = df.coalesce(2)
    next_day = target_date + timedelta(days=1)
    existing_row_count = spark.sql(
        f"""
        SELECT COUNT(*) AS row_count
        FROM {table_name}
        WHERE time >= TIMESTAMP '{target_date:%Y-%m-%d} 00:00:00'
          AND time < TIMESTAMP '{next_day:%Y-%m-%d} 00:00:00'
        """
    ).collect()[0]["row_count"]
    print(f"Rows already in table on {target_date}: {existing_row_count}")
    if existing_row_count > 10_000:
        raise Exception(f"Data for {target_date} already exists in Iceberg. Ingestion skipped.")
    
    df.writeTo(table_name).append()
    print(f"Ingested parquet into Iceberg: {parquet_path}")
    spark.stop()

def create_parquet_for_day(
    day,
    keep_folders: bool = True,
    pia_or_american_ladd_only: bool = False,
):
    """Create parquet output for a single day.
    
    Args:
        day: datetime object or string in 'YYYY-MM-DD' format
        keep_folders: Whether to keep extracted folders after processing
    
    Returns:
        Path to the created parquet partition directory, or None if failed
    """
    if isinstance(day, str):
        day = datetime.strptime(day, "%Y-%m-%d")
    if _day_date(day) in SKIP_DAYS:
        print(f"Skipping {day.strftime('%Y-%m-%d')}")
        return None
    
    version_date = f"v{day.strftime('%Y.%m.%d')}"

    partition_dir = _parquet_partition_dir(day)
    if _parquet_partition_has_data(partition_dir):
        print(f"Parquet partition already exists: {partition_dir}")
        return partition_dir

    print(f"Creating parquet for {version_date}")
    rows_processed = process_version_date(
        version_date,
        keep_folders,
        pia_or_american_ladd_only,
    )

    if rows_processed > 0 and _parquet_partition_has_data(partition_dir):
        return partition_dir
    else:
        return None


def upload_to_s3(parquet_path: Path):
    import boto3
    s3 = boto3.client('s3')
    if parquet_path.is_file():
        paths = [parquet_path]
    else:
        paths = sorted(path for path in parquet_path.rglob("*") if path.is_file())

    for path in paths:
        s3_key = str(path.relative_to(OUTPUT_DIR))
        s3.upload_file(str(path), S3_BUCKET, s3_key)
        print(f"Uploaded {path} to s3://{S3_BUCKET}/{s3_key}")


NO_ADSBLOL_RELEASES_EXIT_CODE = 42


def _format_day_list(days: list[date | datetime]) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in days]


def get_adsblol_releases_to_process_or_exit() -> list[date]:
    """Check release discovery once and exit specially when there is no work."""
    days = get_adsblol_releases_to_process()
    if days:
        print(f"No dates provided, processing releases: {_format_day_list(days)}")
        return days

    print(
        "No adsb.lol releases to process; exiting with code "
        f"{NO_ADSBLOL_RELEASES_EXIT_CODE}."
    )
    sys.exit(NO_ADSBLOL_RELEASES_EXIT_CODE)


def write_daily_pipeline_manifest(processed_days: list[date | datetime]) -> None:
    parameter_name = os.getenv("DAILY_PIPELINE_MANIFEST_SSM_PARAMETER")
    if not parameter_name:
        return

    import boto3

    jobs = []
    for value in processed_days:
        day = _day_date(value)
        next_day = day + timedelta(days=1)
        jobs.append(
            {
                "job_name": f"flights-{day:%Y-%m-%d}",
                "command": [
                    "python",
                    "/app/src/data_engineering/flights/flights_from_model.py",
                    day.isoformat(),
                    next_day.isoformat(),
                    "--download-airport-model-from-s3",
                    "--write-to-s3",
                    "--ingest-to-iceberg",
                    "--prod",
                ],
            }
        )

    boto3.client("ssm").put_parameter(
        Name=parameter_name,
        Value=json.dumps({"flight_inference_jobs": jobs}),
        Type="String",
        Overwrite=True,
    )
    print(f"Wrote daily pipeline manifest with {len(jobs)} flight jobs")



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download adsb.lol data and convert to Parquet'
    )
    parser.add_argument(
        'start_date',
        nargs='?',
        help='Start date (inclusive) in YYYY-MM-DD format'
    )
    parser.add_argument(
        'end_date',
        nargs='?',
        help='End date (exclusive) in YYYY-MM-DD format. If omitted, only start_date is processed.'
    )
    parser.add_argument(
        '--remove-folders',
        dest='keep_folders',
        action='store_false',
        help='Remove extracted folders after processing'
    )
    parser.add_argument(
        '--write-to-s3',
        action='store_true',
        help='Upload the created parquet files to AWS'
    )
    parser.add_argument(
        '--ingest-to-iceberg',
        action='store_true',
        help='Ingest the created parquet files into Iceberg'
    )
    parser.add_argument(
        '--pia-or-american-ladd-only',
        action='store_true',
        help='Only write PIA ICAOs and American LADD aircraft'
    )
    parser.set_defaults(keep_folders=True)
    
    args = parser.parse_args()

    if args.start_date is None and args.end_date is None:
        days = get_adsblol_releases_to_process_or_exit()
    else:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else start + dt.timedelta(days=1)

        days = []
        current = start
        while current < end:
            days.append(current)
            current += dt.timedelta(days=1)

    processed_days = []
    for day in days:
        if _day_date(day) in SKIP_DAYS:
            print(f"\nSkipping {day.strftime('%Y-%m-%d')}")
            continue

        result = create_parquet_for_day(
            day,
            keep_folders=args.keep_folders,
            pia_or_american_ladd_only=args.pia_or_american_ladd_only,
        )
        if result:
            print(f"\n✓ Success! Parquet output created: {result}")
            if args.ingest_to_iceberg:
                ingest_parquet_to_iceberg(result, day)
            if args.write_to_s3:
                upload_to_s3(result)
            processed_days.append(day)
        else:
            print(f"\n✗ Failed to create parquet file for {day.strftime('%Y-%m-%d')}")
            sys.exit(1)

    write_daily_pipeline_manifest(processed_days)
