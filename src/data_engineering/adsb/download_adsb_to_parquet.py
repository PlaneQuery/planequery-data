"""
Downloads adsb.lol data and writes to Parquet files.

This file contains utility functions for downloading and processing adsb.lol trace data.
Used by the historical ADS-B processing pipeline.
"""
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import date, datetime, timedelta
import time
from pathlib import Path
from data_engineering.adsblol.check_for_new_release import (
    get_adsblol_releases_to_process,
    load_releases_txt,
    prepare_releases_for_run,
)
from data_engineering.adsb.trace_files_to_parquet import process_trace_files
from data_engineering.adsb.canonical import (
    ADSBSource,
    SUCCESS_MARKER,
    adsb_day_dir,
    adsb_parquet_root,
)


# ============================================================================
# Configuration
# ============================================================================
from data_engineering.utils import S3_BUCKET, OUTPUT_DIR
ADSBLOL_DIR = OUTPUT_DIR / "data" / "raw" /"adsblol"
RELEASES_DIR = ADSBLOL_DIR / "releases"
EXTRACT_DIR = ADSBLOL_DIR / "extract"
LOCKS_DIR = ADSBLOL_DIR / ".locks"
PARQUET_DIR = adsb_parquet_root(ADSBSource.ADSBLOL)
for _dir in (RELEASES_DIR, EXTRACT_DIR, LOCKS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DOWNLOAD_CONCURRENCY = 2
EXTRACTION_COMPLETE_MARKER = ".extraction-complete.json"

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
    return adsb_day_dir(ADSBSource.ADSBLOL, day)


def _parquet_partition_has_data(partition_dir: Path) -> bool:
    return (partition_dir / "_SUCCESS").is_file()


# ============================================================================
# Release URL Fetching from ALL_RELEASES.txt
# ============================================================================

@contextmanager
def _exclusive_file_lock(lock_path: Path):
    """Hold an advisory lock shared by all local pipeline processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


@contextmanager
def _release_download_slot():
    """Limit concurrent GitHub release downloads across local processes."""
    announced_wait = False
    while True:
        for slot in range(DOWNLOAD_CONCURRENCY):
            lock_path = LOCKS_DIR / f"download-slot-{slot}.lock"
            lock_file = lock_path.open("a", encoding="utf-8")
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock_file.close()
                continue

            try:
                yield slot + 1
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
            return

        if not announced_wait:
            print(
                f"Both GitHub download slots are busy; waiting "
                f"(limit={DOWNLOAD_CONCURRENCY})",
                flush=True,
            )
            announced_wait = True
        time.sleep(0.25)

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
    """Atomically download a release asset with mandatory size verification.
    
    Args:
        asset_url: URL to download from
        file_path: Local path to save to
        expected_size: Expected file size in bytes (for verification)
    
    Returns:
        True if download succeeded and size matches (if provided), False otherwise
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    max_retries = 2
    retry_delay = 30
    timeout_seconds = 140

    for attempt in range(1, max_retries + 1):
        print(f"Downloading {asset_url} (attempt {attempt}/{max_retries})")
        temporary_path: Path | None = None
        try:
            req = urllib.request.Request(asset_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                if response.status != 200:
                    raise OSError(
                        f"unexpected HTTP response: {response.status} {response.reason}"
                    )

                content_length = response.headers.get("Content-Length")
                response_size = int(content_length) if content_length is not None else None
                if expected_size is not None and response_size is not None:
                    if expected_size != response_size:
                        raise OSError(
                            f"server size changed: expected {expected_size} bytes, "
                            f"server reports {response_size}"
                        )
                verified_size = expected_size if expected_size is not None else response_size
                if verified_size is None:
                    raise OSError("server did not provide Content-Length; refusing unverified download")

                if file_path.exists():
                    actual_size = file_path.stat().st_size
                    if actual_size == verified_size:
                        print(
                            f"[SKIP] {file_path} already downloaded and verified "
                            f"({actual_size} bytes)."
                        )
                        return True
                    print(
                        f"[WARN] {file_path} exists but size mismatch "
                        f"(expected {verified_size}, got {actual_size}). Re-downloading."
                    )

                fd, temporary_name = tempfile.mkstemp(
                    dir=file_path.parent,
                    prefix=f".{file_path.name}.",
                    suffix=".part",
                )
                temporary_path = Path(temporary_name)
                bytes_written = 0
                with os.fdopen(fd, "wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
                        bytes_written += len(chunk)
                        if bytes_written > verified_size:
                            raise OSError(
                                f"download exceeded expected size of {verified_size} bytes"
                            )
                    file.flush()
                    os.fsync(file.fileno())

                if bytes_written != verified_size:
                    raise OSError(
                        f"incomplete download: expected {verified_size} bytes, "
                        f"received {bytes_written}"
                    )

                os.replace(temporary_path, file_path)
                temporary_path = None
                print(f"Saved {file_path} ({bytes_written} bytes, verified)")
                return True
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
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return False


def extract_split_archive(file_paths: list[str | Path], extract_dir: str | Path) -> bool:
    """Strictly extract split tar parts into an atomically published directory."""
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
    
    file_paths = [Path(path) for path in sorted(file_paths, key=sort_key)]
    extract_dir = Path(extract_dir)
    completion_marker = extract_dir / EXTRACTION_COMPLETE_MARKER
    if extract_dir.is_dir() and completion_marker.is_file():
        print(f"[SKIP] Verified extraction already exists: {extract_dir}")
        return True

    split_archives: dict[Path, list[Path]] = {}
    for path in file_paths:
        match = re.fullmatch(r"(.+\.tar)\.(?:[a-zA-Z]+|\d+)", path.name)
        if match:
            archive = path.with_name(match.group(1))
            split_archives.setdefault(archive, []).append(path)

    for archive, parts in split_archives.items():
        print(f"Using split archive parts for {archive.name}: {[part.name for part in parts]}")
        if archive in file_paths:
            print(f"Ignoring standalone tar because split parts are available: {archive}")
    file_paths = [path for path in file_paths if path not in split_archives]
    for path in file_paths:
        if path.suffix == ".tar":
            print(f"Using standalone tar because no split parts are available: {path}")

    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=extract_dir.parent,
            prefix=f".{extract_dir.name}.extract-",
        )
    )
    cat_proc: subprocess.Popen | None = None

    try:
        cat_proc = subprocess.Popen(
            ["cat", *map(str, file_paths)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        tar_cmd = ["tar", "xf", "-", "-C", str(staging_dir), "--strip-components=1"]
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
        if cat_proc.returncode != 0:
            raise RuntimeError(f"cat exited {cat_proc.returncode}: {cat_stderr.strip()}")
        if result.returncode != 0:
            raise RuntimeError(f"tar exited {result.returncode}: {tar_stderr.strip()}")
        if not any(path.is_file() for path in staging_dir.rglob("*")):
            raise RuntimeError("archive contained no files")

        marker_payload = {
            "parts": [
                {"name": path.name, "size": path.stat().st_size}
                for path in file_paths
            ]
        }
        (staging_dir / EXTRACTION_COMPLETE_MARKER).write_text(
            json.dumps(marker_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        if extract_dir.exists():
            print(f"Removing unverified extraction directory: {extract_dir}")
            shutil.rmtree(extract_dir)
        os.replace(staging_dir, extract_dir)

        print(f"Successfully extracted archive to {extract_dir}")

        disk = shutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        print(f"Disk space after extraction: {free_gb:.1f}GB free")

        return True
    except Exception as e:
        print(f"Failed to extract split archive: {e}")
        return False
    finally:
        if cat_proc is not None and cat_proc.poll() is None:
            cat_proc.terminate()
            cat_proc.wait()
        shutil.rmtree(staging_dir, ignore_errors=True)


OS_CPU_COUNT = os.cpu_count() or 1
MAX_WORKERS = min(OS_CPU_COUNT, 8)
SKIP_DAYS = {date(2026, 5, 5), date(2026, 5, 6)}


def _day_date(day: date | datetime) -> date:
    if isinstance(day, datetime):
        return day.date()
    return day


def download_and_extract(version_date: str) -> str | None:
    """Download and extract tar files, return extract directory path."""
    lock_path = LOCKS_DIR / f"{version_date}.lock"
    with _exclusive_file_lock(lock_path):
        return _download_and_extract_locked(version_date)


def _download_and_extract_locked(version_date: str) -> str | None:
    release_dir = _release_dir(version_date)
    extract_dir = _extract_dir(version_date)
    release_dir.mkdir(parents=True, exist_ok=True)

    completion_marker = extract_dir / EXTRACTION_COMPLETE_MARKER
    if extract_dir.is_dir() and completion_marker.is_file():
        print(f"[SKIP] Verified extraction already exists: {extract_dir}")
        return str(extract_dir)

    print(f"Fetching release URLs for {version_date}...")
    all_urls = fetch_release_urls(version_date)
    if not all_urls:
        raise FileNotFoundError(f"No adsblol production releases found for {version_date}")

    # Prefer non-tmp URLs; only use tmp if no normal URLs exist.
    normal_urls = [u for u in all_urls if "tmp" not in u]
    tmp_urls = [u for u in all_urls if "tmp" in u]
    use_urls = normal_urls if normal_urls else tmp_urls
    print(f"Using {'normal' if normal_urls else 'tmp'} releases ({len(use_urls)} URLs)")

    downloaded_files = []
    with _release_download_slot() as slot:
        print(
            f"Using GitHub download slot {slot}/{DOWNLOAD_CONCURRENCY} "
            f"for {version_date}",
            flush=True,
        )
        for url in use_urls:
            asset_name = url.rstrip('/').split('/')[-1]
            file_path = release_dir / asset_name
            if not download_asset(url, file_path):
                raise RuntimeError(
                    f"Failed to download adsblol archive for {version_date}: "
                    f"{url} -> {file_path}"
                )
            downloaded_files.append(str(file_path))

    if extract_split_archive(downloaded_files, str(extract_dir)):
        return str(extract_dir)
    raise RuntimeError(
        f"Failed to extract adsblol archives for {version_date} into {extract_dir}: "
        f"{downloaded_files}. See extraction output for the tar error."
    )


def process_version_date(
    version_date: str,
    keep_folders: bool = True,
) -> int:
    """Process a version date: download, extract, and write to parquet.
    
    Args:
        version_date: Format like 'v2026.02.01'
        keep_folders: Whether to keep downloaded release archives and extracted
            folders after processing
    
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
    )
    
    print(f"\nTotal rows written: {total_rows}")
    print(f"Partition dir: {partition_dir}")
    # Only remove temporary source artifacts when explicitly requested.
    if not keep_folders:
        for artifact_dir in (Path(extract_dir), _release_dir(version_date)):
            if artifact_dir.exists():
                print(f"Removing {artifact_dir}")
                shutil.rmtree(artifact_dir)
    
    return total_rows


def ingest_parquet_to_iceberg(parquet_path: Path, target_date: date):
    from data_engineering.spark import get_spark_session
    from pyspark.sql import functions as F

    spark = get_spark_session()
    table_name = "glue.planequery_lake_prod.adsb_messages"
    df = spark.read.parquet(str(parquet_path)).withColumn("source", F.lit("adsb.lol"))
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
):
    """Create parquet output for a single day.
    
    Args:
        day: datetime object or string in 'YYYY-MM-DD' format
        keep_folders: Whether to keep downloaded release archives and extracted
            folders after processing
    
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

    # Concurrent runs can request the same ADS-B partition. Hold this lock
    # through conversion and cleanup, not just download/extraction, so another
    # converter cannot remove a partition while it is being written or read.
    # Use a separate lock because download_and_extract takes the release lock.
    with _exclusive_file_lock(LOCKS_DIR / f"{version_date}.parquet.lock"):
        if _parquet_partition_has_data(partition_dir):
            return partition_dir

        print(f"Creating parquet for {version_date}")
        rows_processed = process_version_date(
            version_date,
            keep_folders,
        )

        if rows_processed > 0 and _parquet_partition_has_data(partition_dir):
            return partition_dir
        else:
            return None


def upload_to_s3(parquet_path: Path):
    import boto3
    s3 = boto3.client('s3')
    paths = sorted(parquet_path.rglob("*.parquet"))
    assert len(paths) == 8, f"Expected 8 parquet files under {parquet_path}, found {len(paths)}"
    success_path = parquet_path / SUCCESS_MARKER
    assert success_path.is_file(), f"Missing completion marker: {success_path}"
    for path in [*paths, success_path]:
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
                    "/app/src/data_engineering/flights/publish.py",
                    day.isoformat(),
                    next_day.isoformat(),
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

    prepare_releases_for_run({_day_date(day).year for day in days})

    processed_days = []
    for day in days:
        if _day_date(day) in SKIP_DAYS:
            print(f"\nSkipping {day.strftime('%Y-%m-%d')}")
            continue

        result = create_parquet_for_day(
            day,
            keep_folders=args.keep_folders,
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
