"""
Downloads adsb.lol data and writes to Parquet files.

This file contains utility functions for downloading and processing adsb.lol trace data.
Used by the historical ADS-B processing pipeline.
"""
import datetime as dt
import glob
import gzip
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
import time
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from data_engineering.adsblol.check_for_new_release import get_adsblol_releases_to_process, load_releases_txt


# ============================================================================
# Configuration
# ============================================================================
from data_engineering.utils import S3_BUCKET, OUTPUT_DIR
ADSBLOL_DIR = OUTPUT_DIR / "data" / "raw" /"adsblol"
PARQUET_DIR = ADSBLOL_DIR / "parquet_output" / "v5"
os.makedirs(PARQUET_DIR, exist_ok=True)

TOKEN = os.environ.get('GITHUB_TOKEN')  # Optional: kept for authenticated asset downloads
HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}


def get_resource_usage() -> str:
    """Get current RAM and disk usage as a formatted string."""
    # RAM usage (RSS = Resident Set Size)
    ram_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS, ru_maxrss is in bytes; on Linux, it's in KB
    if sys.platform == 'darwin':
        ram_gb = ram_bytes / (1024**3)
    else:
        ram_gb = ram_bytes / (1024**2)  # Convert KB to GB
    
    # Disk usage
    disk = shutil.disk_usage('.')
    disk_free_gb = disk.free / (1024**3)
    disk_total_gb = disk.total / (1024**3)
    
    return f"RAM: {ram_gb:.2f}GB | Disk: {disk_free_gb:.1f}GB free / {disk_total_gb:.1f}GB total"


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


def download_asset(asset_url: str, file_path: str, expected_size: int | None = None) -> bool:
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
    Deletes the tar files immediately after extraction to save disk space.
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
        
        # Delete tar files immediately after extraction
        for tar_file in file_paths:
            try:
                os.remove(tar_file)
                print(f"Deleted tar file: {tar_file}")
            except Exception as e:
                print(f"Failed to delete {tar_file}: {e}")
        
        # Check disk usage after deletion
        disk = shutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        print(f"Disk space after tar deletion: {free_gb:.1f}GB free")
        
        return True
    except Exception as e:
        print(f"Failed to extract split archive: {e}")
        shutil.rmtree(extract_dir, ignore_errors=True)
        return False


# ============================================================================
# Trace File Processing (with alt_baro/on_ground handling)
# ============================================================================

DATA_SOURCE = "adsb.lol"


def _str_or_none(v):
    """Return v as a non-empty string, else None."""
    if v is None:
        return None
    s = v if isinstance(v, str) else str(v)
    s = s.strip()
    return s or None


def _to_int(v):
    """Coerce to int; return None if v is None or not convertible."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


DEDUPE_KEY_COLUMNS = ("time", "icao")
DEDUPE_SCORE_EXCLUDE_COLUMNS = set(DEDUPE_KEY_COLUMNS) | {"source"}
DEDUPE_PRIORITY_WEIGHTS = {
    # Prefer rows with real aircraft state over rows that only have metadata.
    "lat": 8,
    "lon": 8,
    "baro_altitude_ft": 8,
    "geom_altitude_ft": 8,
    "baro_rate_fpm": 10,
    "geom_rate_fpm": 10,
    "ground_speed_kt": 8,
    "indicated_airspeed_kt": 6,
    "true_airspeed_kt": 6,
    "mach": 6,
    "track_deg": 8,
    "track_rate_deg_s": 6,
    "roll_deg": 6,
    "magnetic_heading_deg": 6,
    "true_heading_deg": 6,
    "flags": 4,
    "type": 4,
}


def _has_information(v) -> bool:
    """Return True when a field contains useful information for dedupe scoring."""
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple, set, dict)):
        return bool(v)
    return True


def _row_information_score(cols: dict[str, list], row_idx: int) -> int:
    """Score a row by populated fields, prioritizing aircraft state over metadata."""
    return sum(
        DEDUPE_PRIORITY_WEIGHTS.get(col, 1)
        for col in COLUMNS
        if col not in DEDUPE_SCORE_EXCLUDE_COLUMNS
        and _has_information(cols[col][row_idx])
    )


def _merge_duplicate_rows(cols: dict[str, list], row_indices: list[int]) -> dict:
    """Build one row by combining non-empty values from all duplicate rows.

    The highest-scoring aircraft-state row wins when multiple duplicates have
    different non-empty values for the same column. Missing values in that row
    are backfilled from the other duplicates, so metadata-only rows and
    measurement-rich rows are collapsed into one combined output row.
    """
    ranked_indices = sorted(
        row_indices,
        key=lambda row_idx: _row_information_score(cols, row_idx),
        reverse=True,
    )
    merged = {}

    for col in COLUMNS:
        if col in DEDUPE_KEY_COLUMNS:
            merged[col] = cols[col][ranked_indices[0]]
            continue

        # Pick the first informative value in priority order. If every
        # duplicate is empty for this column, preserve the top-ranked row value.
        merged[col] = cols[col][ranked_indices[0]]
        for row_idx in ranked_indices:
            value = cols[col][row_idx]
            if _has_information(value):
                merged[col] = value
                break

    return merged


def _dedupe_cols_by_time_icao(cols: dict[str, list]) -> tuple[dict[str, list], int]:
    """Merge duplicate (time, icao) rows, keeping the most informative values."""
    row_count = len(cols["time"])
    if row_count < 2:
        return cols, 0

    rows_by_key: dict[tuple, list[int]] = {}
    for row_idx in range(row_count):
        key = tuple(cols[col][row_idx] for col in DEDUPE_KEY_COLUMNS)
        rows_by_key.setdefault(key, []).append(row_idx)

    duplicate_count = sum(len(row_indices) - 1 for row_indices in rows_by_key.values())

    if duplicate_count == 0:
        return cols, 0

    deduped_cols = {col: [] for col in COLUMNS}
    for row_indices in rows_by_key.values():
        if len(row_indices) == 1:
            for col in COLUMNS:
                deduped_cols[col].append(cols[col][row_indices[0]])
            continue

        merged = _merge_duplicate_rows(cols, row_indices)
        for col in COLUMNS:
            deduped_cols[col].append(merged[col])

    return deduped_cols, duplicate_count


def process_file(filepath: str) -> dict | None:
    """
    Process a single trace file and return a column-oriented dict (col -> list)
    matching the planequery_lake_dev.adsb_messages Iceberg schema.

    Trace row layout (post-2022):
        [t_offset, lat, lon, altitude, gs, track, flags, baro_rate,
         aircraft_obj, type, geom_alt, geom_rate, ias, roll]
    """
    with gzip.open(filepath, 'rb') as f:
        data = orjson.loads(f.read())

    icao = data.get('icao')
    timestamp = data.get('timestamp')
    trace_data = data.get('trace')
    if icao is None or timestamp is None or trace_data is None or not trace_data:
        return None

    # Constant per-file (top-level) fields. Empty strings -> None.
    registration = _str_or_none(data.get('r'))
    aircraft_type = _str_or_none(data.get('t'))
    owner = _str_or_none(data.get('ownOp'))
    aircraft_description = _str_or_none(data.get('desc'))
    no_reg_data_raw = data.get('noRegData')
    no_reg_data = bool(no_reg_data_raw) if no_reg_data_raw is not None else None

    # dbFlags bitfield: military=&1, interesting=&2, PIA=&4, LADD=&8
    db_flags_raw = data.get('dbFlags')
    if db_flags_raw is None:
        military = interesting = pia = ladd = None
    else:
        db_flags_int = int(db_flags_raw)
        military = bool(db_flags_int & 1)
        interesting = bool(db_flags_int & 2)
        pia = bool(db_flags_int & 4)
        ladd = bool(db_flags_int & 8)

    cols: dict[str, list] = {col: [] for col in COLUMNS}

    for row in trace_data:
        flags_val = row[6] or 0
        # flags & 8: altitude in row[3] is geometric, not barometric
        # flags & 4: vertical rate in row[7] is geometric, not barometric
        alt_is_geom = bool(flags_val & 8)
        vrate_is_geom = bool(flags_val & 4)

        # Altitude / on_ground handling
        altitude = row[3]
        if altitude == "ground":
            on_ground = True
            alt_val = None
        elif isinstance(altitude, (int, float)):
            on_ground = False
            alt_val = int(altitude)
        else:
            on_ground = False
            alt_val = None

        aircraft = row[8] if isinstance(row[8], dict) else {}

        # Per-position trace fields (post-2022); fall back to aircraft snapshot.
        geom_alt = row[10] if row[10] is not None else aircraft.get('alt_geom')
        geom_rate = row[11] if row[11] is not None else aircraft.get('geom_rate')
        ias = row[12] if row[12] is not None else aircraft.get('ias')
        roll = row[13] if row[13] is not None else aircraft.get('roll')
        track = row[5] if row[5] is not None else aircraft.get('track')
        gs = row[4] if row[4] is not None else aircraft.get('gs')

        # Route row[3] altitude to baro vs. geom column based on flags & 8.
        if alt_is_geom:
            baro_alt = aircraft.get('alt_baro')
            if isinstance(baro_alt, (int, float)):
                baro_alt = int(baro_alt)
            else:
                baro_alt = None
            if alt_val is not None and geom_alt is None:
                geom_alt = alt_val
        else:
            baro_alt = alt_val

        # Route row[7] vertical rate to baro vs. geom column based on flags & 4.
        vrate = row[7]
        if vrate_is_geom:
            baro_rate = aircraft.get('baro_rate')
            if geom_rate is None:
                geom_rate = vrate
        else:
            baro_rate = vrate if vrate is not None else aircraft.get('baro_rate')

        cols["time"].append(int((timestamp + row[0]) * 1000))
        cols["icao"].append(icao)
        cols["callsign"].append(_str_or_none(aircraft.get('flight')))
        cols["registration"].append(registration)
        cols["aircraft_type"].append(aircraft_type)

        cols["no_reg_data"].append(no_reg_data)
        cols["pia"].append(pia)
        cols["ladd"].append(ladd)
        cols["military"].append(military)
        cols["interesting"].append(interesting)
        cols["owner"].append(owner)
        cols["aircraft_description"].append(aircraft_description)
        cols["category"].append(_str_or_none(aircraft.get('category')))

        cols["source"].append(DATA_SOURCE)
        cols["flags"].append(row[6])

        cols["type"].append(_str_or_none(row[9]))

        cols["lat"].append(row[1])
        cols["lon"].append(row[2])
        cols["radius_of_containment_m"].append(_to_int(aircraft.get('rc')))

        cols["baro_altitude_ft"].append(baro_alt)
        cols["geom_altitude_ft"].append(_to_int(geom_alt))
        cols["on_ground"].append(on_ground)

        cols["ground_speed_kt"].append(gs)
        cols["indicated_airspeed_kt"].append(ias)
        cols["true_airspeed_kt"].append(aircraft.get('tas'))
        cols["mach"].append(aircraft.get('mach'))

        cols["track_deg"].append(track)
        cols["track_rate_deg_s"].append(aircraft.get('track_rate'))
        cols["roll_deg"].append(roll)
        cols["magnetic_heading_deg"].append(aircraft.get('mag_heading'))
        cols["true_heading_deg"].append(aircraft.get('true_heading'))

        cols["baro_rate_fpm"].append(_to_int(baro_rate))
        cols["geom_rate_fpm"].append(_to_int(geom_rate))

        cols["nav_qnh_hpa"].append(aircraft.get('nav_qnh'))
        cols["nav_altitude_mcp_ft"].append(_to_int(aircraft.get('nav_altitude_mcp')))
        cols["nav_altitude_fms_ft"].append(_to_int(aircraft.get('nav_altitude_fms')))
        cols["nav_heading_deg"].append(aircraft.get('nav_heading'))
        cols["nav_modes"].append(aircraft.get('nav_modes') or [])

        cols["version"].append(aircraft.get('version'))
        cols["nic"].append(aircraft.get('nic'))
        cols["nic_baro"].append(aircraft.get('nic_baro'))
        cols["nac_p"].append(aircraft.get('nac_p'))
        cols["nac_v"].append(aircraft.get('nac_v'))
        cols["sil"].append(aircraft.get('sil'))
        cols["sil_type"].append(_str_or_none(aircraft.get('sil_type')))
        cols["gva"].append(aircraft.get('gva'))
        cols["sda"].append(aircraft.get('sda'))

        cols["squawk"].append(_str_or_none(aircraft.get('squawk')))
        cols["emergency"].append(_str_or_none(aircraft.get('emergency')))
        alert = aircraft.get('alert')
        spi = aircraft.get('spi')
        cols["alert"].append(bool(alert) if alert is not None else None)
        cols["spi"].append(bool(spi) if spi is not None else None)

        cols["rssi_dbfs"].append(aircraft.get('rssi'))

        cols["wind_direction_deg"].append(aircraft.get('wd'))
        cols["wind_speed_kt"].append(aircraft.get('ws'))
        cols["outside_air_temp_c"].append(aircraft.get('oat'))
        cols["total_air_temp_c"].append(aircraft.get('tat'))

        cols["mlat"].append(aircraft.get('mlat') or [])
        cols["tisb"].append(aircraft.get('tisb') or [])

    if not cols["time"]:
        return None

    cols, duplicate_count = _dedupe_cols_by_time_icao(cols)
    if duplicate_count:
        print(f"Deduped {duplicate_count} duplicate rows in {filepath}")

    return cols


# ============================================================================
# Parquet Writing
# ============================================================================

# Column names in canonical order, matching the Iceberg `adsb_messages` table.
from data_engineering.adsb.parquet_schema import COLUMNS, PARQUET_SCHEMA

OS_CPU_COUNT = os.cpu_count() or 1
MAX_WORKERS = min(OS_CPU_COUNT, 4)


def collect_trace_files_with_find(root_dir):
    """Find all trace_full_*.json files in the extracted directory."""
    trace_dict: dict[str, str] = {}
    cmd = ['find', root_dir, '-type', 'f', '-name', 'trace_full_*.json']
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"Error executing find: {result.stderr}")
        return trace_dict
    
    for file_path in result.stdout.strip().split('\n'):
        if file_path:
            filename = os.path.basename(file_path)
            if filename.startswith("trace_full_") and filename.endswith(".json"):
                icao = filename[len("trace_full_"):-len(".json")]
                trace_dict[icao] = file_path
                    
    return trace_dict


def download_and_extract(version_date: str) -> str | None:
    """Download and extract tar files, return extract directory path."""
    extract_dir = os.path.join(ADSBLOL_DIR, f"{version_date}-planes-readsb-prod-0.tar_0")
    
    # Check if already extracted
    if os.path.isdir(extract_dir):
        print(f"[SKIP] Already extracted: {extract_dir}")
        return extract_dir
    
    # Check for existing tar files
    pattern = os.path.join(ADSBLOL_DIR, f"{version_date}-planes-readsb-prod-0*")
    matches = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    
    if matches:
        print(f"Found existing tar files for {version_date}")
        normal_matches = [
            p for p in matches
            if "-planes-readsb-prod-0." in os.path.basename(p)
            and "tmp" not in os.path.basename(p)
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
            file_path = os.path.join(ADSBLOL_DIR, asset_name)
            if download_asset(url, file_path):
                downloaded_files.append(file_path)
    
    if not downloaded_files:
        print(f"No files downloaded for {version_date}")
        return None
    
    # Extract
    if extract_split_archive(downloaded_files, extract_dir):
        return extract_dir
    return None


def process_version_date(version_date: str, keep_folders: bool = False) -> int:
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
    print(get_resource_usage())
    
    # Download and extract
    extract_dir = download_and_extract(version_date)
    if not extract_dir:
        print(f"Failed to download/extract data for {version_date}")
        return 0
    
    # Collect trace files
    all_trace_files = collect_trace_files_with_find(extract_dir)
    print(f"Found {len(all_trace_files)} trace files")
    
    if not all_trace_files:
        print("No trace files found")
        return 0
    
    # Process trace files and write to parquet
    # version_date format: 'v2025.10.01'
    _year, _month, _day = version_date[1:5], version_date[6:8], version_date[9:11]
    partition_dir = os.path.join(PARQUET_DIR, f"year={_year}", f"month={_month}", f"day={_day}")
    os.makedirs(partition_dir, exist_ok=True)
    parquet_path = os.path.join(partition_dir, "part-00000.parquet")

    print(f"Processing trace files and writing to {parquet_path}")
    total_rows = 0
    batch_size = 100_000  # rows before flushing to parquet
    # Limit pending futures to cap RAM from buffered results (completed but not yet consumed)
    SUBMIT_CHUNK = MAX_WORKERS * 32

    from concurrent.futures import ProcessPoolExecutor, as_completed

    def _flush(batch_cols: dict, n: int, writer):
        arrays = [pa.array(batch_cols[col], type=PARQUET_SCHEMA.field(col).type) for col in COLUMNS]
        table = pa.table(dict(zip(COLUMNS, arrays)), schema=PARQUET_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, PARQUET_SCHEMA, compression='zstd')
        writer.write_table(table)
        print(f"Wrote batch: {n} rows (total: {total_rows})")
        return writer

    batch_cols = {col: [] for col in COLUMNS}
    batch_count = 0
    writer = None
    file_items = list(all_trace_files.items())

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for chunk_start in range(0, len(file_items), SUBMIT_CHUNK):
            chunk = file_items[chunk_start:chunk_start + SUBMIT_CHUNK]
            futures = {executor.submit(process_file, path): icao for icao, path in chunk}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        n = len(result["time"])
                        for col in COLUMNS:
                            batch_cols[col].extend(result[col])
                        total_rows += n
                        batch_count += n

                        if batch_count >= batch_size:
                            writer = _flush(batch_cols, batch_count, writer)
                            batch_cols = {col: [] for col in COLUMNS}
                            batch_count = 0
                except Exception as e:
                    icao = futures[future]
                    print(f"Error processing {icao}: {e}")

    if batch_count:
        writer = _flush(batch_cols, batch_count, writer)
        print(f"Wrote final batch: {batch_count} rows")

    if writer:
        writer.close()
    
    print(f"\nTotal rows written: {total_rows}")
    print(f"Parquet file: {parquet_path}")
    print(f"Partition dir: {partition_dir}")    
    # Cleanup extracted directory
    if not keep_folders and os.path.exists(extract_dir):
        print(f"Removing {extract_dir}")
        shutil.rmtree(extract_dir)
    
    print(get_resource_usage())
    return total_rows


def ingest_parquet_to_iceberg(parquet_path: Path, target_date: date):
    from data_engineering.spark import get_spark_session
    spark = get_spark_session()
    df = spark.read.parquet(str(parquet_path))
    df = df.coalesce(2)
    next_day = target_date + timedelta(days=1)
    existing_row_count = spark.sql(
        f"""
        SELECT COUNT(*) AS row_count
        FROM glue.planequery_lake_prod.adsb_messages
        WHERE time >= TIMESTAMP '{target_date:%Y-%m-%d} 00:00:00'
          AND time < TIMESTAMP '{next_day:%Y-%m-%d} 00:00:00'
        """
    ).collect()[0]["row_count"]
    print(f"Rows already in table on {target_date}: {existing_row_count}")
    if existing_row_count > 10_000:
        raise Exception(f"Data for {target_date} already exists in Iceberg. Ingestion skipped.")
    
    df.writeTo("glue.planequery_lake_prod.adsb_messages").append()
    print(f"Ingested full parquet file into Iceberg: {parquet_path}")
    spark.stop()

def create_parquet_for_day(day, keep_folders: bool = False):
    """Create parquet file for a single day.
    
    Args:
        day: datetime object or string in 'YYYY-MM-DD' format
        keep_folders: Whether to keep extracted folders after processing
    
    Returns:
        Path to the created parquet file, or None if failed
    """
    from pathlib import Path
    
    if isinstance(day, str):
        day = datetime.strptime(day, "%Y-%m-%d")
    
    version_date = f"v{day.strftime('%Y.%m.%d')}"

    # Check if parquet already exists
    partition_dir = Path(PARQUET_DIR) / f"year={day.year}" / f"month={day.month:02d}" / f"day={day.day:02d}"
    parquet_path = partition_dir / "part-00000.parquet"
    if parquet_path.exists():
        print(f"Parquet file already exists: {parquet_path}")
        return parquet_path

    print(f"Creating parquet for {version_date}")
    rows_processed = process_version_date(version_date, keep_folders)

    if rows_processed > 0 and parquet_path.exists():
        return parquet_path
    else:
        return None


def upload_to_s3(parquet_path: Path):
    import boto3
    s3 = boto3.client('s3')
    s3_key = str(parquet_path.relative_to(OUTPUT_DIR))
    s3.upload_file(str(parquet_path), S3_BUCKET, s3_key)


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


def _inference_feature_days(processed_days: list[date | datetime]) -> list[date]:
    days = sorted({d.date() if isinstance(d, datetime) else d for d in processed_days})
    latest_day = days[-1]
    feature_days = {days[0] - timedelta(days=1)}
    feature_days.update(day for day in days if day < latest_day)
    return sorted(feature_days)


def write_daily_pipeline_manifest(processed_days: list[date | datetime]) -> None: # TODO: Much of this should be moved to CDK stack or elsewhere.
    parameter_name = os.environ.get("DAILY_PIPELINE_MANIFEST_SSM_PARAMETER")
    if not parameter_name:
        return

    feature_days = _inference_feature_days(processed_days)
    manifest = {
        "processed_adsb_days": _format_day_list(processed_days),
        "inference_feature_days": _format_day_list(feature_days),
        "inference_features_execution_input": {
            "dates": [
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "command": [
                        "--start-date",
                        day.strftime("%Y-%m-%d"),
                        "--end-date",
                        (day + timedelta(days=1)).strftime("%Y-%m-%d"),
                    ],
                }
                for day in feature_days
            ],
            "max_concurrency": int(os.environ.get("INFERENCE_FEATURES_MAX_CONCURRENCY", "1")),
        },
        "flight_inference_jobs": [
            {
                "date": day.strftime("%Y-%m-%d"),
                "job_name": f"daily-flights-{day:%Y-%m-%d}",
                "command": [
                    "python",
                    "/app/src/data_engineering/flights/flights_from_model.py",
                    day.strftime("%Y-%m-%d"),
                    (day + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "--write-to-s3",
                    "--ingest-to-iceberg",
                    "--prod",
                ],
            }
            for day in feature_days
        ],
    }

    import boto3
    ssm = boto3.client("ssm")
    ssm.put_parameter(
        Name=parameter_name,
        Value=orjson.dumps(manifest).decode("utf-8"),
        Type="String",
        Overwrite=True,
    )
    print(f"Wrote daily pipeline manifest to SSM parameter {parameter_name}")


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
        '--keep-folders',
        action='store_true',
        help='Keep extracted folders after processing'
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
        result = create_parquet_for_day(day, args.keep_folders)
        if result:
            print(f"\n✓ Success! Parquet file created: {result}")
            if args.ingest_to_iceberg:
                ingest_parquet_to_iceberg(result, day)
            if args.write_to_s3:
                upload_to_s3(result)
            processed_days.append(day)
        else:
            print(f"\n✗ Failed to create parquet file for {day.strftime('%Y-%m-%d')}")
            sys.exit(1)

    if processed_days:
        write_daily_pipeline_manifest(processed_days)
