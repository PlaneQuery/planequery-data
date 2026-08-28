from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import sys
import os
from pathlib import Path
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import requests
import argparse
import threading

import time
import logging
import polars as pl
environment = os.getenv('ENVIRONMENT', 'development')

from data_engineering.utils import OUTPUT_DIR
ADSB_REAPI_S3_BUCKET = os.getenv("ADSB_REAPI_S3_BUCKET", "planequery-dev-adsb-reapi")
ADSB_REAPI_S3_PREFIX = os.getenv(
    "ADSB_REAPI_S3_PREFIX",
    "feeder-1/data/raw/reapi",
).strip("/")
ADSB_REAPI_DATA_ROOT = Path(
    os.getenv("ADSB_REAPI_DATA_ROOT", str(OUTPUT_DIR / "data" / "raw" / "reapi"))
)
ADSB_REAPI_CHECKPOINT_PATH = ADSB_REAPI_DATA_ROOT / "_state" / "last_message_time_by_icao.json"
ADSB_REAPI_PENDING_UPLOADS_PATH = ADSB_REAPI_DATA_ROOT / "_state" / "pending_uploads.json"
ADSB_PROXY_CACHE_PATH = Path(
    os.getenv(
        "ADSB_PROXY_CACHE_PATH",
        str(ADSB_REAPI_DATA_ROOT / "_state" / "proxy_messages.json"),
    )
)
ADSB_PROXY_CACHE_WINDOW_SECONDS = int(
    os.getenv("ADSB_PROXY_CACHE_WINDOW_SECONDS", str(5 * 60))
)
ADSB_PROXY_TOKEN = os.getenv("ADSB_PROXY_TOKEN")
ADSB_REAPI_UPLOAD_THRESHOLD_BYTES = int(
    os.getenv("ADSB_REAPI_UPLOAD_THRESHOLD_BYTES", str(5 * 1024 * 1024))
)

class FetchError:
    def __init__(self, url, error):
        self.url = url
        self.error = error

def get_current_utc_time_in_milliseconds() -> int:
    return int(time.time() * 1000)
import math
def is_epoch_utc_milliseconds(x, min_year=2000, max_year=2200):
    """
    Return True if x is a Unix epoch timestamp in *milliseconds* (UTC),
    plausibly between Jan 1, min_year and Jan 1, max_year.
    Accepts int or float. Rejects seconds/us/ns by range.
    """
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return False

    # Convert assuming milliseconds
    try:
        dt = datetime.fromtimestamp(x / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return False

    # Year gate ensures we don't accidentally accept seconds (too small)
    # or micro/nanoseconds (too large)
    return (min_year <= dt.year < max_year)

START_TIME_OF_PROGRAM = get_current_utc_time_in_milliseconds()
START_TIME_OF_PROGRAM_MINUS_DAY= START_TIME_OF_PROGRAM - 24*3600*1000

last_message_time_by_icao: dict[str, int] = dict()

from data_engineering.adsb.adsb_messages_types import Adsb_Message_reapi

@dataclass
class ReapiFetchResult:
    rows: list[dict]
    api_call_time: datetime
    last_message_time_updates: dict[str, int]

@dataclass
class ParquetWriteResult:
    local_path: Path
    uploaded_s3_key: str | None
    pending_upload_bytes: int

ADSB_MESSAGE_POLARS_SCHEMA = {
    "api_call_time": pl.Datetime("us"),
    "time": pl.Datetime("us"),
    "data_source": pl.String,
    "raw_message_json": pl.String,
}


class AdsbMessageCache:
    """Thread-safe, durable cache of messages from the last successful fetches.

    The cache is only pruned after a successful upstream response.  Consequently,
    an upstream outage returns the last complete cache, even after it becomes older
    than the nominal five-minute window.
    """

    def __init__(self, path: Path, window_seconds: int = 5 * 60):
        self.path = path
        self.window_ms = window_seconds * 1000
        self._lock = threading.Lock()
        self._messages: list[dict] = []
        self._last_success_at: str | None = None
        self._last_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            self._messages = [message for message in messages if isinstance(message, dict)]
            last_success_at = data.get("last_success_at")
            if isinstance(last_success_at, str):
                self._last_success_at = last_success_at
            logging.info("Loaded %d cached ADS-B messages from %s", len(self._messages), self.path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logging.exception("Could not load ADS-B proxy cache from %s", self.path)

    def record_success(
        self,
        rows: list[dict],
        api_call_time: datetime,
    ) -> None:
        api_call_time = api_call_time.astimezone(timezone.utc)
        cutoff_ms = int(api_call_time.timestamp() * 1000) - self.window_ms
        new_messages = []
        for row in rows:
            message_time = row["time"]
            if message_time.tzinfo is None:
                message_time = message_time.replace(tzinfo=timezone.utc)
            else:
                message_time = message_time.astimezone(timezone.utc)
            new_messages.append(
                {
                    "time": message_time.isoformat(),
                    "time_ms": int(message_time.timestamp() * 1000),
                    "api_call_time": api_call_time.isoformat(),
                    "data_source": row["data_source"],
                    "message": json.loads(row["raw_message_json"]),
                }
            )

        with self._lock:
            self._messages.extend(new_messages)
            deduplicated = {}
            for message in self._messages:
                if _cached_message_time_ms(message) >= cutoff_ms:
                    deduplicated[_cached_message_key(message)] = message
            self._messages = list(deduplicated.values())
            self._last_success_at = api_call_time.isoformat()
            self._last_error = None
            self._save_locked()

    def record_failure(self, error: Exception) -> None:
        with self._lock:
            self._last_error = str(error)

    def snapshot(self) -> dict:
        with self._lock:
            messages = list(self._messages)
            return {
                "messages": messages,
                "count": len(messages),
                "window_seconds": self.window_ms // 1000,
                "stale": self._last_error is not None,
                "last_success_at": self._last_success_at,
                "upstream_error": self._last_error,
            }

    def _save_locked(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "last_success_at": self._last_success_at,
                        "messages": self._messages,
                    },
                    f,
                    separators=(",", ":"),
                )
            tmp_path.replace(self.path)
        except OSError:
            # Keep serving the in-memory cache if the SD card is temporarily
            # unavailable or full.
            logging.exception("Could not persist ADS-B proxy cache to %s", self.path)


def _cached_message_time_ms(message: dict) -> int:
    try:
        return int(message["time_ms"])
    except (KeyError, TypeError, ValueError):
        return 0


def _cached_message_key(message: dict) -> tuple:
    raw_message = message.get("message")
    icao = raw_message.get("hex") if isinstance(raw_message, dict) else None
    return (message.get("data_source"), icao, _cached_message_time_ms(message))


def make_proxy_handler(message_cache: AdsbMessageCache, token: str | None = None):
    class AdsbProxyHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.partition("?")[0]
            if path == "/health":
                snapshot = message_cache.snapshot()
                snapshot.pop("messages")
                status = (
                    HTTPStatus.OK
                    if snapshot["last_success_at"] is not None
                    else HTTPStatus.SERVICE_UNAVAILABLE
                )
                self._send_json(status, snapshot)
                return
            if path not in ("/", "/adsb", "/api/adsb"):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            self._send_json(HTTPStatus.OK, message_cache.snapshot())

        def _send_json(self, status: HTTPStatus, payload: dict) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            logging.info("ADS-B proxy: " + format, *args)

    return AdsbProxyHandler


def start_proxy_server(
    message_cache: AdsbMessageCache,
    host: str,
    port: int,
    token: str | None = None,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_proxy_handler(message_cache, token))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="adsb-proxy", daemon=True)
    thread.start()
    logging.info("ADS-B proxy listening on http://%s:%d", host, port)
    return server


def process_adsb_message_stream(
    primary_source: str = "adsb.lol",
    write_to_s3: bool = False,
    message_cache: AdsbMessageCache | None = None,
) -> str:
    start = time.perf_counter()
    now = time.time()
    fallback_until = getattr(process_adsb_message_stream, "_fallback_until", 0.0)
    fallback_source = "adsb.fi" if primary_source == "adsb.lol" else "adsb.lol"
    
    try:
        if now >= fallback_until:
            try:
                print(f"Using primary source: {primary_source}")
                fetch_result = process_reapi_states(primary_source)
                process_adsb_message_stream._fallback_until = 0.0
                data_source = primary_source
            except Exception as exc:
                logging.warning(f"{primary_source} failed, using {fallback_source} for 10 minutes: %s", exc)
                process_adsb_message_stream._fallback_until = now + 10 * 60
                fetch_result = process_reapi_states(fallback_source)
                data_source = fallback_source
        else:
            print(f"Primary source fallback active, using {fallback_source}")
            fetch_result = process_reapi_states(fallback_source)
            data_source = fallback_source
    except Exception as exc:
        if message_cache is not None:
            message_cache.record_failure(exc)
        raise

    if message_cache is not None:
        message_cache.record_success(fetch_result.rows, fetch_result.api_call_time)

    # process_takeoff_landing_events(adsb_messages)
    print(len(fetch_result.rows), "new messages to write")
    parquet_result = write_adsb_messages_to_parquet_and_s3(
        fetch_result.rows,
        fetch_result.api_call_time,
        write_to_s3=write_to_s3,
    )
    last_message_time_by_icao.update(fetch_result.last_message_time_updates)
    save_last_message_time_for_icaos()
    if parquet_result is not None:
        print(f"Wrote {parquet_result.local_path}")
        if parquet_result.uploaded_s3_key:
            print(f"Uploaded to s3://{ADSB_REAPI_S3_BUCKET}/{parquet_result.uploaded_s3_key}")
        elif write_to_s3:
            print(
                "S3 upload pending; "
                f"{parquet_result.pending_upload_bytes} bytes queued "
                f"of {ADSB_REAPI_UPLOAD_THRESHOLD_BYTES}"
            )
        else:
            print("S3 upload disabled")
    else:
        print("No messages to write")
    end = time.perf_counter()
    print(f"Elapsed process_adsb_message_stream: {_format_duration(end - start)}")
    return data_source

def process_reapi_states(source: str) -> ReapiFetchResult:
    """Generic function to process ADS-B data from either adsb.lol or adsb.fi"""
    data, api_call_time = get_reapi_request(source)
    adsb_message_rows: list[dict] = []
    last_message_time_updates: dict[str, int] = {}
    
    # adsb.lol uses "aircraft", adsb.fi uses "ac"
    messages_key = "aircraft" if source == "adsb.lol" else "ac"
    messages = data[messages_key]
    
    # Handle different timestamp formats: adsb.lol uses seconds, adsb.fi uses milliseconds
    now_raw = data["now"]
    if is_epoch_utc_milliseconds(now_raw):
        now = int(now_raw)  # Already in milliseconds
    else:
        now = int(now_raw * 1000)  # Convert from seconds to milliseconds
    
    print(len(messages), f"messages received from {source}")
    for message in messages:
        adsb_message_reapi = Adsb_Message_reapi.from_json(message)
        if adsb_message_reapi.seen_pos is None:
            continue
        seen_pos = adsb_message_reapi.seen_pos * 1000
        message_time = now - seen_pos
        icao: str = message["hex"]
        last_message_time = last_message_time_updates.get(icao, get_last_message_time(icao))
        if message_time <= last_message_time:
            continue
                
        time = datetime.fromtimestamp(message_time / 1000, tz=timezone.utc)
        adsb_message_row = reapi_to_parquet_row(
            raw_message=message,
            time=time,
            api_call_time=api_call_time,
            data_source=source,
        )
        adsb_message_rows.append(adsb_message_row)
        last_message_time_updates[icao] = int(message_time)
        
    return ReapiFetchResult(
        rows=adsb_message_rows,
        api_call_time=api_call_time,
        last_message_time_updates=last_message_time_updates,
    )

def get_reapi_request(source: str) -> tuple[dict, datetime]:
    """Generic function to fetch ADS-B data from either adsb.lol or adsb.fi"""
    if source == "adsb.lol":
        url = "https://re-api.adsb.lol?all"
    elif source == "adsb.fi":
        url = "https://opendata.adsb.fi/api/v2/snapshot"
    else:
        raise ValueError(f"Unknown source: {source}")
    
    attempts = 2
    delay = 1.0
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            api_call_time = datetime.now(timezone.utc)
            resp = requests.get(url, timeout=60)
            if resp.status_code == 429:
                logging.error("Rate limited by %s API on attempt %d/%d", source, attempt, attempts)
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    retry_after = int(retry_after)
                    delay = float(retry_after+5)
                else:
                    delay = 60.0
                continue

            resp.raise_for_status()
            return resp.json(), api_call_time
        except Exception as e:
            last_err = e
            logging.warning("get_reapi_request for %s attempt %d/%d failed: %s", source, attempt, attempts, e)
            if attempt < attempts:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts") from last_err


def reapi_to_parquet_row(
    *,
    raw_message: dict,
    time: datetime,
    api_call_time: datetime,
    data_source: str,
) -> dict:
    return {
        "api_call_time": _to_naive_utc(api_call_time),
        "time": _to_naive_utc(time),
        "data_source": data_source,
        "raw_message_json": _json_dumps(raw_message),
    }


def _json_dumps(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _to_naive_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None)

def get_last_message_time(icao: str) -> int:
    if icao in last_message_time_by_icao:
        return last_message_time_by_icao[icao]
    last_message_time_by_icao[icao] = 0
    return last_message_time_by_icao[icao]

def get_last_message_time_for_icaos() -> None:
    """Load the last-message checkpoint from the local machine if present."""
    if not ADSB_REAPI_CHECKPOINT_PATH.exists():
        print(f"No ADS-B checkpoint found at {ADSB_REAPI_CHECKPOINT_PATH}; starting fresh")
        return

    with ADSB_REAPI_CHECKPOINT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {ADSB_REAPI_CHECKPOINT_PATH}")

    last_message_time_by_icao.clear()
    for icao, timestamp_ms in data.items():
        try:
            last_message_time_by_icao[str(icao)] = int(timestamp_ms)
        except (TypeError, ValueError):
            logging.warning("Skipping invalid checkpoint value for ICAO %s: %r", icao, timestamp_ms)
    print(f"Loaded {len(last_message_time_by_icao)} ICAO checkpoint entries")


def save_last_message_time_for_icaos() -> None:
    ADSB_REAPI_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ADSB_REAPI_CHECKPOINT_PATH.with_suffix(f"{ADSB_REAPI_CHECKPOINT_PATH.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(last_message_time_by_icao, f, separators=(",", ":"), sort_keys=True)
    tmp_path.replace(ADSB_REAPI_CHECKPOINT_PATH)


def write_adsb_messages_to_parquet_and_s3(
    adsb_messages: list[dict],
    api_call_time: datetime,
    *,
    write_to_s3: bool = False,
) -> ParquetWriteResult | None:
    if not adsb_messages:
        return None

    api_call_time = api_call_time.astimezone(timezone.utc)
    partition_path = _hive_partition_path(api_call_time)
    local_dir = ADSB_REAPI_DATA_ROOT / partition_path
    local_dir.mkdir(parents=True, exist_ok=True)
    filename = f"part-{api_call_time.strftime('%Y%m%dT%H%M%S%fZ')}.parquet"
    local_path = local_dir / filename

    df = pl.from_dicts(adsb_messages, schema=ADSB_MESSAGE_POLARS_SCHEMA, strict=False)
    df.write_parquet(local_path, compression="zstd")

    uploaded_s3_key = None
    pending_upload_bytes = 0
    if write_to_s3:
        pending_uploads = _load_pending_uploads()
        pending_uploads.append(
            {
                "path": str(local_path),
                "api_call_time": api_call_time.isoformat(),
            }
        )
        pending_upload_bytes, pending_uploads = _pending_upload_bytes(pending_uploads)
        _save_pending_uploads(pending_uploads)
        if pending_upload_bytes >= ADSB_REAPI_UPLOAD_THRESHOLD_BYTES:
            uploaded_s3_key, pending_upload_bytes = _combine_and_upload_pending_parquets(
                pending_uploads
            )

    return ParquetWriteResult(
        local_path=local_path,
        uploaded_s3_key=uploaded_s3_key,
        pending_upload_bytes=pending_upload_bytes,
    )


def _load_pending_uploads() -> list[dict[str, str]]:
    if not ADSB_REAPI_PENDING_UPLOADS_PATH.exists():
        return []
    with ADSB_REAPI_PENDING_UPLOADS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {ADSB_REAPI_PENDING_UPLOADS_PATH}")

    pending_uploads: list[dict[str, str]] = []
    for record in data:
        if not isinstance(record, dict):
            logging.warning("Skipping invalid pending upload record: %r", record)
            continue
        path = record.get("path")
        api_call_time = record.get("api_call_time")
        if isinstance(path, str) and isinstance(api_call_time, str):
            pending_uploads.append({"path": path, "api_call_time": api_call_time})
        elif isinstance(path, str) and isinstance(record.get("partition"), str):
            # Backward-compatible migration from the previous pending queue format.
            api_call_time = _api_call_time_from_partition(record["partition"])
            pending_uploads.append({"path": path, "api_call_time": api_call_time.isoformat()})
        else:
            logging.warning("Skipping invalid pending upload record: %r", record)
    return pending_uploads


def _save_pending_uploads(pending_uploads: list[dict[str, str]]) -> None:
    ADSB_REAPI_PENDING_UPLOADS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ADSB_REAPI_PENDING_UPLOADS_PATH.with_suffix(
        f"{ADSB_REAPI_PENDING_UPLOADS_PATH.suffix}.tmp"
    )
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(pending_uploads, f, separators=(",", ":"), sort_keys=True)
    tmp_path.replace(ADSB_REAPI_PENDING_UPLOADS_PATH)


def _pending_upload_bytes(
    pending_uploads: list[dict[str, str]],
) -> tuple[int, list[dict[str, str]]]:
    total_bytes = 0
    existing_uploads: list[dict[str, str]] = []
    for record in pending_uploads:
        path = Path(record["path"])
        if not path.exists():
            logging.warning("Dropping missing pending upload file: %s", path)
            continue
        total_bytes += path.stat().st_size
        existing_uploads.append(record)
    return total_bytes, existing_uploads


def _combine_and_upload_pending_parquets(
    pending_uploads: list[dict[str, str]],
) -> tuple[str | None, int]:
    import boto3

    pending_upload_bytes, pending_uploads = _pending_upload_bytes(pending_uploads)
    if not pending_uploads:
        _save_pending_uploads([])
        return None, 0

    s3 = boto3.client("s3")
    paths = [Path(record["path"]) for record in pending_uploads]
    latest_api_call_time = max(
        _parse_api_call_time(record["api_call_time"])
        for record in pending_uploads
    )
    upload_partition = _hive_partition_path(latest_api_call_time)
    batch_id = _pending_upload_batch_id(paths)
    batch_filename = f"batch-{latest_api_call_time.strftime('%Y%m%dT%H%M%S%fZ')}-{batch_id}.parquet"
    batch_path = ADSB_REAPI_DATA_ROOT / upload_partition / batch_filename
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    if batch_path.exists():
        batch_path.unlink()

    dfs = [_read_upload_parquet_normalized(path) for path in paths]
    df = pl.concat(dfs, how="vertical")
    df.write_parquet(batch_path, compression="zstd")

    s3_key = (
        f"{ADSB_REAPI_S3_PREFIX}/{upload_partition}/{batch_filename}"
        if ADSB_REAPI_S3_PREFIX
        else f"{upload_partition}/{batch_filename}"
    )
    s3.upload_file(str(batch_path), ADSB_REAPI_S3_BUCKET, s3_key)

    for path in paths:
        _delete_file_if_exists(path)
    _delete_file_if_exists(batch_path)
    _save_pending_uploads([])
    return s3_key, 0


def _pending_upload_batch_id(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for path in sorted(paths, key=lambda p: str(p)):
        stat = path.stat()
        h.update(str(path).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(stat.st_mtime_ns).encode("utf-8"))
    return h.hexdigest()[:16]


def _read_upload_parquet_normalized(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    expressions = []
    for column, dtype in ADSB_MESSAGE_POLARS_SCHEMA.items():
        if column in df.columns:
            expressions.append(pl.col(column).cast(dtype, strict=False).alias(column))
        else:
            expressions.append(pl.lit(None).cast(dtype).alias(column))
    return df.select(expressions)


def _delete_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logging.exception("Failed to delete uploaded local parquet file: %s", path)


def _parse_api_call_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _api_call_time_from_partition(partition: str) -> datetime:
    parts = {}
    for segment in partition.split("/"):
        key, value = segment.split("=", 1)
        parts[key] = int(value)
    return datetime(
        parts["year"],
        parts["month"],
        parts["day"],
        parts["hour"],
        tzinfo=timezone.utc,
    )


def _hive_partition_path(dt: datetime) -> Path:
    dt = dt.astimezone(timezone.utc)
    return (
        Path(f"year={dt.year:04d}")
        / f"month={dt.month:02d}"
        / f"day={dt.day:02d}"
        / f"hour={dt.hour:02d}"
    )

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes):02d}:{sec:05.2f} (mm:ss)"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}:{int(minutes):02d}:{int(sec):02d} (hh:mm:ss)"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADS-B message processor')
    parser.add_argument('--primary-source', choices=['adsb.lol', 'adsb.fi'], 
                        default='adsb.lol', help='Primary ADS-B data source to try first')
    parser.add_argument(
        "--write-to-s3",
        action="store_true",
        help="Upload created ReAPI parquet files to S3 after writing them locally",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Proxy bind address")
    parser.add_argument("--port", type=int, default=8000, help="Proxy HTTP port")
    parser.add_argument(
        "--no-http",
        action="store_true",
        help="Disable the HTTP proxy and only run ADS-B ingestion",
    )
    args = parser.parse_args()
    
    # python -m src.adsb_updating.main
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    print(f"Starting adsb_updating in {environment} environment")
    print(f"Primary source: {args.primary_source}")
    print(f"Writing ADS-B ReAPI parquet locally under {ADSB_REAPI_DATA_ROOT}")
    if args.write_to_s3:
        print(f"Uploading ADS-B ReAPI parquet to s3://{ADSB_REAPI_S3_BUCKET}/{ADSB_REAPI_S3_PREFIX}")
        print(f"Combining and uploading pending parquet after {ADSB_REAPI_UPLOAD_THRESHOLD_BYTES} bytes")
    else:
        print("S3 upload disabled; pass --write-to-s3 to upload parquet files")
    logging.info("Logging at Info level")
    get_last_message_time_for_icaos()
    message_cache = AdsbMessageCache(
        ADSB_PROXY_CACHE_PATH,
        window_seconds=ADSB_PROXY_CACHE_WINDOW_SECONDS,
    )
    proxy_server = None
    if not args.no_http:
        proxy_server = start_proxy_server(
            message_cache,
            args.host,
            args.port,
            token=ADSB_PROXY_TOKEN,
        )
    interval = 60*2  # 2 minute
    data_source = args.primary_source
    try:
        while True:
            loop_start = time.perf_counter()
            try:
                data_source = process_adsb_message_stream(
                    args.primary_source,
                    write_to_s3=args.write_to_s3,
                    message_cache=message_cache,
                )
            except Exception as e:
                logging.exception("Error in process_adsb_message_stream: %s", e)
            elapsed = time.perf_counter() - loop_start
            interval = 30 if data_source == "adsb.lol" else 60 * 2
            sleep_time = interval - elapsed
            print(f"Sleeping for {sleep_time:.2f} seconds before next fetch...")
            if sleep_time > 0:
                time.sleep(sleep_time)
            # if sleep_time <= 0, loop continues immediately
    except KeyboardInterrupt:
        logging.info("Stopping ADS-B service")
    finally:
        if proxy_server is not None:
            proxy_server.shutdown()
            proxy_server.server_close()
