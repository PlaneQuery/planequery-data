import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
BUCKET = "planequery-swim-logs-prod"
from data_engineering.utils import OUTPUT_DIR
BASE_DOWNLOAD_DIR = OUTPUT_DIR / "data/raw"

SOURCE_CONFIG = {
    "sfdps": "sfdps-logs",
    "tfms": "tfms-logs",
}

_s3_client = None
_s3_paginator = None


def _get_s3_client_and_paginator():
    global _s3_client, _s3_paginator
    if _s3_client is None or _s3_paginator is None:
        try:
            import boto3
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "boto3 is required only for S3 log downloads. Install it with: pip install boto3"
            ) from exc
        _s3_client = boto3.client("s3")
        _s3_paginator = _s3_client.get_paginator("list_objects_v2")
    return _s3_client, _s3_paginator


def date_range(start: date, end: date):
    """Yield each date from start up to but not including end."""
    current = start
    while current < end:
        yield current
        current += timedelta(days=1)


def prefix_for_date(d: date, s3_folder: str) -> str:
    return f"{s3_folder}/year={d.year}/month={d.month:02d}/day={d.day:02d}/"


def gather_keys(prefix: str) -> list[str]:
    _, paginator = _get_s3_client_and_paginator()
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = os.path.basename(key)
            if filename.endswith(".gz") or filename.endswith(".gzip"):
                keys.append(key)
    return keys


def download_file(key: str, download_dir: str):
    s3, _ = _get_s3_client_and_paginator()
    filename = os.path.basename(key)
    local_gz_path = os.path.join(download_dir, filename)
    s3.download_file(BUCKET, key, local_gz_path)
    print(f"Downloaded: {local_gz_path}")


def download_swim_logs(source: str, start_date: date, end_date: date, workers: int = 3):
    if source not in SOURCE_CONFIG:
        raise ValueError(f"source must be one of: {', '.join(SOURCE_CONFIG)}")
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date.")

    s3_folder = SOURCE_CONFIG[source]

    all_keys: list[tuple[str, str]] = []  # (s3_key, local_download_dir)
    for d in date_range(start_date, end_date):
        prefix = prefix_for_date(d, s3_folder)
        download_dir = os.path.join(BASE_DOWNLOAD_DIR, prefix)
        os.makedirs(download_dir, exist_ok=True)
        keys = gather_keys(prefix)
        print(f"{d.isoformat()}: found {len(keys)} file(s) under {prefix}")
        all_keys.extend((key, download_dir) for key in keys)

    if not all_keys:
        print("No files found for the given date range.")
        return

    print(f"\nDownloading {len(all_keys)} file(s) with {workers} worker(s)...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(lambda job: download_file(*job), all_keys)

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SFDPS or TFMS logs from S3 for a date range."
    )
    parser.add_argument(
        "source",
        choices=list(SOURCE_CONFIG.keys()),
        help="Log source to download: 'sfdps' or 'tfms'.",
    )
    parser.add_argument(
        "start_date",
        type=date.fromisoformat,
        help="Start date (inclusive) in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "end_date",
        type=date.fromisoformat,
        help="End date (exclusive) in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel download workers (default: 3).",
    )
    args = parser.parse_args()

    download_swim_logs(
        source=args.source,
        start_date=args.start_date,
        end_date=args.end_date,
        workers=args.workers,
    )
