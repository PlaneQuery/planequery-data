#!/usr/bin/env python3
"""
Export ClickHouse MergeTree daily partitions to S3 in Hive layout:
  s3://BUCKET/PREFIX/year=YYYY/month=MM/day=DD/data.parquet

Supports exporting a specified inclusive date range:
  ./export_adsb_messages.py 2025-01-03 2025-01-05

Requires env vars:
  CH_HOST, CH_USERNAME, CH_PASSWORD, S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
Optional env vars:
  CH_PORT=8443, CH_SECURE=true, CH_DATABASE=default, TABLE=adsb_messages, S3_PREFIX=exports/adsb_messages
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import clickhouse_connect


DONE_FILE = Path("export_done.txt")


def env(name: str, default: str | None = None, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and (v is None or v.strip() == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def parse_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "y", "on")


def sql_str(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def load_done_set(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_done(path: Path, part: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(part + "\n")


def parse_iso_date(s: str) -> date:
    # Accept YYYY-MM-DD
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date '{s}'. Expected format YYYY-MM-DD.") from e


def daterange_inclusive(start: date, end: date) -> Iterable[date]:
    if end < start:
        raise ValueError(f"End date {end} is before start date {start}.")
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def yyyymmdd_int(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def hive_key(prefix: str, d: date) -> str:
    yyyy = d.strftime("%Y")
    mm = d.strftime("%m")
    dd = d.strftime("%d")
    return f"{prefix.rstrip('/')}/year={yyyy}/month={mm}/day={dd}/data.parquet"


def export_one_day(
    client,
    db: str,
    table: str,
    p_yyyymmdd: int,
    s3_bucket: str,
    s3_key: str,
    aws_key: str,
    aws_secret: str,
) -> None:
    q = f"""
    INSERT INTO FUNCTION s3(
      'https://s3.amazonaws.com/{s3_bucket}/{s3_key}',
      {sql_str(aws_key)},
      {sql_str(aws_secret)},
      'Parquet'
    )
    SELECT *
    FROM {db}.{table}
    WHERE toYYYYMMDD(time) = {p_yyyymmdd}
    """
    client.command(q)


def usage() -> str:
    return (
        "Usage:\n"
        "  export_adsb_messages.py <start_date> <end_date>\n\n"
        "Example:\n"
        "  export_adsb_messages.py 2025-01-03 2025-01-05\n"
    )


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(usage(), file=sys.stderr)
        return 2

    start = parse_iso_date(argv[1])
    end = parse_iso_date(argv[2])

    ch_host = env("CH_HOST", required=True)
    ch_user = env("CH_USERNAME", required=True)
    ch_pass = env("CH_PASSWORD", required=True)

    ch_port = int(env("CH_PORT", "8443"))
    ch_secure = parse_bool(env("CH_SECURE", "true"))
    db = env("CH_DATABASE", "default")
    table = env("TABLE", "adsb_messages")

    s3_bucket = env("S3_BUCKET", required=True)
    s3_prefix = env("S3_PREFIX", "exports/adsb_messages")

    aws_key = env("AWS_ACCESS_KEY_ID", required=True)
    aws_secret = env("AWS_SECRET_ACCESS_KEY", required=True)

    client = clickhouse_connect.get_client(
        host=ch_host,
        username=ch_user,
        password=ch_pass,
        port=ch_port,
        secure=ch_secure,
        database=db,
        connect_timeout=30,
        send_receive_timeout=600,  # raise for very large days
    )

    done = load_done_set(DONE_FILE)
    days = list(daterange_inclusive(start, end))

    print(f"Exporting {len(days)} day(s): {start.isoformat()} .. {end.isoformat()}")
    print(f"Resume markers loaded: {len(done)} day(s) already done (by YYYYMMDD in {DONE_FILE})")

    for i, d in enumerate(days, start=1):
        p = yyyymmdd_int(d)  # ClickHouse partition key value
        p_str = str(p)

        if p_str in done:
            print(f"[{i}/{len(days)}] skip {d.isoformat()} (partition {p_str}) already done")
            continue

        key = hive_key(s3_prefix, d)
        print(f"[{i}/{len(days)}] export {d.isoformat()} (partition {p_str}) -> s3://{s3_bucket}/{key}")

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                export_one_day(
                    client,
                    db=db,
                    table=table,
                    p_yyyymmdd=p,
                    s3_bucket=s3_bucket,
                    s3_key=key,
                    aws_key=aws_key,
                    aws_secret=aws_secret,
                )
                append_done(DONE_FILE, p_str)
                done.add(p_str)
                print(f"  done {d.isoformat()}")
                break
            except Exception as e:
                print(f"  attempt {attempt}/{max_attempts} failed: {e}", file=sys.stderr)
                if attempt == max_attempts:
                    raise
                time.sleep(10 * attempt)

    print("Completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
