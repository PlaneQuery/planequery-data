#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
LOCK_FILE = Path("/tmp/planequery-daily.lock")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_range(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current < end:
        days.append(current)
        current += timedelta(days=1)
    return days


def next_day(day: date) -> date:
    return day + timedelta(days=1)


def fmt(day: date) -> str:
    return day.strftime("%Y-%m-%d")


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def inference_feature_days(adsb_days: list[date]) -> list[date]:
    days = sorted(set(adsb_days))
    latest_day = days[-1]
    feature_days = {days[0] - timedelta(days=1)}
    feature_days.update(day for day in days if day < latest_day)
    return sorted(feature_days)


@contextmanager
def pipeline_lock(lock_file: Path, wait: bool):
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("w", encoding="utf-8") as f:
        flags = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(f, flags)
        except BlockingIOError:
            print(f"Another daily pipeline run is already active: {lock_file}")
            yield False
            return
        yield True


def discover_adsb_days(args: argparse.Namespace) -> list[date]:
    if args.start_date:
        end_date = args.end_date or next_day(args.start_date)
        return date_range(args.start_date, end_date)

    from data_engineering.adsblol.check_for_new_release import (
        get_adsblol_releases_to_process,
    )

    return sorted(get_adsblol_releases_to_process())


def run_command(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(SRC_DIR)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def run_adsb_day(day: date, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "src/data_engineering/adsb/download_adsb_to_parquet.py",
        fmt(day),
        fmt(next_day(day)),
    ]
    if args.keep_folders:
        command.append("--keep-folders")
    if args.write_to_s3:
        command.append("--write-to-s3")
    if args.ingest_to_iceberg:
        command.append("--ingest-to-iceberg")
    run_command(command, args.dry_run)


def run_features_day(day: date, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "src/flights_ai_model/features.py",
        "--features",
        "inference",
        "--start-date",
        fmt(day),
        "--end-date",
        fmt(next_day(day)),
    ]
    if args.write_to_s3:
        command.append("--write-to-s3")
    run_command(command, args.dry_run)


def run_flights_day(day: date, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "src/data_engineering/flights/flights_from_model.py",
        fmt(day),
        fmt(next_day(day)),
    ]
    if args.write_to_s3:
        command.append("--write-to-s3")
    if args.ingest_to_iceberg:
        command.append("--ingest-to-iceberg")
    if args.prod:
        command.append("--prod")
    run_command(command, args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local daily PlaneQuery ADS-B, feature, and flight pipeline."
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        help="Manual ADS-B release start date, inclusive. Defaults to release discovery.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="Manual ADS-B release end date, exclusive. Defaults to start-date + 1 day.",
    )
    parser.add_argument("--lock-file", type=Path, default=LOCK_FILE)
    parser.add_argument("--wait-for-lock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-folders", action="store_true")
    parser.add_argument("--no-write-to-s3", dest="write_to_s3", action="store_false")
    parser.add_argument(
        "--no-ingest-to-iceberg",
        dest="ingest_to_iceberg",
        action="store_false",
    )
    parser.add_argument("--dev", dest="prod", action="store_false")
    parser.set_defaults(write_to_s3=True, ingest_to_iceberg=True, prod=True)
    return parser


def main() -> int:
    load_env_file(REPO_ROOT / ".env")
    args = build_parser().parse_args()
    if args.end_date and not args.start_date:
        raise SystemExit("--end-date requires --start-date")
    if args.start_date and args.end_date and args.end_date <= args.start_date:
        raise SystemExit("--end-date must be after --start-date")

    with pipeline_lock(args.lock_file, args.wait_for_lock) as have_lock:
        if not have_lock:
            return 0

        adsb_days = discover_adsb_days(args)
        if not adsb_days:
            print("No adsb.lol releases to process.")
            return 0

        feature_days = inference_feature_days(adsb_days)
        print("ADS-B release days:", ", ".join(fmt(day) for day in adsb_days))
        print("Feature/flight days:", ", ".join(fmt(day) for day in feature_days))

        for day in adsb_days:
            run_adsb_day(day, args)
        for day in feature_days:
            run_features_day(day, args)
        for day in feature_days:
            run_flights_day(day, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
