import argparse
import datetime as dt
import gzip
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from data_engineering.adsb.canonical import (
    ADSBSource,
    adsb_day_is_complete,
    adsb_parquet_root,
    mark_adsb_day_complete,
)
from data_engineering.adsb.parquet_schema import COLUMNS, PARQUET_SCHEMA
from data_engineering.adsb.partitions import ICAO_BUCKET_COUNT, icao_bucket
from data_engineering.adsbx.download_readsb_day import download_complete_path
from data_engineering.utils import OUTPUT_DIR


DEFAULT_INPUT_ROOT = OUTPUT_DIR / "data" / "raw" / "adsbx" / "readsb-hist"
DEFAULT_INPUT_PATH = DEFAULT_INPUT_ROOT / "2026" / "03" / "01"
DEFAULT_BUCKETED_OUTPUT_ROOT = adsb_parquet_root(ADSBSource.ADSBX)
FILENAME_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})Z\.json\.gz$")


def _day_dir(day: dt.date) -> Path:
    return DEFAULT_INPUT_ROOT / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"


def _has_snapshot_files(input_path: Path) -> bool:
    return any(input_path.glob("*.json.gz"))


def _ensure_day_downloaded(day: dt.date) -> Path:
    day_dir = _day_dir(day)
    completion_path = download_complete_path(day_dir)
    if completion_path.is_file() and _has_snapshot_files(day_dir):
        return day_dir

    downloader = Path(__file__).with_name("download_readsb_day.py")
    cmd = [
        sys.executable,
        str(downloader),
        day.isoformat(),
        "--out-dir",
        str(DEFAULT_INPUT_ROOT),
    ]
    print(f"Download for {day.isoformat()} is incomplete; downloading to {day_dir}")
    subprocess.run(cmd, check=True)

    if not completion_path.is_file() or not _has_snapshot_files(day_dir):
        raise RuntimeError(
            f"Download finished without marking a complete snapshot day under {day_dir}"
        )
    return day_dir


def _str_or_none(value):
    if value is None:
        return None
    value = value if isinstance(value, str) else str(value)
    value = value.strip()
    return value or None


def _to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def readsb_snapshot_files(
    input_path: Path,
    start_after: str | None = None,
) -> list[Path]:
    """Return every available readsb-hist snapshot, without cadence sampling."""
    if input_path.is_file():
        input_dir = input_path.parent
    else:
        input_dir = input_path

    files = sorted(
        path for path in input_dir.glob("*.json.gz") if FILENAME_RE.fullmatch(path.name)
    )
    if start_after is not None:
        files = [path for path in files if path.name > start_after]
    return files


def _db_flags(raw):
    if raw is None:
        return None, None, None, None
    try:
        raw = int(raw)
    except (TypeError, ValueError):
        return None, None, None, None
    military = bool(raw & 1)
    interesting = bool(raw & 2)
    pia = bool(raw & 4)
    ladd = bool(raw & 8)
    return military, interesting, pia, ladd


def _append_aircraft(cols: dict[str, list], aircraft: dict, now: float) -> bool:
    icao = _str_or_none(aircraft.get("hex"))
    seen_pos = aircraft.get("seen_pos")
    lat = aircraft.get("lat")
    lon = aircraft.get("lon")
    if icao is None or seen_pos is None or lat is None or lon is None:
        return False

    # Timestamp the position, not the snapshot or the last non-position message.
    # Without seen_pos, coordinates may be stale and cannot be timed reliably.
    try:
        lat, lon, seen_pos, now = map(float, (lat, lon, seen_pos, now))
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in (lat, lon, seen_pos, now)):
        return False
    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and 0 <= seen_pos <= now):
        return False
    position_time = now - seen_pos

    altitude = aircraft.get("alt_baro")
    on_ground = altitude == "ground"
    baro_altitude_ft = None if on_ground else _to_int(altitude)

    no_reg_data_raw = aircraft.get("noRegData")
    no_reg_data = bool(no_reg_data_raw) if no_reg_data_raw is not None else None
    military, interesting, pia, ladd = _db_flags(aircraft.get("dbFlags"))

    cols["time"].append(int(position_time * 1000))
    cols["icao"].append(icao)
    cols["callsign"].append(_str_or_none(aircraft.get("flight")))
    cols["registration"].append(_str_or_none(aircraft.get("r")))
    cols["aircraft_type"].append(_str_or_none(aircraft.get("t")))
    cols["aircraft-year"].append(_to_int(aircraft.get("year")))

    cols["lat"].append(lat)
    cols["lon"].append(lon)
    cols["radius_of_containment_m"].append(_to_int(aircraft.get("rc")))
    cols["on_ground"].append(on_ground)

    cols["baro_altitude_ft"].append(baro_altitude_ft)
    cols["geom_altitude_ft"].append(_to_int(aircraft.get("alt_geom")))
    cols["baro_rate_fpm"].append(_to_int(aircraft.get("baro_rate")))
    cols["geom_rate_fpm"].append(_to_int(aircraft.get("geom_rate")))

    cols["ground_speed_kt"].append(aircraft.get("gs"))
    cols["indicated_airspeed_kt"].append(aircraft.get("ias"))
    cols["true_airspeed_kt"].append(aircraft.get("tas"))
    cols["mach"].append(aircraft.get("mach"))

    cols["track_deg"].append(aircraft.get("track"))
    cols["track_rate_deg_s"].append(aircraft.get("track_rate"))
    cols["roll_deg"].append(aircraft.get("roll"))
    cols["magnetic_heading_deg"].append(aircraft.get("mag_heading"))
    cols["true_heading_deg"].append(aircraft.get("true_heading"))

    cols["nav_qnh_hpa"].append(aircraft.get("nav_qnh"))
    cols["nav_altitude_mcp_ft"].append(_to_int(aircraft.get("nav_altitude_mcp")))
    cols["nav_altitude_fms_ft"].append(_to_int(aircraft.get("nav_altitude_fms")))
    cols["nav_heading_deg"].append(aircraft.get("nav_heading"))
    cols["nav_modes"].append(aircraft.get("nav_modes") or [])

    cols["squawk"].append(_str_or_none(aircraft.get("squawk")))
    cols["emergency"].append(_str_or_none(aircraft.get("emergency")))
    alert = aircraft.get("alert")
    spi = aircraft.get("spi")
    cols["alert"].append(bool(alert) if alert is not None else None)
    cols["spi"].append(bool(spi) if spi is not None else None)

    cols["version"].append(aircraft.get("version"))
    cols["nic"].append(aircraft.get("nic"))
    cols["nic_baro"].append(aircraft.get("nic_baro"))
    cols["nac_p"].append(aircraft.get("nac_p"))
    cols["nac_v"].append(aircraft.get("nac_v"))
    cols["sil"].append(aircraft.get("sil"))
    cols["sil_type"].append(_str_or_none(aircraft.get("sil_type")))
    cols["gva"].append(aircraft.get("gva"))
    cols["sda"].append(aircraft.get("sda"))

    cols["type"].append(_str_or_none(aircraft.get("type")))
    cols["flags"].append(None)
    cols["rssi_dbfs"].append(aircraft.get("rssi"))

    cols["no_reg_data"].append(no_reg_data)
    cols["pia"].append(pia)
    cols["ladd"].append(ladd)
    cols["military"].append(military)
    cols["interesting"].append(interesting)
    cols["owner"].append(_str_or_none(aircraft.get("ownOp")))
    cols["aircraft_description"].append(_str_or_none(aircraft.get("desc")))
    cols["category"].append(_str_or_none(aircraft.get("category")))

    cols["wind_direction_deg"].append(aircraft.get("wd"))
    cols["wind_speed_kt"].append(aircraft.get("ws"))
    cols["outside_air_temp_c"].append(aircraft.get("oat"))
    cols["total_air_temp_c"].append(aircraft.get("tat"))

    cols["mlat"].append(aircraft.get("mlat") or [])
    cols["tisb"].append(aircraft.get("tisb") or [])
    return True


def read_snapshot(path: Path) -> dict[str, list]:
    with gzip.open(path, "rb") as f:
        data = orjson.loads(f.read())

    now = data.get("now")
    cols = {col: [] for col in COLUMNS}
    for aircraft in data.get("aircraft", []):
        _append_aircraft(cols, aircraft, now)
    return cols


def _write_batch(writer: pq.ParquetWriter, cols: dict[str, list]) -> int:
    row_count = len(cols["time"])
    arrays = [pa.array(cols[col], type=PARQUET_SCHEMA.field(col).type) for col in COLUMNS]
    table = pa.table(dict(zip(COLUMNS, arrays)), schema=PARQUET_SCHEMA)
    writer.write_table(table)
    return row_count


def _empty_cols() -> dict[str, list]:
    return {col: [] for col in COLUMNS}


def _bucketed_output_paths(input_path: Path, output_root: Path) -> dict[int, Path]:
    day_dir = input_path.parent if input_path.is_file() else input_path
    year, month, day = day_dir.parts[-3:]
    return {
        bucket: (
            output_root
            / f"year={year}"
            / f"month={month}"
            / f"day={day}"
            / f"icao_bucket={bucket}"
            / "data.parquet"
        )
        for bucket in range(ICAO_BUCKET_COUNT)
    }


def _extend_bucket_cols(
    batch_cols_by_bucket: dict[int, dict[str, list]],
    snapshot_cols: dict[str, list],
) -> dict[int, int]:
    row_counts_by_bucket = {bucket: 0 for bucket in range(ICAO_BUCKET_COUNT)}
    for row_idx, icao in enumerate(snapshot_cols["icao"]):
        bucket = icao_bucket(icao)
        for col in COLUMNS:
            batch_cols_by_bucket[bucket][col].append(snapshot_cols[col][row_idx])
        row_counts_by_bucket[bucket] += 1
    return row_counts_by_bucket


def _write_bucket_batch(
    writers: dict[int, pq.ParquetWriter | None],
    parquet_paths: dict[int, Path],
    batch_cols_by_bucket: dict[int, dict[str, list]],
    bucket: int,
) -> int:
    row_count = len(batch_cols_by_bucket[bucket]["time"])
    if not row_count:
        return 0
    parquet_paths[bucket].parent.mkdir(parents=True, exist_ok=True)
    if writers[bucket] is None:
        writers[bucket] = pq.ParquetWriter(
            parquet_paths[bucket],
            PARQUET_SCHEMA,
            compression="zstd",
        )
    _write_batch(writers[bucket], batch_cols_by_bucket[bucket])
    batch_cols_by_bucket[bucket] = _empty_cols()
    return row_count


def write_readsb_day_to_bucketed_parquet(
    input_path: Path,
    output_root: Path = DEFAULT_BUCKETED_OUTPUT_ROOT,
    batch_size: int = 250_000,
    start_after: str | None = None,
) -> dict[int, Path]:
    input_path = Path(input_path)
    output_root = Path(output_root)
    files = readsb_snapshot_files(input_path, start_after=start_after)
    if not files:
        raise FileNotFoundError(f"No readsb snapshots found under {input_path}")

    day_dir = next(iter(_bucketed_output_paths(input_path, output_root).values())).parents[1]
    if day_dir.exists():
        shutil.rmtree(day_dir)
    parquet_paths = _bucketed_output_paths(input_path, output_root)
    batch_cols_by_bucket = {
        bucket: _empty_cols()
        for bucket in range(ICAO_BUCKET_COUNT)
    }
    batch_count_by_bucket = {
        bucket: 0
        for bucket in range(ICAO_BUCKET_COUNT)
    }
    total_rows_by_bucket = {
        bucket: 0
        for bucket in range(ICAO_BUCKET_COUNT)
    }
    writers: dict[int, pq.ParquetWriter | None] = {
        bucket: None
        for bucket in range(ICAO_BUCKET_COUNT)
    }
    skipped_files = []

    try:
        for index, path in enumerate(files, start=1):
            try:
                snapshot_cols = read_snapshot(path)
            except (EOFError, OSError, orjson.JSONDecodeError) as exc:
                skipped_files.append((path.name, exc))
                print(f"skipping {path.name}: {type(exc).__name__}: {exc}")
                continue

            row_counts = _extend_bucket_cols(batch_cols_by_bucket, snapshot_cols)
            for bucket, row_count in row_counts.items():
                batch_count_by_bucket[bucket] += row_count
                total_rows_by_bucket[bucket] += row_count
                if batch_count_by_bucket[bucket] >= batch_size:
                    written_rows = _write_bucket_batch(
                        writers,
                        parquet_paths,
                        batch_cols_by_bucket,
                        bucket,
                    )
                    print(
                        f"wrote bucket {bucket}: {written_rows:,} rows "
                        f"({total_rows_by_bucket[bucket]:,} total)"
                    )
                    batch_count_by_bucket[bucket] = 0

            if index % 500 == 0:
                print(f"processed {index:,}/{len(files):,} files")

        for bucket in range(ICAO_BUCKET_COUNT):
            if batch_count_by_bucket[bucket]:
                written_rows = _write_bucket_batch(
                    writers,
                    parquet_paths,
                    batch_cols_by_bucket,
                    bucket,
                )
                print(
                    f"wrote bucket {bucket} final: {written_rows:,} rows "
                    f"({total_rows_by_bucket[bucket]:,} total)"
                )
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

    if skipped_files:
        print("skipped files:")
        for name, exc in skipped_files:
            print(f"  {name}: {type(exc).__name__}: {exc}")

    total_rows = sum(total_rows_by_bucket.values())
    mark_adsb_day_complete(day_dir)
    print(f"wrote bucketed parquet under: {output_root}")
    print(f"total rows written: {total_rows:,}")
    return parquet_paths


def ensure_readsb_day_bucketed_parquet(
    day: dt.date,
    output_root: Path = DEFAULT_BUCKETED_OUTPUT_ROOT,
) -> None:
    if Path(output_root) == DEFAULT_BUCKETED_OUTPUT_ROOT:
        complete = adsb_day_is_complete(ADSBSource.ADSBX, day)
    else:
        partition_dir = (
            Path(output_root)
            / f"year={day.year}"
            / f"month={day.month:02d}"
            / f"day={day.day:02d}"
        )
        complete = (partition_dir / "_SUCCESS").is_file()
    if complete:
        return

    input_path = _ensure_day_downloaded(day)
    write_readsb_day_to_bucketed_parquet(
        input_path=input_path,
        output_root=output_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "date",
        nargs="?",
        help="UTC date in YYYY-MM-DD format. Downloads day if missing.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_BUCKETED_OUTPUT_ROOT,
    )
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--start-after")
    args = parser.parse_args()

    if args.date:
        day = dt.date.fromisoformat(args.date)
        input_path = _ensure_day_downloaded(day)
    else:
        input_path = DEFAULT_INPUT_PATH

    write_readsb_day_to_bucketed_parquet(
        input_path=input_path,
        output_root=args.output_root,
        batch_size=args.batch_size,
        start_after=args.start_after,
    )


if __name__ == "__main__":
    main()
