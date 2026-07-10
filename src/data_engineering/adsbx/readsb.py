import argparse
import gzip
import re
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from data_engineering.adsb.parquet_schema import COLUMNS, PARQUET_SCHEMA


DEFAULT_INPUT_PATH = Path(
    "/Volumes/T2-SSD/planequery/data/raw/adsb-exchange/readsb-hist/"
    "2026/03/01/000000Z.json.gz"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/T2-SSD/planequery/data/processed/adsb-exchange/readsb-hist/parquet"
)
FILENAME_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})Z\.json\.gz$")


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


def _is_15_second_snapshot(path: Path) -> bool:
    match = FILENAME_RE.match(path.name)
    if not match:
        return False
    hour, minute, second = (int(part) for part in match.groups())
    return (hour * 3600 + minute * 60 + second) % 15 == 0


def readsb_snapshot_files(
    input_path: Path,
    cadence_seconds: int = 15,
    start_after: str | None = None,
) -> list[Path]:
    """Return readsb-hist snapshot files sampled from a 5-second archive."""
    if cadence_seconds != 15:
        raise ValueError("Only 15-second sampling is currently supported")

    if input_path.is_file():
        input_dir = input_path.parent
    else:
        input_dir = input_path

    files = sorted(input_dir.glob("*.json.gz"))
    sampled_files = [path for path in files if _is_15_second_snapshot(path)]
    if start_after is not None:
        sampled_files = [path for path in sampled_files if path.name > start_after]
    return sampled_files


def default_output_path(input_path: Path) -> Path:
    day_dir = input_path.parent if input_path.is_file() else input_path
    year, month, day = day_dir.parts[-3:]
    return (
        DEFAULT_OUTPUT_ROOT
        / f"year={year}"
        / f"month={month}"
        / f"day={day}"
        / "part-00000.parquet"
    )


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

    altitude = aircraft.get("alt_baro")
    on_ground = altitude == "ground"
    baro_altitude_ft = None if on_ground else _to_int(altitude)

    no_reg_data_raw = aircraft.get("noRegData")
    no_reg_data = bool(no_reg_data_raw) if no_reg_data_raw is not None else None
    military, interesting, pia, ladd = _db_flags(aircraft.get("dbFlags"))

    cols["time"].append(int((now - float(seen_pos)) * 1000))
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

    now = float(data["now"])
    cols = {col: [] for col in COLUMNS}
    for aircraft in data.get("aircraft", []):
        _append_aircraft(cols, aircraft, now)
    return cols


def _extend_cols(target: dict[str, list], source: dict[str, list]) -> int:
    row_count = len(source["time"])
    if not row_count:
        return 0
    for col in COLUMNS:
        target[col].extend(source[col])
    return row_count


def _write_batch(writer: pq.ParquetWriter, cols: dict[str, list]) -> int:
    row_count = len(cols["time"])
    arrays = [pa.array(cols[col], type=PARQUET_SCHEMA.field(col).type) for col in COLUMNS]
    table = pa.table(dict(zip(COLUMNS, arrays)), schema=PARQUET_SCHEMA)
    writer.write_table(table)
    return row_count


def write_readsb_day_to_parquet(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path | None = None,
    batch_size: int = 250_000,
    start_after: str | None = None,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else default_output_path(input_path)
    files = readsb_snapshot_files(input_path, start_after=start_after)
    if not files:
        raise FileNotFoundError(f"No 15-second readsb snapshots found under {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_cols = {col: [] for col in COLUMNS}
    batch_count = 0
    total_rows = 0
    skipped_files = []

    with pq.ParquetWriter(output_path, PARQUET_SCHEMA, compression="zstd") as writer:
        for index, path in enumerate(files, start=1):
            try:
                snapshot_cols = read_snapshot(path)
            except (EOFError, OSError, orjson.JSONDecodeError) as exc:
                skipped_files.append((path.name, exc))
                print(f"skipping {path.name}: {type(exc).__name__}: {exc}")
                continue

            batch_count += _extend_cols(batch_cols, snapshot_cols)
            if batch_count >= batch_size:
                written_rows = _write_batch(writer, batch_cols)
                total_rows += written_rows
                print(f"wrote {written_rows:,} rows ({total_rows:,} total) at {path.name}")
                batch_cols = {col: [] for col in COLUMNS}
                batch_count = 0
            if index % 500 == 0:
                print(f"processed {index:,}/{len(files):,} files")

        if batch_count:
            written_rows = _write_batch(writer, batch_cols)
            total_rows += written_rows
            print(f"wrote {written_rows:,} final rows ({total_rows:,} total)")

    if skipped_files:
        print("skipped files:")
        for name, exc in skipped_files:
            print(f"  {name}: {type(exc).__name__}: {exc}")
    print(f"wrote parquet: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--start-after")
    args = parser.parse_args()

    write_readsb_day_to_parquet(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        start_after=args.start_after,
    )


if __name__ == "__main__":
    main()
