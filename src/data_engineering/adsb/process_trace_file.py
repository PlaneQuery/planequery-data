"""Process one adsb.lol trace file into ADS-B message parquet columns."""

import gzip
import zlib
from pathlib import Path

import orjson

from data_engineering.adsb.deduplication import (
    dedupe_cols_by_time_icao as _dedupe_cols_by_time_icao,
)
from data_engineering.adsb.parquet_schema import COLUMNS

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


def process_file(
    filepath: str | Path,
    log_dedupes: bool = True,
    pia_or_american_ladd_only: bool = False,
) -> dict | None:
    """
    Process a single trace file and return a column-oriented dict (col -> list)
    matching the planequery_lake_dev.adsb_messages Iceberg schema.

    Trace row layout (post-2022):
        [t_offset, lat, lon, altitude, gs, track, flags, baro_rate,
         aircraft_obj, type, geom_alt, geom_rate, ias, roll]
    """
    filepath = Path(filepath)
    try:
        raw = filepath.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        data = orjson.loads(raw)
    except (OSError, EOFError, zlib.error, orjson.JSONDecodeError) as exc:
        print(f"Skipping unreadable trace file {filepath}: {exc}")
        return None

    icao = data.get('icao')
    timestamp = data.get('timestamp')
    trace_data = data.get('trace')
    if icao is None or timestamp is None or trace_data is None or not trace_data:
        return None

    # Constant per-file (top-level) fields. Empty strings -> None.
    registration = _str_or_none(data.get('r'))
    aircraft_type = _str_or_none(data.get('t'))
    aircraft_year = _to_int(data.get('year'))
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

    if pia_or_american_ladd_only:
        icao_is_american = "a00000" <= icao.lower() <= "afffff"
        if not (pia or (icao_is_american and ladd)):
            return None

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
        cols["aircraft-year"].append(aircraft_year)

        cols["no_reg_data"].append(no_reg_data)
        cols["pia"].append(pia)
        cols["ladd"].append(ladd)
        cols["military"].append(military)
        cols["interesting"].append(interesting)
        cols["owner"].append(owner)
        cols["aircraft_description"].append(aircraft_description)
        cols["category"].append(_str_or_none(aircraft.get('category')))

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
    if duplicate_count and log_dedupes:
        print(f"Deduped {duplicate_count} duplicate rows in {filepath}")

    return cols
