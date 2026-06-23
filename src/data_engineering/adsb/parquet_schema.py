"""Import-safe mirror of download_adsb_to_parquet.py's ADS-B parquet schema."""

import pyarrow as pa


# Column names in canonical order, matching the Iceberg `adsb_messages` table.
COLUMNS = [
    "time", "icao", "callsign", "registration", "aircraft_type",
    "aircraft-year",
    "lat", "lon", "radius_of_containment_m", "on_ground",
    "baro_altitude_ft", "geom_altitude_ft", "baro_rate_fpm", "geom_rate_fpm",
    "ground_speed_kt", "indicated_airspeed_kt", "true_airspeed_kt", "mach",
    "track_deg", "track_rate_deg_s", "roll_deg",
    "magnetic_heading_deg", "true_heading_deg",
    "nav_qnh_hpa", "nav_altitude_mcp_ft", "nav_altitude_fms_ft",
    "nav_heading_deg", "nav_modes",
    "squawk", "emergency", "alert", "spi",
    "version", "nic", "nic_baro", "nac_p", "nac_v",
    "sil", "sil_type", "gva", "sda",
    "type", "flags", "rssi_dbfs",
    "no_reg_data", "pia", "ladd", "military", "interesting",
    "owner", "aircraft_description", "category",
    "wind_direction_deg", "wind_speed_kt",
    "outside_air_temp_c", "total_air_temp_c",
    "mlat", "tisb",
]


# PyArrow schema for Parquet writing. Mirrors the Iceberg `adsb_messages` table.
PARQUET_SCHEMA = pa.schema([
    # identity / time
    ("time", pa.timestamp("ms")),
    ("icao", pa.string()),
    ("callsign", pa.string()),
    ("registration", pa.string()),
    ("aircraft_type", pa.string()),
    ("aircraft-year", pa.int32()),

    # position / state
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("radius_of_containment_m", pa.int32()),
    ("on_ground", pa.bool_()),

    # altitude / vertical motion
    ("baro_altitude_ft", pa.int32()),
    ("geom_altitude_ft", pa.int32()),
    ("baro_rate_fpm", pa.int32()),
    ("geom_rate_fpm", pa.int32()),

    # speed / horizontal motion
    ("ground_speed_kt", pa.float32()),
    ("indicated_airspeed_kt", pa.float32()),
    ("true_airspeed_kt", pa.float32()),
    ("mach", pa.float32()),

    # heading / orientation
    ("track_deg", pa.float32()),
    ("track_rate_deg_s", pa.float32()),
    ("roll_deg", pa.float32()),
    ("magnetic_heading_deg", pa.float32()),
    ("true_heading_deg", pa.float32()),

    # navigation
    ("nav_qnh_hpa", pa.float32()),
    ("nav_altitude_mcp_ft", pa.int32()),
    ("nav_altitude_fms_ft", pa.int32()),
    ("nav_heading_deg", pa.float32()),
    ("nav_modes", pa.list_(pa.string())),

    # transponder / status
    ("squawk", pa.string()),
    ("emergency", pa.string()),
    ("alert", pa.bool_()),
    ("spi", pa.bool_()),

    # ADS-B integrity / accuracy
    ("version", pa.int32()),
    ("nic", pa.int32()),
    ("nic_baro", pa.int32()),
    ("nac_p", pa.int32()),
    ("nac_v", pa.int32()),
    ("sil", pa.int32()),
    ("sil_type", pa.string()),
    ("gva", pa.int32()),
    ("sda", pa.int32()),

    # message metadata
    ("type", pa.string()),
    ("flags", pa.int32()),
    ("rssi_dbfs", pa.float32()),

    # aircraft metadata (dbFlags bitfield split out)
    ("no_reg_data", pa.bool_()),
    ("pia", pa.bool_()),
    ("ladd", pa.bool_()),
    ("military", pa.bool_()),
    ("interesting", pa.bool_()),
    ("owner", pa.string()),
    ("aircraft_description", pa.string()),
    ("category", pa.string()),

    # derived weather / air data
    ("wind_direction_deg", pa.float32()),
    ("wind_speed_kt", pa.float32()),
    ("outside_air_temp_c", pa.float32()),
    ("total_air_temp_c", pa.float32()),

    # data provenance
    ("mlat", pa.list_(pa.string())),
    ("tisb", pa.list_(pa.string())),
])
