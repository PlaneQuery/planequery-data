from __future__ import annotations
from dataclasses import dataclass, fields
from datetime import datetime
from typing import List, Literal, Optional

import polars as pl


# ---- Enums ----

AircraftCategory = Literal[
    'A0','A1','A2','A3','A4','A5','A6','A7',
    'B0','B1','B2','B3','B4','B5','B6','B7',
    'C0','C1','C2','C3','C4','C5','C6','C7',
    'D0','D1','D2','D3','D4','D5','D6','D7',''
]

AircraftEmergency = Literal[
    '', 'none', 'general', 'downed', 'lifeguard',
    'minfuel', 'nordo', 'unlawful', 'reserved'
]

NavMode = Literal[
    'althold', 'approach', 'autopilot', 'lnav', 'tcas', 'vnav'
]

DataSource = Literal[
    '', 'adsb.lol', 'adsb.fi', 'other'
]


@dataclass
class Adsb_Message_reapi:
    """Raw realtime API message used by the S3-backed snapshot writer."""

    hex: str
    type: str
    flight: Optional[str] = None
    r: Optional[str] = None
    t: Optional[str] = None
    desc: Optional[str] = None
    ownOp: Optional[str] = None
    year: Optional[str] = None
    alt_baro: Optional[int] = None
    on_ground: bool = False
    alt_geom: Optional[int] = None
    gs: Optional[float] = None
    track: Optional[float] = None
    calc_track: Optional[float] = None
    baro_rate: Optional[int] = None
    geom_rate: Optional[int] = None
    true_heading: Optional[float] = None
    mag_heading: Optional[float] = None
    ias: Optional[int] = None
    tas: Optional[int] = None
    mach: Optional[float] = None
    track_rate: Optional[float] = None
    roll: Optional[float] = None
    wd: Optional[int] = None
    ws: Optional[int] = None
    oat: Optional[int] = None
    tat: Optional[int] = None
    squawk: Optional[str] = None
    emergency: Optional[str] = None
    category: Optional[str] = None
    nav_qnh: Optional[float] = None
    nav_altitude_mcp: Optional[int] = None
    nav_altitude_fms: Optional[int] = None
    nav_heading: Optional[float] = None
    nav_modes: Optional[List[str]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    gpsOkBefore: Optional[float] = None
    gpsOkLat: Optional[float] = None
    gpsOkLon: Optional[float] = None
    rr_lat: Optional[float] = None
    rr_lon: Optional[float] = None
    lastPosition: Optional[dict] = None
    nic: Optional[int] = None
    rc: Optional[int] = None
    seen_pos: Optional[float] = None
    version: Optional[int] = None
    nic_baro: Optional[int] = None
    nac_p: Optional[int] = None
    nac_v: Optional[int] = None
    sil: Optional[int] = None
    sil_type: Optional[str] = None
    gva: Optional[int] = None
    sda: Optional[int] = None
    alert: Optional[int] = None
    spi: Optional[int] = None
    dbFlags: Optional[int] = None
    mlat: Optional[List[str]] = None
    tisb: Optional[List[str]] = None
    messages: Optional[int] = None
    seen: Optional[float] = None
    rssi: Optional[float] = None
    acas_ra: Optional[dict] = None

    @staticmethod
    def from_json(data: dict) -> "Adsb_Message_reapi":
        alt_baro = data.get("alt_baro")
        on_ground = alt_baro == "ground"
        field_names = {field.name for field in fields(Adsb_Message_reapi)}
        return Adsb_Message_reapi(
            **{
                key: value
                for key, value in data.items()
                if key != "alt_baro" and key in field_names
            },
            alt_baro=None if on_ground else alt_baro,
            on_ground=on_ground,
        )

@dataclass
class AdsbMessageRow:
    # Primary
    time: datetime
    icao: str

    # Metadata
    r: str
    t: str
    dbFlags: int
    noRegData: bool
    ownOp: str
    year: int
    desc: str

    # Position / motion
    lat: float
    lon: float
    alt_baro: Optional[int]
    on_ground: bool
    gs: Optional[float]
    track: Optional[float]
    flags: int
    baro_rate: Optional[int]
    source: str
    alt_geom: Optional[int]
    geom_rate: Optional[int]
    ias: Optional[int]
    roll: Optional[float]

    # Aircraft integrity / accuracy
    alert: Optional[int]
    gva: Optional[int]
    nac_p: Optional[int]
    nac_v: Optional[int]
    nic: Optional[int]
    nic_baro: Optional[int]
    rc: Optional[int]
    sda: Optional[int]
    sil: Optional[int]
    sil_type: str
    spi: Optional[int]
    type: str
    version: Optional[int]

    # Aircraft identity
    category: AircraftCategory
    emergency: AircraftEmergency
    flight: str
    squawk: str

    # Navigation
    nav_altitude_fms: Optional[int]
    nav_altitude_mcp: Optional[int]
    nav_modes: List[NavMode]
    nav_qnh: Optional[float]
    mach: Optional[float]
    mag_heading: Optional[float]
    oat: Optional[int]
    tas: Optional[int]
    tat: Optional[int]
    true_heading: Optional[float]
    wd: Optional[int]
    ws: Optional[int]
    track_rate: Optional[float]
    nav_heading: Optional[float]

    # Signal arrays
    mlat: List[str]
    tisb: List[str]
    rssi: Optional[float]

    # Provenance
    data_source: DataSource


@dataclass(frozen=True)
class AdsbMessage:
    time: datetime
    icao: str
    lat: float
    lon: float
    callsign: Optional[str] = None
    registration: Optional[str] = None
    aircraft_type: Optional[str] = None
    aircraft_year: Optional[int] = None
    radius_of_containment_m: Optional[int] = None
    on_ground: bool = False
    baro_altitude_ft: Optional[int] = None
    geom_altitude_ft: Optional[int] = None
    baro_rate_fpm: Optional[int] = None
    geom_rate_fpm: Optional[int] = None
    ground_speed_kt: Optional[float] = None
    indicated_airspeed_kt: Optional[float] = None
    true_airspeed_kt: Optional[float] = None
    mach: Optional[float] = None
    track_deg: Optional[float] = None
    track_rate_deg_s: Optional[float] = None
    roll_deg: Optional[float] = None
    magnetic_heading_deg: Optional[float] = None
    true_heading_deg: Optional[float] = None
    nav_qnh_hpa: Optional[float] = None
    nav_altitude_mcp_ft: Optional[int] = None
    nav_altitude_fms_ft: Optional[int] = None
    nav_heading_deg: Optional[float] = None
    nav_modes: Optional[List[str]] = None
    squawk: Optional[str] = None
    emergency: Optional[str] = None
    alert: bool = False
    spi: bool = False
    version: Optional[int] = None
    nic: Optional[int] = None
    nic_baro: Optional[int] = None
    nac_p: Optional[int] = None
    nac_v: Optional[int] = None
    sil: Optional[int] = None
    sil_type: Optional[str] = None
    gva: Optional[int] = None
    sda: Optional[int] = None
    type: Optional[str] = None
    flags: Optional[int] = None
    rssi_dbfs: Optional[float] = None
    no_reg_data: bool = False
    pia: bool = False
    ladd: bool = False
    military: bool = False
    interesting: bool = False
    owner: Optional[str] = None
    aircraft_description: Optional[str] = None
    category: Optional[str] = None
    wind_direction_deg: Optional[float] = None
    wind_speed_kt: Optional[float] = None
    outside_air_temp_c: Optional[float] = None
    total_air_temp_c: Optional[float] = None
    mlat: Optional[List[str]] = None
    tisb: Optional[List[str]] = None

    @property
    def dbFlags(self) -> int:
        flags = 0
        if self.military:
            flags |= 1
        if self.interesting:
            flags |= 2
        if self.pia:
            flags |= 4
        if self.ladd:
            flags |= 8
        return flags


def _bool_or_false(value: bool | None) -> bool:
    return False if value is None else bool(value)


def adsb_messages_from_parquet_df(df: "pl.DataFrame") -> list[AdsbMessage]:
    return [
        AdsbMessage(
            time=row["time"],
            icao=row["icao"],
            lat=row["lat"],
            lon=row["lon"],
            callsign=row.get("callsign"),
            registration=row.get("registration"),
            aircraft_type=row.get("aircraft_type"),
            aircraft_year=row.get("aircraft-year"),
            radius_of_containment_m=row.get("radius_of_containment_m"),
            on_ground=_bool_or_false(row.get("on_ground")),
            baro_altitude_ft=row.get("baro_altitude_ft"),
            geom_altitude_ft=row.get("geom_altitude_ft"),
            baro_rate_fpm=row.get("baro_rate_fpm"),
            geom_rate_fpm=row.get("geom_rate_fpm"),
            ground_speed_kt=row.get("ground_speed_kt"),
            indicated_airspeed_kt=row.get("indicated_airspeed_kt"),
            true_airspeed_kt=row.get("true_airspeed_kt"),
            mach=row.get("mach"),
            track_deg=row.get("track_deg"),
            track_rate_deg_s=row.get("track_rate_deg_s"),
            roll_deg=row.get("roll_deg"),
            magnetic_heading_deg=row.get("magnetic_heading_deg"),
            true_heading_deg=row.get("true_heading_deg"),
            nav_qnh_hpa=row.get("nav_qnh_hpa"),
            nav_altitude_mcp_ft=row.get("nav_altitude_mcp_ft"),
            nav_altitude_fms_ft=row.get("nav_altitude_fms_ft"),
            nav_heading_deg=row.get("nav_heading_deg"),
            nav_modes=row.get("nav_modes"),
            squawk=row.get("squawk"),
            emergency=row.get("emergency"),
            alert=_bool_or_false(row.get("alert")),
            spi=_bool_or_false(row.get("spi")),
            version=row.get("version"),
            nic=row.get("nic"),
            nic_baro=row.get("nic_baro"),
            nac_p=row.get("nac_p"),
            nac_v=row.get("nac_v"),
            sil=row.get("sil"),
            sil_type=row.get("sil_type"),
            gva=row.get("gva"),
            sda=row.get("sda"),
            type=row.get("type"),
            flags=row.get("flags"),
            rssi_dbfs=row.get("rssi_dbfs"),
            no_reg_data=_bool_or_false(row.get("no_reg_data")),
            pia=_bool_or_false(row.get("pia")),
            ladd=_bool_or_false(row.get("ladd")),
            military=_bool_or_false(row.get("military")),
            interesting=_bool_or_false(row.get("interesting")),
            owner=row.get("owner"),
            aircraft_description=row.get("aircraft_description"),
            category=row.get("category"),
            wind_direction_deg=row.get("wind_direction_deg"),
            wind_speed_kt=row.get("wind_speed_kt"),
            outside_air_temp_c=row.get("outside_air_temp_c"),
            total_air_temp_c=row.get("total_air_temp_c"),
            mlat=row.get("mlat"),
            tisb=row.get("tisb"),
        )
        for row in df.iter_rows(named=True)
    ]
