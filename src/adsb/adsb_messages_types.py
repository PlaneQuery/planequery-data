from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, List

@dataclass
class Adsb_Message_reapi:
    # Identification
    hex: str
    type: str
    flight: Optional[str] = None
    r: Optional[str] = None
    t: Optional[str] = None

    # Metadata
    desc: Optional[str] = None
    ownOp: Optional[str] = None
    year: Optional[str] = None

    # Altitude
    alt_baro: Optional[int] = None
    on_ground: bool = False
    alt_geom: Optional[int] = None

    # Kinematics
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

    # Transponder / status
    squawk: Optional[str] = None
    emergency: Optional[str] = None
    category: Optional[str] = None

    # Navigation
    nav_qnh: Optional[float] = None
    nav_altitude_mcp: Optional[int] = None
    nav_altitude_fms: Optional[int] = None
    nav_heading: Optional[float] = None
    nav_modes: Optional[List[str]] = None

    # Position
    lat: Optional[float] = None
    lon: Optional[float] = None
    gpsOkBefore: Optional[float] = None
    gpsOkLat: Optional[float] = None
    gpsOkLon: Optional[float] = None
    rr_lat: Optional[float] = None
    rr_lon: Optional[float] = None
    lastPosition: Optional[dict] = None

    # Integrity / accuracy
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

    # Flags
    alert: Optional[int] = None
    spi: Optional[int] = None
    dbFlags: Optional[int] = None

    # Arrays
    mlat: Optional[List[str]] = None
    tisb: Optional[List[str]] = None

    # Counters / signal
    messages: Optional[int] = None
    seen: Optional[float] = None
    rssi: Optional[float] = None

    # ACAS Resolution Advisory
    acas_ra: Optional[dict] = None

    @staticmethod
    def from_json(d: dict) -> Adsb_Message_reapi:
        alt_baro_val = d.get("alt_baro")
        on_ground = alt_baro_val == "ground"

        return Adsb_Message_reapi(
            **{k: v for k, v in d.items() if k not in {"alt_baro"}},
            alt_baro=None if on_ground else alt_baro_val,
            on_ground=on_ground,
        )


from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Literal


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
    '', 'adsb.lol', 'other'
]


# ---- Row type ----

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
    ground_speed: Optional[float]
    track_degrees: Optional[float]
    flags: int
    vertical_rate: Optional[int]
    source: str
    geometric_altitude: Optional[int]
    geometric_vertical_rate: Optional[int]
    indicated_airspeed: Optional[int]
    roll_angle: Optional[float]

    # Aircraft integrity / accuracy
    aircraft_alert: Optional[int]
    aircraft_alt_geom: Optional[int]
    aircraft_gva: Optional[int]
    aircraft_nac_p: Optional[int]
    aircraft_nac_v: Optional[int]
    aircraft_nic: Optional[int]
    aircraft_nic_baro: Optional[int]
    aircraft_rc: Optional[int]
    aircraft_sda: Optional[int]
    aircraft_sil: Optional[int]
    aircraft_sil_type: str
    aircraft_spi: Optional[int]
    aircraft_track: Optional[float]
    aircraft_type: str
    aircraft_version: Optional[int]

    # Aircraft identity
    aircraft_category: AircraftCategory
    aircraft_emergency: AircraftEmergency
    aircraft_flight: str
    aircraft_squawk: str

    # Navigation
    aircraft_baro_rate: Optional[int]
    aircraft_nav_altitude_fms: Optional[int]
    aircraft_nav_altitude_mcp: Optional[int]
    aircraft_nav_modes: List[NavMode]
    aircraft_nav_qnh: Optional[float]
    aircraft_geom_rate: Optional[int]
    aircraft_ias: Optional[int]
    aircraft_mach: Optional[float]
    aircraft_mag_heading: Optional[float]
    aircraft_oat: Optional[int]
    aircraft_roll: Optional[float]
    aircraft_tas: Optional[int]
    aircraft_tat: Optional[int]
    aircraft_true_heading: Optional[float]
    aircraft_wd: Optional[int]
    aircraft_ws: Optional[int]
    aircraft_track_rate: Optional[float]
    aircraft_nav_heading: Optional[float]

    # Provenance
    data_source: DataSource


def reapi_to_row(
    msg: Adsb_Message_reapi,
    *,
    time: datetime,
    data_source: str,
    flags: int = 0,
    noRegData: bool = False,
    ownOp: str = "",
    year: int = 0,
    desc: str = "",
) -> AdsbMessageRow:
    """
    Convert a reapi ADS-B message into a ClickHouse adsb_messages row.
    """
    
    # Use calc_track as fallback for track
    track_value = msg.track or msg.calc_track
    
    # Handle missing lat/lon by using lastPosition if available
    lat_value = msg.lat
    lon_value = msg.lon
    if lat_value is None and msg.lastPosition:
        lat_value = msg.lastPosition.get('lat')
    if lon_value is None and msg.lastPosition:
        lon_value = msg.lastPosition.get('lon')

    # Handle year conversion safely
    year_value = year
    if msg.year:
        try:
            year_value = int(msg.year)
        except (ValueError, TypeError):
            year_value = year

    return AdsbMessageRow(
        # Primary
        time=time,
        icao=msg.hex.strip(),

        # Metadata - use message fields if available, fall back to parameters
        r=(msg.r or "").strip(),
        t=(msg.t or "").strip(),
        dbFlags=msg.dbFlags or 0,
        noRegData=noRegData,
        ownOp=msg.ownOp or ownOp,
        year=year_value,
        desc=msg.desc or desc,

        # Position / motion
        lat=lat_value or 0.0,
        lon=lon_value or 0.0,
        alt_baro=msg.alt_baro,
        on_ground=msg.on_ground,
        ground_speed=msg.gs,
        track_degrees=track_value,
        flags=flags,
        vertical_rate=msg.baro_rate,
        source="",
        geometric_altitude=msg.alt_geom,
        geometric_vertical_rate=msg.geom_rate,
        indicated_airspeed=msg.ias,
        roll_angle=msg.roll,

        # Aircraft integrity / accuracy
        aircraft_alert=msg.alert,
        aircraft_alt_geom=msg.alt_geom,
        aircraft_gva=msg.gva,
        aircraft_nac_p=msg.nac_p,
        aircraft_nac_v=msg.nac_v,
        aircraft_nic=msg.nic,
        aircraft_nic_baro=msg.nic_baro,
        aircraft_rc=msg.rc,
        aircraft_sda=msg.sda,
        aircraft_sil=msg.sil,
        aircraft_sil_type=msg.sil_type or "",
        aircraft_spi=msg.spi,
        aircraft_track=track_value,
        aircraft_type=msg.t or "",
        aircraft_version=msg.version,

        # Aircraft identity
        aircraft_category=msg.category or "",
        aircraft_emergency=msg.emergency or "",
        aircraft_flight=(msg.flight or "").strip(),
        aircraft_squawk=msg.squawk or "",

        # Navigation
        aircraft_baro_rate=msg.baro_rate,
        aircraft_nav_altitude_fms=msg.nav_altitude_fms,
        aircraft_nav_altitude_mcp=msg.nav_altitude_mcp,
        aircraft_nav_modes=list(msg.nav_modes or []),
        aircraft_nav_qnh=msg.nav_qnh,
        aircraft_geom_rate=msg.geom_rate,
        aircraft_ias=msg.ias,
        aircraft_mach=msg.mach,
        aircraft_mag_heading=msg.mag_heading,
        aircraft_oat=msg.oat,
        aircraft_roll=msg.roll,
        aircraft_tas=msg.tas,
        aircraft_tat=msg.tat,
        aircraft_true_heading=msg.true_heading,
        aircraft_wd=msg.wd,
        aircraft_ws=msg.ws,
        aircraft_track_rate=msg.track_rate,
        aircraft_nav_heading=msg.nav_heading,

        # Provenance
        data_source=data_source,
    )


import clickhouse_connect
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
    from clickhouse_connect.driver import Client


def insert_adsb_messages(client: 'Client', adsb_messages: List[AdsbMessageRow]) -> None:
    """
    Insert ADS-B message rows into the adsb_messages_realtime table.
    Uses dataclass field order to avoid manually specifying columns.
    """
    if not adsb_messages:
        return
    
    # Convert dataclass instances to tuples in field order
    rows = []
    for msg in adsb_messages:
        # Use dataclasses.astuple to get values in field definition order
        row_values = dataclasses.astuple(msg)
        rows.append(row_values)
    
    # Insert all rows at once - ClickHouse will match by column order
    client.insert('adsb_messages_realtime', rows)