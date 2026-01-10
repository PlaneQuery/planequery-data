from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

@dataclass
class Adsb_Message_reapi:
    # Identification
    hex: str
    type: str
    flight: Optional[str] = None
    r: Optional[str] = None
    t: Optional[str] = None

    # Altitude
    alt_baro: Optional[int] = None
    on_ground: bool = False
    alt_geom: Optional[int] = None

    # Kinematics
    gs: Optional[float] = None
    track: Optional[float] = None
    baro_rate: Optional[int] = None
    geom_rate: Optional[int] = None
    true_heading: Optional[float] = None
    mag_heading: Optional[float] = None  # Add this line

    # Transponder / status
    squawk: Optional[str] = None
    emergency: Optional[str] = None
    category: Optional[str] = None

    # Navigation
    nav_qnh: Optional[float] = None
    nav_altitude_mcp: Optional[int] = None
    nav_heading: Optional[float] = None
    nav_modes: Optional[List[str]] = None

    # Position
    lat: Optional[float] = None
    lon: Optional[float] = None

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
    '', 'adsb.lol', 'adsbexchange', 'airplanes.live'
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
    source: str,
    flags: int = 0,
    noRegData: bool = False,
    ownOp: str = "",
    year: int = 0,
    desc: str = "",
) -> AdsbMessageRow:
    """
    Convert a reapi ADS-B message into a ClickHouse adsb_messages row.
    """

    return AdsbMessageRow(
        # Primary
        time=time,
        icao=msg.hex,

        # Metadata
        r=msg.r or "",
        t=msg.t or "",
        dbFlags=msg.dbFlags or 0,
        noRegData=noRegData,
        ownOp=ownOp,
        year=year,
        desc=desc,

        # Position / motion
        lat=msg.lat or 0.0,
        lon=msg.lon or 0.0,
        alt_baro=msg.alt_baro,
        on_ground=msg.on_ground,
        ground_speed=msg.gs,
        track_degrees=msg.track,
        flags=flags,
        vertical_rate=msg.baro_rate,
        source=source,
        geometric_altitude=msg.alt_geom,
        geometric_vertical_rate=msg.geom_rate,
        indicated_airspeed=None,
        roll_angle=None,

        # Aircraft integrity / accuracy
        aircraft_alert=None,
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
        aircraft_track=msg.track,
        aircraft_type=msg.t or "",
        aircraft_version=msg.version,

        # Aircraft identity
        aircraft_category=msg.category or "",
        aircraft_emergency=msg.emergency or "",
        aircraft_flight=msg.flight or "",
        aircraft_squawk=msg.squawk or "",

        # Navigation
        aircraft_baro_rate=msg.baro_rate,
        aircraft_nav_altitude_fms=None,
        aircraft_nav_altitude_mcp=msg.nav_altitude_mcp,
        aircraft_nav_modes=list(msg.nav_modes or []),
        aircraft_nav_qnh=msg.nav_qnh,
        aircraft_geom_rate=msg.geom_rate,
        aircraft_ias=None,
        aircraft_mach=None,
        aircraft_mag_heading=None,
        aircraft_oat=None,
        aircraft_roll=None,
        aircraft_tas=None,
        aircraft_tat=None,
        aircraft_true_heading=msg.true_heading,
        aircraft_wd=None,
        aircraft_ws=None,
        aircraft_track_rate=None,
        aircraft_nav_heading=msg.nav_heading,

        # Provenance
        data_source=source,
    )