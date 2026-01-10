from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

@dataclass
class Adsb_Message:
    time: datetime                      # Timestamp of the ADS-B message
    icao24: str 
    lat: float
    lon: float
    alt_baro: Optional[int] = None      # Barometric altitude # -1000 ft is "ground"
    baro_rate: Optional[int] = None   # Climb/descent rate
    gs: Optional[float] = None          # Ground speed
    track: Optional[float] = None       # Heading in degrees
    alt_geom: Optional[int] = None       # Geometric altitude
    geom_rate: Optional[int] = None    # Geometric rate 
    squawk: Optional[str] = None        # Squawk code
    dbFlags: Optional[int] = None # DB flags
    category: Optional[str] = None # Category of the aircraft
    temp_icao24: Optional[str] = None    # New optional field for temporary icao24
    time_ms = 0
    leg_marker = 0
    leg_reason: Optional[str] = None
    on_ground: bool = False # TODO: Implement this (and on_ground logic properly)

from dataclasses import dataclass
from typing import Optional, List


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
    def from_json(d: dict) -> "Adsb_Message_reapi":
        alt_baro_val = d.get("alt_baro")
        on_ground = alt_baro_val == "ground"

        return Adsb_Message_reapi(
            **{k: v for k, v in d.items() if k not in {"alt_baro"}},
            alt_baro=None if on_ground else alt_baro_val,
            on_ground=on_ground,
        )
