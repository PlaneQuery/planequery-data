"""Immutable airport catalog models."""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


EARTH_RADIUS_KM = 6371.0
AIRPORT_CODE_COLUMNS = (
    "ident",
    "local_code",
    "icao_code",
    "gps_code",
    "iata_code",
)
DEFAULT_AIRPORT_CODE_PRIORITY = (
    "local_code",
    "ident",
    "icao_code",
    "gps_code",
    "iata_code",
)
US_FAA_AIRSPACE_ISO_COUNTRIES = {"US", "PR", "GU", "VI", "AS", "MP", "UM"}
US_FAA_ISLAND_AIRSPACE_ISO_COUNTRIES = {"PR", "GU", "VI", "AS", "MP", "UM"}
US_FAA_ISLAND_AIRSPACE_ISO_REGIONS = {"US-HI", "US-AK"}


class AirportType(StrEnum):
    SMALL_AIRPORT = "small_airport"
    MEDIUM_AIRPORT = "medium_airport"
    LARGE_AIRPORT = "large_airport"
    SEAPLANE_BASE = "seaplane_base"
    HELIPORT = "heliport"
    CLOSED = "closed"
    BALLOONPORT = "balloonport"


FIXED_WING_AIRPORT_TYPES = frozenset(
    {
        AirportType.SMALL_AIRPORT,
        AirportType.MEDIUM_AIRPORT,
        AirportType.LARGE_AIRPORT,
    }
)


class AirportMatchProvenance(StrEnum):
    KNOWN_ON_FLIGHT_DATE = "known_on_flight_date"
    LEARNED_DURING_GRACE_PERIOD = "learned_during_grace_period"
    ACTIVE_ON_FLIGHT_DATE_CLOSED_BY_CUTOFF = (
        "active_on_flight_date_closed_by_cutoff"
    )


@dataclass(frozen=True)
class Airport:
    """One airport state, with equality based on its stable OurAirports ID."""

    id: int
    ident: str = field(compare=False)
    type: AirportType = field(compare=False)
    name: str = field(compare=False)
    lat: float = field(compare=False)
    lon: float = field(compare=False)
    elevation_ft: int | None = field(default=None, compare=False)
    continent: str | None = field(default=None, compare=False)
    iso_country: str | None = field(default=None, compare=False)
    iso_region: str | None = field(default=None, compare=False)
    municipality: str | None = field(default=None, compare=False)
    scheduled_service: str | None = field(default=None, compare=False)
    icao_code: str | None = field(default=None, compare=False)
    iata_code: str | None = field(default=None, compare=False)
    gps_code: str | None = field(default=None, compare=False)
    local_code: str | None = field(default=None, compare=False)
    home_link: str | None = field(default=None, compare=False)
    wikipedia_link: str | None = field(default=None, compare=False)
    keywords: str | None = field(default=None, compare=False)

    @property
    def is_in_us_faa_airspace(self) -> bool:
        return self.iso_country in US_FAA_AIRSPACE_ISO_COUNTRIES

    @property
    def is_in_us_faa_island_airspace(self) -> bool:
        return (
            self.iso_country in US_FAA_AIRSPACE_ISO_COUNTRIES
            and (
                self.iso_country in US_FAA_ISLAND_AIRSPACE_ISO_COUNTRIES
                or self.iso_region in US_FAA_ISLAND_AIRSPACE_ISO_REGIONS
            )
        )

    def distance_to(self, lat: float, lon: float) -> float:
        phi_1 = math.radians(self.lat)
        phi_2 = math.radians(lat)
        delta_phi = math.radians(lat - self.lat)
        delta_lambda = math.radians(lon - self.lon)
        haversine_value = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi_1)
            * math.cos(phi_2)
            * math.sin(delta_lambda / 2.0) ** 2
        )
        return EARTH_RADIUS_KM * 2 * math.atan2(
            math.sqrt(haversine_value),
            math.sqrt(1 - haversine_value),
        )


@dataclass(frozen=True)
class AirportMatch:
    id: int
    ident: str
    distance_km: float
    provenance: AirportMatchProvenance
    source_commit_sha: str
    airport: Airport = field(compare=False)

@dataclass(frozen=True)
class AirportCatalogProvenance:
    flight_date: date
    observed_commit_sha: str
    knowledge_cutoff_date: date
    knowledge_commit_sha: str
    catalog_available_date: date
    policy_version: str

    @property
    def knowledge_complete(self) -> bool:
        year = self.flight_date.year + (self.flight_date.month == 12)
        month = 1 if self.flight_date.month == 12 else self.flight_date.month + 1
        day = min(self.flight_date.day, calendar.monthrange(year, month)[1])
        return self.knowledge_cutoff_date == self.flight_date.replace(
            year=year,
            month=month,
            day=day,
        )
