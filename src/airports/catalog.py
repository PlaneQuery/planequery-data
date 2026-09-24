"""Flight-date views over the derived OurAirports history."""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Protocol, Self

import polars as pl

from airports.matcher import AirportMatcher
from airports.models import (
    AIRPORT_CODE_COLUMNS,
    DEFAULT_AIRPORT_CODE_PRIORITY,
    FIXED_WING_AIRPORT_TYPES,
    Airport,
    AirportCatalogProvenance,
    AirportMatchProvenance,
    AirportType,
)


AIRPORT_POLICY_VERSION = "2"


class AirportHistorySource(Protocol):
    def as_of(self, observed_on: date) -> pl.DataFrame: ...

    def source_commit_sha(self, observed_on: date) -> str: ...

    def aliases_through(
        self,
        start_date: date,
        cutoff_date: date,
    ) -> pl.DataFrame: ...


def add_calendar_month(day: date) -> date:
    year = day.year + (day.month == 12)
    month = 1 if day.month == 12 else day.month + 1
    return day.replace(
        year=year,
        month=month,
        day=min(day.day, calendar.monthrange(year, month)[1]),
    )


def _airport_from_row(row: dict[str, object], id_column: str) -> Airport:
    return Airport(
        id=int(row[id_column]),
        ident=str(row["ident"]),
        type=AirportType(str(row["type"])),
        name=str(row["name"]),
        lat=float(row["latitude_deg"]),
        lon=float(row["longitude_deg"]),
        elevation_ft=row["elevation_ft"],
        continent=row["continent"],
        iso_country=row["iso_country"],
        iso_region=row["iso_region"],
        municipality=row["municipality"],
        scheduled_service=row["scheduled_service"],
        icao_code=row["icao_code"],
        iata_code=row["iata_code"],
        gps_code=row["gps_code"],
        local_code=row["local_code"],
        home_link=row["home_link"],
        wikipedia_link=row["wikipedia_link"],
        keywords=row["keywords"],
    )


def _read_state(
    frame: pl.DataFrame,
) -> tuple[dict[int, Airport], dict[int, str]]:
    id_column = "airport_id" if "airport_id" in frame.columns else "id"
    airports: dict[int, Airport] = {}
    commits: dict[int, str] = {}
    for row in frame.iter_rows(named=True):
        airport = _airport_from_row(row, id_column)
        airports[airport.id] = airport
        commits[airport.id] = str(row["source_commit_sha"])
    return airports, commits


@dataclass(frozen=True)
class AirportCatalogView:
    provenance: AirportCatalogProvenance
    airports: tuple[Airport, ...]
    id_index: Mapping[int, Airport]
    ident_index: Mapping[str, Airport]
    code_index: Mapping[str, tuple[Airport, ...]]
    code_type_index: Mapping[str, Mapping[str, tuple[Airport, ...]]]
    airport_provenance: Mapping[int, AirportMatchProvenance]
    source_commit_by_airport_id: Mapping[int, str]
    _spatial_airport_ids: tuple[int, ...] = field(repr=False)
    _matcher: AirportMatcher = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_matcher",
            AirportMatcher(
                tuple(self.id_index[airport_id] for airport_id in self._spatial_airport_ids),
                self.airport_provenance,
                self.source_commit_by_airport_id,
            ),
        )

    @property
    def matcher(self) -> AirportMatcher:
        return self._matcher

    @classmethod
    def from_history(
        cls,
        history: AirportHistorySource,
        flight_date: date,
        catalog_available_date: date,
    ) -> Self:
        knowledge_cutoff = min(add_calendar_month(flight_date), catalog_available_date)
        observed_frame = history.as_of(flight_date)
        knowledge_frame = history.as_of(knowledge_cutoff)
        alias_frame = history.aliases_through(flight_date, knowledge_cutoff)
        observed, observed_commits = _read_state(observed_frame)
        knowledge, knowledge_commits = _read_state(knowledge_frame)

        effective: dict[int, Airport] = {}
        match_provenance: dict[int, AirportMatchProvenance] = {}
        source_commits: dict[int, str] = {}
        spatial_ids: list[int] = []

        for airport_id in sorted(observed.keys() | knowledge.keys()):
            if airport_id in observed:
                airport = observed[airport_id]
                cutoff_airport = knowledge.get(airport_id)
                provenance = AirportMatchProvenance.KNOWN_ON_FLIGHT_DATE
                if (
                    airport.type is not AirportType.CLOSED
                    and cutoff_airport is not None
                    and cutoff_airport.type is AirportType.CLOSED
                ):
                    provenance = (
                        AirportMatchProvenance.ACTIVE_ON_FLIGHT_DATE_CLOSED_BY_CUTOFF
                    )
                source_commit_sha = observed_commits[airport_id]
                spatial_type = airport.type
            else:
                airport = knowledge[airport_id]
                provenance = AirportMatchProvenance.LEARNED_DURING_GRACE_PERIOD
                source_commit_sha = knowledge_commits[airport_id]
                spatial_type = airport.type

            effective[airport_id] = airport
            match_provenance[airport_id] = provenance
            source_commits[airport_id] = source_commit_sha
            if spatial_type in FIXED_WING_AIRPORT_TYPES:
                spatial_ids.append(airport_id)

        # Codes valid on the flight day take precedence over later
        # reassignments. If a typed code was not known on the flight day, use
        # its earliest ownership observed during the grace period.
        typed_airport_ids: dict[str, dict[str, set[int]]] = {
            column: {} for column in AIRPORT_CODE_COLUMNS
        }
        for airport_id, airport in observed.items():
            for column in AIRPORT_CODE_COLUMNS:
                code = getattr(airport, column)
                if code is not None:
                    typed_airport_ids[column].setdefault(code, set()).add(airport_id)

        observed_typed_codes = {
            column: set(codes) for column, codes in typed_airport_ids.items()
        }
        alias_first_seen: dict[tuple[str, str], object] = {}
        if "observed_at" in alias_frame.columns:
            alias_frame = alias_frame.sort("observed_at")
        for position, alias in enumerate(alias_frame.iter_rows(named=True)):
            airport_id = int(alias["airport_id"])
            if airport_id not in effective:
                continue
            code = str(alias["code"])
            code_type = str(alias["code_type"])
            first_seen = alias.get("observed_at", position)
            key = (code_type, code)
            if code not in observed_typed_codes[code_type]:
                if key not in alias_first_seen:
                    alias_first_seen[key] = first_seen
                if alias_first_seen[key] == first_seen:
                    typed_airport_ids[code_type].setdefault(code, set()).add(
                        airport_id
                    )

        knowledge_typed_ids: dict[str, dict[str, set[int]]] = {
            column: {} for column in AIRPORT_CODE_COLUMNS
        }
        for airport_id, airport in knowledge.items():
            for column in AIRPORT_CODE_COLUMNS:
                code = getattr(airport, column)
                if code is not None:
                    knowledge_typed_ids[column].setdefault(code, set()).add(
                        airport_id
                    )
        for column, codes in knowledge_typed_ids.items():
            for code, airport_ids in codes.items():
                typed_airport_ids[column].setdefault(code, airport_ids)

        code_airport_ids: dict[str, set[int]] = {}
        for codes in typed_airport_ids.values():
            for code, airport_ids in codes.items():
                code_airport_ids.setdefault(code, set()).update(airport_ids)

        code_index = {
            code: tuple(effective[airport_id] for airport_id in sorted(airport_ids))
            for code, airport_ids in code_airport_ids.items()
        }
        code_type_index = {
            column: MappingProxyType(
                {
                    code: tuple(
                        effective[airport_id]
                        for airport_id in sorted(airport_ids)
                    )
                    for code, airport_ids in codes.items()
                }
            )
            for column, codes in typed_airport_ids.items()
        }
        ident_index = {
            ident: effective[next(iter(airport_ids))]
            for ident, airport_ids in typed_airport_ids["ident"].items()
            if len(airport_ids) == 1
        }
        airports = tuple(effective[airport_id] for airport_id in sorted(effective))
        return cls(
            provenance=AirportCatalogProvenance(
                flight_date=flight_date,
                observed_commit_sha=history.source_commit_sha(flight_date),
                knowledge_cutoff_date=knowledge_cutoff,
                knowledge_commit_sha=history.source_commit_sha(knowledge_cutoff),
                catalog_available_date=catalog_available_date,
                policy_version=AIRPORT_POLICY_VERSION,
            ),
            airports=airports,
            id_index=MappingProxyType(effective),
            ident_index=MappingProxyType(ident_index),
            code_index=MappingProxyType(code_index),
            code_type_index=MappingProxyType(code_type_index),
            airport_provenance=MappingProxyType(match_provenance),
            source_commit_by_airport_id=MappingProxyType(source_commits),
            _spatial_airport_ids=tuple(spatial_ids),
        )

    def get_airport_by_id(self, airport_id: int) -> Airport | None:
        return self.id_index.get(airport_id)

    def get_airport_by_ident(self, ident: str) -> Airport | None:
        return self.ident_index.get(ident)

    def resolve_code(
        self,
        code: str,
        *,
        lat: float | None = None,
        lon: float | None = None,
        code_priority: Sequence[str] = DEFAULT_AIRPORT_CODE_PRIORITY,
        prioritize_faa_airspace: bool = True,
    ) -> Airport | None:
        if code_priority:
            for column in code_priority:
                candidates = set(self.code_type_index[column].get(code, ()))
                if candidates:
                    break
            else:
                return None
        else:
            candidates = set(self.code_index.get(code, ()))
            if not candidates:
                return None

        active_candidates = {
            airport
            for airport in candidates
            if airport.type is not AirportType.CLOSED
        }
        if active_candidates:
            candidates = active_candidates
        if len(candidates) == 1:
            return candidates.pop()

        if prioritize_faa_airspace:
            faa_candidates = {
                airport for airport in candidates if airport.is_in_us_faa_airspace
            }
            if faa_candidates:
                candidates = faa_candidates
        if len(candidates) == 1:
            return candidates.pop()

        if lat is not None:
            return min(
                candidates,
                key=lambda airport: (airport.distance_to(lat, lon), airport.ident),
            )

        idents = sorted(airport.ident for airport in candidates)
        raise ValueError(
            f"Multiple airports found for code {code!r} with no clear priority: {idents}"
        )
