import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from airports.models import AirportCatalogProvenance
    from data_engineering.adsb.canonical import ADSBSource


@dataclass(frozen=True)
class FlightManifest:
    flight_date: date
    observed_commit_sha: str
    knowledge_cutoff_date: date
    knowledge_commit_sha: str
    catalog_available_date: date
    policy_version: str
    adsb_source: str | None = None
    adsb_parquet_version: int | None = None
    adsb_input_format: str | None = None
    adsb_input_version: int | None = None

    @property
    def catalog_cache_key(self) -> tuple[date, str, str, str]:
        """State that can change airport assignments for this flight day."""
        return (
            self.flight_date,
            self.observed_commit_sha,
            self.knowledge_commit_sha,
            self.policy_version,
        )

    @property
    def flight_cache_key(
        self,
    ) -> tuple[date, str, str, str, str | None, int | None, str | None, int | None]:
        """Input state that can change the generated flights for a day."""
        return (
            *self.catalog_cache_key,
            self.adsb_source,
            self.adsb_parquet_version,
            self.adsb_input_format,
            self.adsb_input_version,
        )

    @classmethod
    def from_provenance(
        cls,
        provenance: "AirportCatalogProvenance",
        *,
        adsb_source: "ADSBSource | None" = None,
    ) -> "FlightManifest":
        source_name = None
        adsb_parquet_version = None
        if adsb_source is not None:
            from data_engineering.adsb.canonical import (
                ADSB_PARQUET_VERSIONS,
                coerce_adsb_source,
            )

            adsb_source = coerce_adsb_source(adsb_source)
            source_name = adsb_source.value
            adsb_parquet_version = ADSB_PARQUET_VERSIONS[adsb_source]
        return cls(
            flight_date=provenance.flight_date,
            observed_commit_sha=provenance.observed_commit_sha,
            knowledge_cutoff_date=provenance.knowledge_cutoff_date,
            knowledge_commit_sha=provenance.knowledge_commit_sha,
            catalog_available_date=provenance.catalog_available_date,
            policy_version=provenance.policy_version,
            adsb_source=source_name,
            adsb_parquet_version=adsb_parquet_version,
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        values = asdict(self)
        for field in (
            "flight_date",
            "knowledge_cutoff_date",
            "catalog_available_date",
        ):
            values[field] = values[field].isoformat()
        path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "FlightManifest":
        values = json.loads(path.read_text(encoding="utf-8"))
        for field in (
            "flight_date",
            "knowledge_cutoff_date",
            "catalog_available_date",
        ):
            values[field] = date.fromisoformat(values[field])
        return cls(**values)


def flight_manifest_path(flights_path: Path) -> Path:
    return flights_path.with_name("flight_manifest.json")
