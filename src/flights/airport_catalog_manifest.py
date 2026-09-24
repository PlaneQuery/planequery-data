import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from airports.models import AirportCatalogProvenance


@dataclass(frozen=True)
class AirportCatalogManifest:
    flight_date: date
    observed_commit_sha: str
    knowledge_cutoff_date: date
    knowledge_commit_sha: str
    catalog_available_date: date
    policy_version: str

    @property
    def catalog_cache_key(self) -> tuple[date, str, str, str]:
        """State that can change airport assignments for this flight day."""
        return (
            self.flight_date,
            self.observed_commit_sha,
            self.knowledge_commit_sha,
            self.policy_version,
        )

    @classmethod
    def from_provenance(
        cls,
        provenance: "AirportCatalogProvenance",
    ) -> "AirportCatalogManifest":
        return cls(
            flight_date=provenance.flight_date,
            observed_commit_sha=provenance.observed_commit_sha,
            knowledge_cutoff_date=provenance.knowledge_cutoff_date,
            knowledge_commit_sha=provenance.knowledge_commit_sha,
            catalog_available_date=provenance.catalog_available_date,
            policy_version=provenance.policy_version,
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
    def read(cls, path: Path) -> "AirportCatalogManifest":
        values = json.loads(path.read_text(encoding="utf-8"))
        for field in (
            "flight_date",
            "knowledge_cutoff_date",
            "catalog_available_date",
        ):
            values[field] = date.fromisoformat(values[field])
        return cls(**values)


def airport_catalog_manifest_path(flights_path: Path) -> Path:
    return flights_path.with_name("airport_catalog.json")
