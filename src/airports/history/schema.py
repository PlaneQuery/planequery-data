from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from data_engineering.utils import OUTPUT_DIR


OURAIRPORTS_REPOSITORY_URL = (
    "https://github.com/davidmegginson/ourairports-data.git"
)
BASELINE_DATE = date(2024, 1, 1)
SCHEMA_VERSION = 2

AIRPORT_VERSIONS_FILENAME = "airport_versions.parquet"
ALIAS_VERSIONS_FILENAME = "airport_alias_versions.parquet"
COMMITS_FILENAME = "commits.parquet"
METADATA_FILENAME = "metadata.json"

ALIAS_COLUMNS = (
    "ident",
    "icao_code",
    "iata_code",
    "gps_code",
    "local_code",
)

INTEGER_AIRPORT_COLUMNS = {"id", "elevation_ft"}
FLOAT_AIRPORT_COLUMNS = {"latitude_deg", "longitude_deg"}


@dataclass(frozen=True)
class HistoryPaths:
    history_dir: Path
    repository: Path

    @classmethod
    def defaults(cls) -> "HistoryPaths":
        ourairports_dir = OUTPUT_DIR / "data" / "raw" / "ourairports"
        return cls(
            history_dir=ourairports_dir / "history",
            repository=ourairports_dir / "repository",
        )

    @property
    def airport_versions(self) -> Path:
        return self.history_dir / AIRPORT_VERSIONS_FILENAME

    @property
    def alias_versions(self) -> Path:
        return self.history_dir / ALIAS_VERSIONS_FILENAME

    @property
    def commits(self) -> Path:
        return self.history_dir / COMMITS_FILENAME

    @property
    def metadata(self) -> Path:
        return self.history_dir / METADATA_FILENAME

    @property
    def required_files(self) -> tuple[Path, ...]:
        return (
            self.airport_versions,
            self.alias_versions,
            self.commits,
            self.metadata,
        )


@dataclass(frozen=True)
class Commit:
    sha: str
    committed_at: datetime
