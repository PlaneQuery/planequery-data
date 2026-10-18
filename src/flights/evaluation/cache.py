"""Persistent, provenance-aware cache for daily flight evaluations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePath
from typing import Any, Self

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.adsb.canonical import ADSB_PARQUET_VERSIONS, ADSBSource
from data_engineering.utils import OUTPUT_DIR
from flights.evaluation.models import (
    EVALUATION_STATS_SCHEMA,
    EvaluationResult,
    EvaluationStats,
    evaluation_stats_frame,
)
from flights.flight_source import ADSBFlightSource, FlightSource


EVALUATION_CACHE_VERSION = 7
RESULT_FILENAME = "result.parquet"
STATS_FILENAME = "stats.parquet"
MANIFEST_FILENAME = "manifest.json"

_CACHE_DISCOVERY_SCHEMA = {
    "cache_version": pl.Int64,
    "test_source_version": pl.Int64,
    "test_source_config_fingerprint": pl.String,
    "gold_source_version": pl.Int64,
    "gold_source_config_fingerprint": pl.String,
    "adsb_parquet_version": pl.Int64,
    "airport_catalog_fingerprint": pl.String,
    "cache_path": pl.String,
    "cache_created_at": pl.String,
}
_PROVENANCE_FIELDS = (
    "flight_date",
    "observed_commit_sha",
    "knowledge_commit_sha",
    "policy_version",
)


def default_evaluation_cache_dir() -> Path:
    return OUTPUT_DIR / "data" / "cache" / "evaluation"


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _normalized_path(value: PurePath) -> dict[str, Any]:
    path = Path(value).expanduser().resolve()
    stat = path.stat()
    return {
        "type": "path",
        "path": str(path),
        "exists": True,
        "stat": {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
    }


def _normalize(value: Any) -> Any:
    """Convert configuration values to deterministic, JSON-safe values."""

    if isinstance(value, PurePath):
        return _normalized_path(value)
    if isinstance(value, date):
        return {
            "type": "date",
            "value": value.isoformat(),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": _qualified_type_name(value),
            "fields": {
                item.name: _normalize(getattr(value, item.name))
                for item in fields(value)
            },
        }
    raise TypeError(f"Unsupported flight source setting: {value!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceFingerprint:
    """Stable identity and normalized configuration for one flight source."""

    name: str
    version: int
    config_fingerprint: str

    @classmethod
    def from_source(cls, source: FlightSource) -> Self:
        return cls(
            name=source.name,
            version=source.version,
            config_fingerprint=_fingerprint(_normalize(source)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "config_fingerprint": self.config_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        return cls(
            name=str(value["name"]),
            version=int(value["version"]),
            config_fingerprint=str(value["config_fingerprint"]),
        )


@dataclass(frozen=True)
class AirportCatalogFingerprint:
    """Normalized airport-catalog provenance used by an evaluation."""

    fingerprint: str

    @classmethod
    def from_catalog(
        cls,
        airport_catalog: AirportCatalogView,
        test_date: date,
    ) -> Self:
        provenance = airport_catalog.provenance
        assert provenance.flight_date == test_date
        normalized = {
            field_name: _normalize(getattr(provenance, field_name))
            for field_name in _PROVENANCE_FIELDS
        }
        return cls(fingerprint=_fingerprint(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {"fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        return cls(fingerprint=str(value["fingerprint"]))


@dataclass(frozen=True)
class EvaluationCacheKey:
    """All input state capable of changing one daily evaluation."""

    test_date: date
    test_source: SourceFingerprint
    gold_source: SourceFingerprint
    adsb_source: str | None
    adsb_parquet_version: int | None
    airport_catalog: AirportCatalogFingerprint
    cache_version: int = EVALUATION_CACHE_VERSION

    @classmethod
    def from_inputs(
        cls,
        test_date: date,
        test_source: FlightSource,
        gold_source: FlightSource,
        adsb_source: ADSBSource | None,
        airport_catalog: AirportCatalogView,
    ) -> Self:
        airport_fingerprint = AirportCatalogFingerprint.from_catalog(
            airport_catalog,
            test_date,
        )

        source_uses_adsb = isinstance(
            test_source,
            ADSBFlightSource,
        ) or isinstance(gold_source, ADSBFlightSource)
        normalized_adsb_source: str | None = None
        adsb_parquet_version: int | None = None
        if source_uses_adsb and adsb_source is not None:
            normalized_adsb_source = adsb_source.value
            uses_canonical = any(
                isinstance(source, ADSBFlightSource)
                and source.flight_manifest(
                    airport_catalog.provenance, adsb_source
                ).adsb_parquet_version is not None
                for source in (test_source, gold_source)
            )
            if uses_canonical:
                adsb_parquet_version = ADSB_PARQUET_VERSIONS[adsb_source]

        return cls(
            test_date=test_date,
            test_source=SourceFingerprint.from_source(test_source),
            gold_source=SourceFingerprint.from_source(gold_source),
            adsb_source=normalized_adsb_source,
            adsb_parquet_version=adsb_parquet_version,
            airport_catalog=airport_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_version": self.cache_version,
            "test_date": self.test_date.isoformat(),
            "test_source": self.test_source.to_dict(),
            "gold_source": self.gold_source.to_dict(),
            "adsb_source": self.adsb_source,
            "adsb_parquet_version": self.adsb_parquet_version,
            "airport_catalog": self.airport_catalog.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        adsb_source = value.get("adsb_source")
        adsb_parquet_version = value.get("adsb_parquet_version")
        return cls(
            test_date=date.fromisoformat(str(value["test_date"])),
            test_source=SourceFingerprint.from_dict(value["test_source"]),
            gold_source=SourceFingerprint.from_dict(value["gold_source"]),
            adsb_source=None if adsb_source is None else str(adsb_source),
            adsb_parquet_version=(
                None
                if adsb_parquet_version is None
                else int(adsb_parquet_version)
            ),
            airport_catalog=AirportCatalogFingerprint.from_dict(
                value["airport_catalog"]
            ),
            cache_version=int(value["cache_version"]),
        )


@dataclass(frozen=True)
class CachedEvaluationStats:
    """Typed daily stats and provenance without the detailed comparison."""

    stats: EvaluationStats
    key: EvaluationCacheKey
    created_at: datetime
    path: Path


def _partition_value(value: object | None) -> str:
    return "none" if value is None else str(value)


def _stats_match_key(
    stats: EvaluationStats,
    key: EvaluationCacheKey,
) -> bool:
    adsb_source = (
        stats.adsb_source.value if stats.adsb_source is not None else None
    )
    return (
        stats.test_date == key.test_date
        and stats.test_source == key.test_source.name
        and stats.gold_source == key.gold_source.name
        and adsb_source == key.adsb_source
    )


SourceFilter = FlightSource | str


def _source_filter(
    value: SourceFilter | Sequence[SourceFilter] | None,
) -> set[str] | None:
    if value is None:
        return None
    items = (value,) if isinstance(value, (FlightSource, str)) else value
    return {item.name if isinstance(item, FlightSource) else item for item in items}


def _adsb_filter(
    value: ADSBSource | str | Sequence[ADSBSource | str] | None,
) -> set[str] | None:
    if value is None:
        return None
    items = (value,) if isinstance(value, (ADSBSource, str)) else value
    return {ADSBSource(item).value for item in items}


def _date_filter(
    value: date | Sequence[date] | None,
) -> set[date] | None:
    if value is None:
        return None
    return {value} if isinstance(value, date) else set(value)


class EvaluationCache:
    """Read, write, and discover provenance-keyed daily evaluations."""

    def __init__(
        self,
        root: Path | str | None = None,
    ) -> None:
        self.root = (
            default_evaluation_cache_dir() if root is None else Path(root)
        )

    def make_key(
        self,
        test_date: date,
        test_source: FlightSource,
        gold_source: FlightSource,
        adsb_source: ADSBSource | None,
        airport_catalog: AirportCatalogView,
    ) -> EvaluationCacheKey:
        return EvaluationCacheKey.from_inputs(
            test_date,
            test_source,
            gold_source,
            adsb_source,
            airport_catalog,
        )

    def path_for(self, key: EvaluationCacheKey) -> Path:
        return self.root.joinpath(
            f"cache_version={key.cache_version}",
            f"test_source={key.test_source.name}",
            f"test_source_version={key.test_source.version}",
            "test_source_config_fingerprint="
            f"{key.test_source.config_fingerprint}",
            f"gold_source={key.gold_source.name}",
            f"gold_source_version={key.gold_source.version}",
            "gold_source_config_fingerprint="
            f"{key.gold_source.config_fingerprint}",
            f"adsb_source={_partition_value(key.adsb_source)}",
            "adsb_parquet_version="
            f"{_partition_value(key.adsb_parquet_version)}",
            "airport_catalog_fingerprint="
            f"{key.airport_catalog.fingerprint}",
            f"test_date={key.test_date.isoformat()}",
        )

    def _read_manifest(
        self,
        path: Path,
    ) -> tuple[EvaluationCacheKey, datetime]:
        value = json.loads(
            (path / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        created_at = datetime.fromisoformat(str(value["created_at"]))
        return EvaluationCacheKey.from_dict(value["key"]), created_at

    def _read_stats(
        self,
        path: Path,
        key: EvaluationCacheKey,
    ) -> EvaluationStats:
        frame = pl.read_parquet(path / STATS_FILENAME)
        if frame.height != 1:
            raise ValueError("Expected one stats row")
        stats = EvaluationStats.from_row(frame.row(0, named=True))
        if not _stats_match_key(stats, key):
            raise ValueError("Stats do not match cache key")
        return stats

    def load(
        self,
        key: EvaluationCacheKey,
    ) -> EvaluationResult | None:
        """Return a cache hit, or ``None`` when no readable entry exists."""
        path = self.path_for(key)
        try:
            stored_key, _ = self._read_manifest(path)
            if stored_key != key:
                return None
            stats = self._read_stats(path, key)
            comparison = pl.read_parquet(path / RESULT_FILENAME)
        except Exception:
            return None
        return EvaluationResult(comparison=comparison, stats=stats)

    def store(
        self,
        key: EvaluationCacheKey,
        result: pl.DataFrame,
        stats: EvaluationStats,
    ) -> Path:
        """Write one result/stats pair and return its directory."""
        if not _stats_match_key(stats, key):
            raise ValueError("EvaluationStats metadata does not match cache key")
        path = self.path_for(key)
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / MANIFEST_FILENAME
        manifest_path.unlink(missing_ok=True)
        result.write_parquet(path / RESULT_FILENAME)
        evaluation_stats_frame([stats]).write_parquet(path / STATS_FILENAME)
        manifest_path.write_text(
            json.dumps(
                {
                    "key": key.to_dict(),
                    "created_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def find_stats(
        self,
        *,
        test_source: SourceFilter | Sequence[SourceFilter] | None = None,
        gold_source: SourceFilter | Sequence[SourceFilter] | None = None,
        adsb_source: ADSBSource | str | Sequence[ADSBSource | str] | None = None,
        test_dates: date | Sequence[date] | None = None,
    ) -> tuple[CachedEvaluationStats, ...]:
        """Return readable cached stats without loading comparison files."""

        test_sources = _source_filter(test_source)
        gold_sources = _source_filter(gold_source)
        adsb_sources = _adsb_filter(adsb_source)
        dates = _date_filter(test_dates)

        entries: list[CachedEvaluationStats] = []
        manifests = (
            sorted(self.root.rglob(MANIFEST_FILENAME))
            if self.root.is_dir()
            else []
        )
        for manifest_path in manifests:
            path = manifest_path.parent
            try:
                key, created_at = self._read_manifest(path)
            except Exception:
                continue
            if (
                key.cache_version != EVALUATION_CACHE_VERSION
                or self.path_for(key) != path
            ):
                continue
            if (
                test_sources is not None
                and key.test_source.name not in test_sources
            ):
                continue
            if (
                gold_sources is not None
                and key.gold_source.name not in gold_sources
            ):
                continue
            if adsb_sources is not None and key.adsb_source not in adsb_sources:
                continue
            if dates is not None and key.test_date not in dates:
                continue

            try:
                pl.read_parquet_schema(path / RESULT_FILENAME)
                stats = self._read_stats(path, key)
            except Exception:
                continue
            entries.append(
                CachedEvaluationStats(
                    stats=stats,
                    key=key,
                    created_at=created_at,
                    path=path,
                )
            )

        return tuple(
            sorted(
                entries,
                key=lambda entry: (
                    entry.stats.test_date,
                    entry.stats.test_source,
                    entry.stats.gold_source,
                    entry.stats.adsb_source.value
                    if entry.stats.adsb_source is not None
                    else "",
                    entry.created_at,
                    str(entry.path),
                ),
            )
        )

    def list_entries(
        self,
        *,
        test_source: SourceFilter | Sequence[SourceFilter] | None = None,
        gold_source: SourceFilter | Sequence[SourceFilter] | None = None,
        adsb_source: ADSBSource | str | Sequence[ADSBSource | str] | None = None,
        test_dates: date | Sequence[date] | None = None,
    ) -> pl.DataFrame:
        """Return a tabular view of matching cached stats and provenance."""

        cached_entries = self.find_stats(
            test_source=test_source,
            gold_source=gold_source,
            adsb_source=adsb_source,
            test_dates=test_dates,
        )
        entries: list[pl.DataFrame] = []
        for entry in cached_entries:
            stats = entry.stats
            key = entry.key
            entries.append(
                evaluation_stats_frame([stats]).with_columns(
                    pl.lit(key.cache_version).alias("cache_version"),
                    pl.lit(key.test_source.version).alias(
                        "test_source_version"
                    ),
                    pl.lit(key.test_source.config_fingerprint).alias(
                        "test_source_config_fingerprint"
                    ),
                    pl.lit(key.gold_source.version).alias(
                        "gold_source_version"
                    ),
                    pl.lit(key.gold_source.config_fingerprint).alias(
                        "gold_source_config_fingerprint"
                    ),
                    pl.lit(key.adsb_parquet_version).alias(
                        "adsb_parquet_version"
                    ),
                    pl.lit(key.airport_catalog.fingerprint).alias(
                        "airport_catalog_fingerprint"
                    ),
                    pl.lit(str(entry.path)).alias("cache_path"),
                    pl.lit(entry.created_at.isoformat()).alias(
                        "cache_created_at"
                    ),
                )
            )

        if not entries:
            return pl.DataFrame(
                schema={
                    **EVALUATION_STATS_SCHEMA,
                    **_CACHE_DISCOVERY_SCHEMA,
                }
            )

        return pl.concat(entries, how="diagonal_relaxed").sort(
            ["test_date", "test_src", "gold_src", "adsb_src"],
            nulls_last=True,
        )
