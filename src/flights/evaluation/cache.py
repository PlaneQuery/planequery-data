"""Persistent, provenance-aware cache for daily flight evaluations.

The cache deliberately lives below a function-derived root instead of a
module-level path.  This keeps tests (and callers which replace ``OUTPUT_DIR``)
from retaining the value that happened to be present when this module was
imported.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Self
from urllib.parse import quote

import polars as pl

from data_engineering.adsb.canonical import ADSB_PARQUET_VERSIONS, ADSBSource
from flights.evaluation.models import (
    EVALUATION_STATS_SCHEMA,
    EvaluationResult,
    EvaluationStats,
    evaluation_stats_frame,
)


EVALUATION_CACHE_VERSION = 3
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
    """Return the cache root using the current ``OUTPUT_DIR`` value."""

    # Import the module, rather than its constant, so monkeypatching OUTPUT_DIR
    # after this module is imported continues to work.
    from data_engineering import utils as data_engineering_utils

    return (
        Path(data_engineering_utils.OUTPUT_DIR)
        / "data"
        / "cache"
        / "evaluation"
    )


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _normalized_path(value: PurePath) -> dict[str, Any]:
    path = Path(value).expanduser()
    try:
        normalized_path = path.resolve(strict=False)
    except OSError:
        normalized_path = path.absolute()

    normalized: dict[str, Any] = {
        "type": "path",
        "path": str(normalized_path),
    }
    try:
        stat = normalized_path.stat()
    except FileNotFoundError:
        normalized["exists"] = False
    except OSError as error:
        # The path still participates in the fingerprint.  Avoid repr(error),
        # which can contain platform-specific or otherwise unstable details.
        normalized.update(
            exists=None,
            stat_error=type(error).__name__,
        )
    else:
        normalized.update(
            exists=True,
            stat={
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
        )
    return normalized


def _normalize(value: Any) -> Any:
    """Convert configuration values to deterministic, JSON-safe values."""

    if isinstance(value, PurePath):
        return _normalized_path(value)
    if isinstance(value, datetime):
        return {
            "type": "datetime",
            "value": value.isoformat(),
        }
    if isinstance(value, date):
        return {
            "type": "date",
            "value": value.isoformat(),
        }
    if isinstance(value, Enum):
        return _normalize(value.value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "float", "value": "nan"}
        if math.isinf(value):
            return {
                "type": "float",
                "value": "infinity" if value > 0 else "-infinity",
            }
        return value
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": _qualified_type_name(value),
            "fields": {
                item.name: _normalize(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        normalized_items = [
            (_canonical_json(_normalize(key)), _normalize(item))
            for key, item in value.items()
        ]
        return {
            "type": "mapping",
            "items": sorted(normalized_items, key=lambda item: item[0]),
        }
    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalize(item) for item in value]
        return {
            "type": "set",
            "items": sorted(normalized_items, key=_canonical_json),
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [_normalize(item) for item in value],
        }

    # Numpy scalar types and similar values generally expose ``item``.  This
    # keeps a source's simple numeric configuration useful without importing
    # any optional numeric dependency here.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            pass
        else:
            if scalar is not value:
                return _normalize(scalar)

    # Known flight sources are dataclasses, so this is a conservative fallback
    # for a custom immutable value in one of their fields.  The class is always
    # retained and repr gives custom values a chance to provide stable state.
    return {
        "type": _qualified_type_name(value),
        "repr": repr(value),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _source_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Flight source versions must be integers")
    return value


@dataclass(frozen=True)
class SourceFingerprint:
    """Stable identity and normalized configuration for one flight source."""

    name: str
    version: int
    config_fingerprint: str
    config_json: str

    @classmethod
    def from_source(cls, source: object) -> Self:
        name = getattr(source, "name", None)
        version = getattr(source, "version", None)
        if not isinstance(name, str) or not name:
            raise TypeError("A cached flight source must have a non-empty name")
        if version is None:
            raise TypeError(f"Cached flight source {name!r} must have a version")

        if is_dataclass(source) and not isinstance(source, type):
            config = _normalize(source)
        else:
            # FlightSource implementations are expected to be dataclasses, but
            # preserving the concrete type and public attributes makes custom
            # test/source implementations deterministic too.
            public_state = {
                key: item
                for key, item in vars(source).items()
                if not key.startswith("_")
            }
            config = {
                "type": _qualified_type_name(source),
                "fields": _normalize(public_state),
            }
        config_json = _canonical_json(config)
        return cls(
            name=name,
            version=_source_version(version),
            config_fingerprint=hashlib.sha256(
                config_json.encode("utf-8")
            ).hexdigest(),
            config_json=config_json,
        )

    @property
    def config(self) -> Any:
        return json.loads(self.config_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "config_fingerprint": self.config_fingerprint,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        config = value["config"]
        config_json = _canonical_json(config)
        fingerprint = str(value["config_fingerprint"])
        if _fingerprint(config) != fingerprint:
            raise ValueError("Source configuration fingerprint does not match")
        return cls(
            name=str(value["name"]),
            version=_source_version(value["version"]),
            config_fingerprint=fingerprint,
            config_json=config_json,
        )


def _provenance_value(provenance: object, field_name: str) -> Any:
    if isinstance(provenance, Mapping):
        return provenance.get(field_name)
    return getattr(provenance, field_name, None)


@dataclass(frozen=True)
class AirportCatalogFingerprint:
    """Normalized airport-catalog provenance used by an evaluation."""

    fingerprint: str
    provenance_json: str

    @classmethod
    def from_catalog(
        cls,
        airport_catalog: object,
        test_date: date,
    ) -> Self | None:
        provenance = getattr(airport_catalog, "provenance", None)
        if provenance is None:
            return None

        raw = {
            field_name: _provenance_value(provenance, field_name)
            for field_name in _PROVENANCE_FIELDS
        }
        if raw["flight_date"] is None:
            raw["flight_date"] = test_date
        try:
            provenance_date = _coerce_date(raw["flight_date"])
        except (TypeError, ValueError):
            return None
        if provenance_date != test_date:
            return None
        if any(
            not isinstance(raw[field_name], str) or not raw[field_name]
            for field_name in (
                "observed_commit_sha",
                "knowledge_commit_sha",
                "policy_version",
            )
        ):
            return None

        normalized = {
            field_name: _normalize(raw[field_name])
            for field_name in _PROVENANCE_FIELDS
        }
        provenance_json = _canonical_json(normalized)
        return cls(
            fingerprint=hashlib.sha256(
                provenance_json.encode("utf-8")
            ).hexdigest(),
            provenance_json=provenance_json,
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self.provenance_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        provenance = value["provenance"]
        provenance_json = _canonical_json(provenance)
        fingerprint = str(value["fingerprint"])
        if _fingerprint(provenance) != fingerprint:
            raise ValueError("Airport catalog fingerprint does not match")
        return cls(
            fingerprint=fingerprint,
            provenance_json=provenance_json,
        )


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
        test_date: date | datetime | str,
        test_source: object,
        gold_source: object,
        adsb_source: ADSBSource | str | None,
        airport_catalog: object,
        *,
        cache_version: int = EVALUATION_CACHE_VERSION,
    ) -> Self | None:
        """Create a key, or disable caching when provenance is unavailable."""

        day = _coerce_date(test_date)
        airport_fingerprint = AirportCatalogFingerprint.from_catalog(
            airport_catalog,
            day,
        )
        if airport_fingerprint is None:
            return None

        # Import locally to keep module initialization independent of the
        # flight-source module and, in particular, evaluation.py.
        from flights.flight_source import ADSBFlightSource

        source_uses_adsb = isinstance(
            test_source,
            ADSBFlightSource,
        ) or isinstance(gold_source, ADSBFlightSource)
        normalized_adsb_source: str | None = None
        adsb_parquet_version: int | None = None
        if source_uses_adsb and adsb_source is not None:
            source = ADSBSource(adsb_source)
            normalized_adsb_source = source.value
            adsb_parquet_version = ADSB_PARQUET_VERSIONS[source]

        return cls(
            test_date=day,
            test_source=SourceFingerprint.from_source(test_source),
            gold_source=SourceFingerprint.from_source(gold_source),
            adsb_source=normalized_adsb_source,
            adsb_parquet_version=adsb_parquet_version,
            airport_catalog=airport_fingerprint,
            cache_version=cache_version,
        )

    @property
    def test_source_name(self) -> str:
        return self.test_source.name

    @property
    def test_source_version(self) -> int:
        return self.test_source.version

    @property
    def gold_source_name(self) -> str:
        return self.gold_source.name

    @property
    def gold_source_version(self) -> int:
        return self.gold_source.version

    @property
    def airport_catalog_fingerprint(self) -> str:
        return self.airport_catalog.fingerprint

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
            test_date=_coerce_date(value["test_date"]),
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
    text = "none" if value is None else str(value)
    # Escape slashes, equals signs, and other characters which could alter the
    # Hive directory structure while retaining readable source names.
    return quote(text, safe="-._")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.write_parquet(temporary_path)
        with temporary_path.open("rb") as parquet_file:
            os.fsync(parquet_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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


def _as_filter_items(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, date, datetime, Enum)):
        return [value]
    if hasattr(value, "name") and not isinstance(value, Sequence):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _source_filter(value: Any) -> set[str] | None:
    items = _as_filter_items(value)
    if items is None:
        return None
    return {
        str(getattr(item, "name", item))
        for item in items
    }


def _adsb_filter(value: Any) -> set[str | None] | None:
    items = _as_filter_items(value)
    if items is None:
        return None
    values: set[str | None] = set()
    for item in items:
        if item is None or (isinstance(item, str) and item.lower() == "none"):
            values.add(None)
        else:
            values.add(ADSBSource(item).value)
    return values


def _date_filter(value: Any) -> set[date] | None:
    items = _as_filter_items(value)
    if items is None:
        return None
    return {_coerce_date(item) for item in items}


class EvaluationCache:
    """Read, write, and discover provenance-keyed daily evaluations."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        cache_version: int = EVALUATION_CACHE_VERSION,
    ) -> None:
        self.root = (
            default_evaluation_cache_dir() if root is None else Path(root)
        )
        self.cache_version = cache_version

    def make_key(
        self,
        test_date: date | datetime | str,
        test_source: object,
        gold_source: object,
        adsb_source: ADSBSource | str | None,
        airport_catalog: object,
    ) -> EvaluationCacheKey | None:
        return EvaluationCacheKey.from_inputs(
            test_date,
            test_source,
            gold_source,
            adsb_source,
            airport_catalog,
            cache_version=self.cache_version,
        )

    def path_for(self, key: EvaluationCacheKey) -> Path:
        return (
            self.root
            / f"cache_version={_partition_value(key.cache_version)}"
            / f"test_source={_partition_value(key.test_source.name)}"
            / f"test_source_version={_partition_value(key.test_source.version)}"
            / (
                "test_source_config_fingerprint="
                f"{_partition_value(key.test_source.config_fingerprint)}"
            )
            / f"gold_source={_partition_value(key.gold_source.name)}"
            / f"gold_source_version={_partition_value(key.gold_source.version)}"
            / (
                "gold_source_config_fingerprint="
                f"{_partition_value(key.gold_source.config_fingerprint)}"
            )
            / f"adsb_source={_partition_value(key.adsb_source)}"
            / (
                "adsb_parquet_version="
                f"{_partition_value(key.adsb_parquet_version)}"
            )
            / (
                "airport_catalog_fingerprint="
                f"{_partition_value(key.airport_catalog.fingerprint)}"
            )
            / f"test_date={key.test_date.isoformat()}"
        )

    def _read_manifest(
        self,
        path: Path,
    ) -> tuple[EvaluationCacheKey, datetime]:
        value = json.loads(
            (path / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        if not isinstance(value, Mapping):
            raise ValueError("Evaluation cache manifest must be a JSON object")
        created_at = datetime.fromisoformat(str(value["created_at"]))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
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
        key: EvaluationCacheKey | None,
    ) -> EvaluationResult | None:
        """Return a cache hit, or ``None`` when no readable entry exists."""
        if key is None:
            return None
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
        key: EvaluationCacheKey | None,
        result: pl.DataFrame,
        stats: EvaluationStats,
        *,
        overwrite: bool = False,
    ) -> Path | None:
        """Atomically commit one result/stats pair and return its directory."""

        if key is None:
            return None
        if not _stats_match_key(stats, key):
            raise ValueError("EvaluationStats metadata does not match cache key")
        path = self.path_for(key)

        if not overwrite and self.load(key) is not None:
            return path

        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / MANIFEST_FILENAME
        manifest_path.unlink(missing_ok=True)
        _atomic_write_parquet(result, path / RESULT_FILENAME)
        _atomic_write_parquet(
            evaluation_stats_frame([stats]),
            path / STATS_FILENAME,
        )
        _atomic_write_bytes(
            manifest_path,
            (
                json.dumps(
                    {
                        "key": key.to_dict(),
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        return path

    def find_stats(
        self,
        *,
        test_source: object | Sequence[object] | None = None,
        gold_source: object | Sequence[object] | None = None,
        adsb_source: (
            ADSBSource | str | Sequence[ADSBSource | str | None] | None
        ) = None,
        test_dates: (
            date | datetime | str | Sequence[date | datetime | str] | None
        ) = None,
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
                key.cache_version != self.cache_version
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
        test_source: object | Sequence[object] | None = None,
        gold_source: object | Sequence[object] | None = None,
        adsb_source: (
            ADSBSource | str | Sequence[ADSBSource | str | None] | None
        ) = None,
        test_dates: (
            date | datetime | str | Sequence[date | datetime | str] | None
        ) = None,
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

        listed = pl.concat(entries, how="diagonal_relaxed")
        sort_columns = [
            column
            for column in ("test_date", "test_src", "gold_src", "adsb_src")
            if column in listed.columns
        ]
        return listed.sort(sort_columns, nulls_last=True)


__all__ = [
    "AirportCatalogFingerprint",
    "CachedEvaluationStats",
    "EVALUATION_CACHE_VERSION",
    "EvaluationCache",
    "EvaluationCacheKey",
    "MANIFEST_FILENAME",
    "RESULT_FILENAME",
    "STATS_FILENAME",
    "SourceFingerprint",
    "default_evaluation_cache_dir",
]
