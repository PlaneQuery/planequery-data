from __future__ import annotations

import csv
import io
import json
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from itertools import pairwise
from pathlib import Path

import polars as pl

from airports.history.schema import (
    ALIAS_COLUMNS,
    BASELINE_DATE,
    FLOAT_AIRPORT_COLUMNS,
    HistoryPaths,
    INTEGER_AIRPORT_COLUMNS,
    OURAIRPORTS_REPOSITORY_URL,
    SCHEMA_VERSION,
)


AirportRow = dict[str, object]
AirportRows = dict[int, AirportRow]
AliasKey = tuple[int, str]
CommitRecord = tuple[str, datetime]


def _git(
    repository: Path,
    *arguments: str,
    text: bool = True,
) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def ensure_repository(
    repository: Path,
    repository_url: str = OURAIRPORTS_REPOSITORY_URL,
) -> None:
    if repository.exists():
        return

    repository.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-tags", repository_url, str(repository)],
        check=True,
    )


def latest_airports_commit(repository: Path) -> str:
    return str(
        _git(
            repository,
            "log",
            "-1",
            "--format=%H",
            "--",
            "airports.csv",
        )
    ).strip()


def refresh_repository(repository: Path) -> None:
    _git(repository, "fetch", "origin", "HEAD")
    _git(repository, "merge", "--ff-only", "FETCH_HEAD")


def _parse_git_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _parse_commits(output: str | bytes) -> list[CommitRecord]:
    return [
        (sha, _parse_git_datetime(committed_at))
        for sha, committed_at in (
            line.split("\t", 1) for line in str(output).splitlines()
        )
    ]


def _assert_monotonic_commits(
    commits: list[CommitRecord],
    previous_committed_at: datetime | None = None,
) -> None:
    committed_at = [timestamp for _, timestamp in commits]
    if previous_committed_at is not None and committed_at:
        assert previous_committed_at <= committed_at[0], (
            "OurAirports airport commit timestamps are not monotonic"
        )
    assert all(left <= right for left, right in pairwise(committed_at)), (
        "OurAirports airport commit timestamps are not monotonic"
    )


def _airports_commits(repository: Path) -> list[CommitRecord]:
    commits = _parse_commits(
        _git(
            repository,
            "log",
            "--reverse",
            "--format=%H%x09%cI",
            "--",
            "airports.csv",
        )
    )
    _assert_monotonic_commits(commits)
    return commits


def _commits_from_baseline(
    repository: Path,
    baseline_date: date,
) -> list[CommitRecord]:
    commits = _airports_commits(repository)
    if not commits:
        raise ValueError("Airport repository contains no airports.csv history")
    baseline_start = datetime.combine(
        baseline_date,
        time.min,
        tzinfo=timezone.utc,
    )
    seed_index = max(
        (
            index
            for index, (_, committed_at) in enumerate(commits)
            if committed_at < baseline_start
        ),
        default=0,
    )
    return commits[seed_index:]


def _commits_after(
    repository: Path,
    commit_sha: str,
    committed_at: datetime,
) -> list[CommitRecord]:
    commits = _parse_commits(
        _git(
            repository,
            "log",
            "--reverse",
            "--format=%H%x09%cI",
            f"{commit_sha}..HEAD",
            "--",
            "airports.csv",
        )
    )
    _assert_monotonic_commits(commits, committed_at)
    return commits


def _read_airports_csv(
    repository: Path,
    commit_sha: str,
) -> tuple[list[str], AirportRows]:
    contents = _git(
        repository,
        "show",
        f"{commit_sha}:airports.csv",
        text=False,
    )
    reader = csv.DictReader(
        io.StringIO(bytes(contents).decode("utf-8-sig"), newline="")
    )
    columns = list(reader.fieldnames or ())
    rows: AirportRows = {}
    for source_row in reader:
        row = {
            column: _typed_airport_value(column, source_row.get(column))
            for column in columns
        }
        rows[int(row["id"])] = row
    return columns, rows


def _typed_airport_value(column: str, value: object) -> object:
    if value is None or value == "":
        return None
    if column in INTEGER_AIRPORT_COLUMNS:
        return int(value)
    if column in FLOAT_AIRPORT_COLUMNS:
        return float(value)
    return value if isinstance(value, str) else str(value)


def _union_columns(
    airport_columns: list[str],
    snapshot_columns: list[str],
) -> list[str]:
    return [
        *airport_columns,
        *(column for column in snapshot_columns if column not in airport_columns),
    ]


def _normalize_airports(
    airports: AirportRows,
    airport_columns: list[str],
) -> AirportRows:
    return {
        airport_id: {
            column: _typed_airport_value(column, row.get(column))
            for column in airport_columns
        }
        for airport_id, row in airports.items()
    }


def _airport_version(
    row: AirportRow,
    airport_columns: list[str],
    observed_at: datetime,
    source_commit_sha: str,
) -> dict[str, object]:
    return {
        "observed_at": observed_at,
        "observed_until": None,
        "present": True,
        "source_commit_sha": source_commit_sha,
        **{column: row.get(column) for column in airport_columns},
    }


def _alias_state(airports: AirportRows) -> dict[AliasKey, str]:
    return {
        (airport_id, code_type): str(row[code_type])
        for airport_id, row in airports.items()
        for code_type in ALIAS_COLUMNS
        if row.get(code_type) is not None
    }


def _airport_schema(
    airport_columns: list[str],
) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "observed_at": pl.Datetime("us", "UTC"),
        "observed_until": pl.Datetime("us", "UTC"),
        "present": pl.Boolean,
        "source_commit_sha": pl.String,
    }
    for column in airport_columns:
        if column in INTEGER_AIRPORT_COLUMNS:
            schema[column] = pl.Int64
        elif column in FLOAT_AIRPORT_COLUMNS:
            schema[column] = pl.Float64
        else:
            schema[column] = pl.String
    return schema


def _alias_schema() -> dict[str, pl.DataType]:
    return {
        "airport_id": pl.Int64,
        "code": pl.String,
        "code_type": pl.String,
        "observed_at": pl.Datetime("us", "UTC"),
        "observed_until": pl.Datetime("us", "UTC"),
        "source_commit_sha": pl.String,
    }


def _commit_schema() -> dict[str, pl.DataType]:
    return {
        "commit_sha": pl.String,
        "committed_at": pl.Datetime("us", "UTC"),
        "airport_count": pl.Int64,
        "airport_change_count": pl.Int64,
        "alias_change_count": pl.Int64,
        "is_baseline": pl.Boolean,
        "state_applied": pl.Boolean,
    }


def _records_frame(
    records: list[dict[str, object]],
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    return pl.from_dicts(
        records,
        schema=schema,
        infer_schema_length=None,
    ).select(*schema)


def _align_airport_frame(
    frame: pl.DataFrame,
    airport_columns: list[str],
) -> pl.DataFrame:
    schema = _airport_schema(airport_columns)
    missing = [
        pl.lit(None, dtype=data_type).alias(column)
        for column, data_type in schema.items()
        if column not in frame.columns
    ]
    if missing:
        frame = frame.with_columns(*missing)
    return frame.select(*schema)


def _write_catalog(
    paths: HistoryPaths,
    airport_versions: pl.DataFrame,
    alias_versions: pl.DataFrame,
    commit_manifest: pl.DataFrame,
    airport_columns: list[str],
) -> None:
    paths.history_dir.mkdir(parents=True, exist_ok=True)
    _align_airport_frame(airport_versions, airport_columns).write_parquet(
        paths.airport_versions,
        compression="zstd",
        statistics=True,
    )
    alias_versions.select(*_alias_schema()).write_parquet(
        paths.alias_versions,
        compression="zstd",
        statistics=True,
    )
    commit_manifest.select(*_commit_schema()).write_parquet(
        paths.commits,
        compression="zstd",
        statistics=True,
    )


@dataclass
class _HistoryDelta:
    airport_versions: list[dict[str, object]]
    alias_versions: list[dict[str, object]]
    commit_manifest: list[dict[str, object]]
    airport_closures: dict[int, datetime]
    alias_closures: dict[AliasKey, datetime]
    airport_columns: list[str]
    latest_commit_sha: str
    latest_committed_at: datetime
    latest_state_commit_sha: str
    latest_state_committed_at: datetime


def _derive_delta(
    repository: Path,
    commits: list[CommitRecord],
    *,
    airport_columns: list[str],
    current_airports: AirportRows,
    existing_open_airport_ids: set[int],
    existing_open_alias_keys: set[AliasKey],
    latest_state_commit_sha: str | None = None,
    latest_state_committed_at: datetime | None = None,
    baseline_commit_sha: str | None = None,
) -> _HistoryDelta:
    airport_versions: list[dict[str, object]] = []
    alias_versions: list[dict[str, object]] = []
    commit_manifest: list[dict[str, object]] = []
    airport_closures: dict[int, datetime] = {}
    alias_closures: dict[AliasKey, datetime] = {}
    airport_version_indexes: dict[int, int] = {}
    alias_version_indexes: dict[AliasKey, int] = {}
    current_aliases = _alias_state(current_airports)

    for commit_index, (commit_sha, committed_at) in enumerate(commits):
        snapshot_columns, snapshot_airports = _read_airports_csv(
            repository,
            commit_sha,
        )
        if commit_index == 0 and commit_sha == baseline_commit_sha:
            assert snapshot_airports, (
                "The baseline airport snapshot has no data rows"
            )

        if not snapshot_airports:
            commit_manifest.append(
                {
                    "commit_sha": commit_sha,
                    "committed_at": committed_at,
                    "airport_count": 0,
                    "airport_change_count": 0,
                    "alias_change_count": 0,
                    "is_baseline": commit_sha == baseline_commit_sha,
                    "state_applied": False,
                }
            )
            continue

        next_airport_columns = _union_columns(
            airport_columns,
            snapshot_columns,
        )
        if next_airport_columns != airport_columns:
            current_airports = _normalize_airports(
                current_airports,
                next_airport_columns,
            )
            airport_columns = next_airport_columns
        if set(snapshot_columns) == set(airport_columns):
            next_airports = snapshot_airports
        else:
            next_airports = _normalize_airports(
                snapshot_airports,
                airport_columns,
            )
        changed_airports = {
            airport_id
            for airport_id in current_airports.keys() | next_airports.keys()
            if current_airports.get(airport_id) != next_airports.get(airport_id)
        }
        for airport_id in sorted(changed_airports):
            if airport_id in airport_version_indexes:
                airport_versions[airport_version_indexes.pop(airport_id)][
                    "observed_until"
                ] = committed_at
            elif airport_id in existing_open_airport_ids:
                airport_closures.setdefault(airport_id, committed_at)

            airport_version_indexes[airport_id] = len(airport_versions)
            if airport_id in next_airports:
                airport_versions.append(
                    _airport_version(
                        next_airports[airport_id],
                        airport_columns,
                        committed_at,
                        commit_sha,
                    )
                )
            else:
                airport_versions.append(
                    {
                        "observed_at": committed_at,
                        "observed_until": None,
                        "present": False,
                        "source_commit_sha": commit_sha,
                        "id": airport_id,
                    }
                )

        next_aliases = _alias_state(next_airports)
        changed_aliases = {
            key
            for key in current_aliases.keys() | next_aliases.keys()
            if current_aliases.get(key) != next_aliases.get(key)
        }
        for key in sorted(changed_aliases):
            if key in alias_version_indexes:
                alias_versions[alias_version_indexes.pop(key)][
                    "observed_until"
                ] = committed_at
            elif key in existing_open_alias_keys:
                alias_closures.setdefault(key, committed_at)

            if key in next_aliases:
                airport_id, code_type = key
                alias_version_indexes[key] = len(alias_versions)
                alias_versions.append(
                    {
                        "airport_id": airport_id,
                        "code": next_aliases[key],
                        "code_type": code_type,
                        "observed_at": committed_at,
                        "observed_until": None,
                        "source_commit_sha": commit_sha,
                    }
                )

        commit_manifest.append(
            {
                "commit_sha": commit_sha,
                "committed_at": committed_at,
                "airport_count": len(next_airports),
                "airport_change_count": len(changed_airports),
                "alias_change_count": len(changed_aliases),
                "is_baseline": commit_sha == baseline_commit_sha,
                "state_applied": True,
            }
        )
        current_airports = next_airports
        current_aliases = next_aliases
        latest_state_commit_sha = commit_sha
        latest_state_committed_at = committed_at

    assert latest_state_commit_sha is not None
    assert latest_state_committed_at is not None
    latest_commit_sha, latest_committed_at = commits[-1]
    return _HistoryDelta(
        airport_versions=airport_versions,
        alias_versions=alias_versions,
        commit_manifest=commit_manifest,
        airport_closures=airport_closures,
        alias_closures=alias_closures,
        airport_columns=airport_columns,
        latest_commit_sha=latest_commit_sha,
        latest_committed_at=latest_committed_at,
        latest_state_commit_sha=latest_state_commit_sha,
        latest_state_committed_at=latest_state_committed_at,
    )


def _current_airports(
    airport_versions: pl.DataFrame,
    airport_columns: list[str],
) -> AirportRows:
    current = airport_versions.filter(
        pl.col("observed_until").is_null() & pl.col("present")
    ).select(*airport_columns)
    return {
        int(row["id"]): row
        for row in current.iter_rows(named=True)
    }


def _open_airport_ids(airport_versions: pl.DataFrame) -> set[int]:
    return set(
        airport_versions.filter(pl.col("observed_until").is_null())
        .get_column("id")
        .to_list()
    )


def _open_alias_keys(alias_versions: pl.DataFrame) -> set[AliasKey]:
    return {
        (int(row["airport_id"]), str(row["code_type"]))
        for row in alias_versions.filter(
            pl.col("observed_until").is_null()
        ).iter_rows(named=True)
    }


def _close_airport_versions(
    airport_versions: pl.DataFrame,
    closures: dict[int, datetime],
) -> pl.DataFrame:
    if not closures:
        return airport_versions
    closure_frame = pl.DataFrame(
        {
            "id": list(closures),
            "_closed_at": list(closures.values()),
        },
        schema={
            "id": pl.Int64,
            "_closed_at": pl.Datetime("us", "UTC"),
        },
    )
    return (
        airport_versions.join(closure_frame, on="id", how="left")
        .with_columns(
            pl.when(
                pl.col("observed_until").is_null()
                & pl.col("_closed_at").is_not_null()
            )
            .then(pl.col("_closed_at"))
            .otherwise(pl.col("observed_until"))
            .alias("observed_until")
        )
        .drop("_closed_at")
    )


def _close_alias_versions(
    alias_versions: pl.DataFrame,
    closures: dict[AliasKey, datetime],
) -> pl.DataFrame:
    if not closures:
        return alias_versions
    closure_frame = pl.DataFrame(
        [
            {
                "airport_id": airport_id,
                "code_type": code_type,
                "_closed_at": closed_at,
            }
            for (airport_id, code_type), closed_at in closures.items()
        ],
        schema={
            "airport_id": pl.Int64,
            "code_type": pl.String,
            "_closed_at": pl.Datetime("us", "UTC"),
        },
    )
    return (
        alias_versions.join(
            closure_frame,
            on=["airport_id", "code_type"],
            how="left",
        )
        .with_columns(
            pl.when(
                pl.col("observed_until").is_null()
                & pl.col("_closed_at").is_not_null()
            )
            .then(pl.col("_closed_at"))
            .otherwise(pl.col("observed_until"))
            .alias("observed_until")
        )
        .drop("_closed_at")
    )


def _append_records(
    frame: pl.DataFrame,
    records: list[dict[str, object]],
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    if not records:
        return frame.select(*schema)
    return pl.concat(
        [frame.select(*schema), _records_frame(records, schema)],
        how="vertical",
    )


def _write_metadata(paths: HistoryPaths, metadata: dict[str, object]) -> None:
    paths.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_history(
    paths: HistoryPaths | None = None,
    *,
    baseline_date: date = BASELINE_DATE,
    repository_url: str = OURAIRPORTS_REPOSITORY_URL,
) -> HistoryPaths:
    paths = paths or HistoryPaths.defaults()
    ensure_repository(paths.repository, repository_url)
    commits = _commits_from_baseline(paths.repository, baseline_date)
    baseline_commit_sha = commits[0][0]
    delta = _derive_delta(
        paths.repository,
        commits,
        airport_columns=[],
        current_airports={},
        existing_open_airport_ids=set(),
        existing_open_alias_keys=set(),
        baseline_commit_sha=baseline_commit_sha,
    )
    airport_versions = _records_frame(
        delta.airport_versions,
        _airport_schema(delta.airport_columns),
    )
    alias_versions = _records_frame(delta.alias_versions, _alias_schema())
    commit_manifest = _records_frame(delta.commit_manifest, _commit_schema())
    _write_catalog(
        paths,
        airport_versions,
        alias_versions,
        commit_manifest,
        delta.airport_columns,
    )

    built_at = datetime.now(timezone.utc).isoformat()
    _write_metadata(
        paths,
        {
            "schema_version": SCHEMA_VERSION,
            "repository_url": repository_url,
            "repository_path": str(paths.repository),
            "baseline_date": baseline_date.isoformat(),
            "built_at": built_at,
            "updated_at": built_at,
            "repository_checked_at": built_at,
            "first_commit_sha": commits[0][0],
            "first_committed_at": commits[0][1].isoformat(),
            "latest_commit_sha": delta.latest_commit_sha,
            "latest_committed_at": delta.latest_committed_at.isoformat(),
            "latest_state_commit_sha": delta.latest_state_commit_sha,
            "latest_state_committed_at": (
                delta.latest_state_committed_at.isoformat()
            ),
            "commit_count": commit_manifest.height,
            "airport_version_count": airport_versions.height,
            "alias_version_count": alias_versions.height,
            "airport_columns": delta.airport_columns,
        },
    )
    return paths


def update_history(paths: HistoryPaths) -> HistoryPaths:
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    commits = _commits_after(
        paths.repository,
        str(metadata["latest_commit_sha"]),
        datetime.fromisoformat(str(metadata["latest_committed_at"])),
    )
    if not commits:
        return paths

    airport_columns = [str(column) for column in metadata["airport_columns"]]
    airport_versions = pl.read_parquet(paths.airport_versions)
    alias_versions = pl.read_parquet(paths.alias_versions)
    commit_manifest = pl.read_parquet(paths.commits)
    delta = _derive_delta(
        paths.repository,
        commits,
        airport_columns=airport_columns,
        current_airports=_current_airports(
            airport_versions,
            airport_columns,
        ),
        existing_open_airport_ids=_open_airport_ids(airport_versions),
        existing_open_alias_keys=_open_alias_keys(alias_versions),
        latest_state_commit_sha=str(metadata["latest_state_commit_sha"]),
        latest_state_committed_at=datetime.fromisoformat(
            str(metadata["latest_state_committed_at"])
        ),
    )

    airport_versions = _align_airport_frame(
        _close_airport_versions(
            airport_versions,
            delta.airport_closures,
        ),
        delta.airport_columns,
    )
    airport_versions = _append_records(
        airport_versions,
        delta.airport_versions,
        _airport_schema(delta.airport_columns),
    )
    alias_versions = _append_records(
        _close_alias_versions(alias_versions, delta.alias_closures),
        delta.alias_versions,
        _alias_schema(),
    )
    commit_manifest = _append_records(
        commit_manifest,
        delta.commit_manifest,
        _commit_schema(),
    )
    _write_catalog(
        paths,
        airport_versions,
        alias_versions,
        commit_manifest,
        delta.airport_columns,
    )

    updated_at = datetime.now(timezone.utc).isoformat()
    metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "updated_at": updated_at,
            "repository_checked_at": updated_at,
            "latest_commit_sha": delta.latest_commit_sha,
            "latest_committed_at": delta.latest_committed_at.isoformat(),
            "latest_state_commit_sha": delta.latest_state_commit_sha,
            "latest_state_committed_at": (
                delta.latest_state_committed_at.isoformat()
            ),
            "commit_count": commit_manifest.height,
            "airport_version_count": airport_versions.height,
            "alias_version_count": alias_versions.height,
            "airport_columns": delta.airport_columns,
        }
    )
    _write_metadata(paths, metadata)
    return paths
