from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

from airports.history import (
    AirportHistory,
    HistoryPaths,
    build_history,
    update_history,
)
from airports.history import build as history_build
from airports.history.schema import SCHEMA_VERSION
from airports.history import store as history_store


AIRPORT_COLUMNS = (
    "id",
    "ident",
    "type",
    "name",
    "latitude_deg",
    "longitude_deg",
    "elevation_ft",
    "continent",
    "iso_country",
    "iso_region",
    "municipality",
    "scheduled_service",
    "icao_code",
    "iata_code",
    "gps_code",
    "local_code",
    "home_link",
    "wikipedia_link",
    "keywords",
    "test_extra_field",
)


def _git(repository: Path, *arguments: str, env=None) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _airport(
    airport_id: int,
    ident: str,
    airport_type: str = "small_airport",
    *,
    name: str | None = None,
) -> dict[str, str]:
    return {
        "id": str(airport_id),
        "ident": ident,
        "type": airport_type,
        "name": name or ident,
        "latitude_deg": "40.0",
        "longitude_deg": "-75.0",
        "elevation_ft": "100",
        "continent": "NA",
        "iso_country": "US",
        "iso_region": "US-PA",
        "municipality": "Testville",
        "scheduled_service": "no",
        "icao_code": ident if len(ident) == 4 else "",
        "iata_code": "",
        "gps_code": ident,
        "local_code": ident[-3:],
        "home_link": "",
        "wikipedia_link": "",
        "keywords": "",
        "test_extra_field": f"extra-{airport_id}",
    }


def _commit_airports(
    repository: Path,
    rows: list[dict[str, str]],
    committed_at: str,
    message: str,
    *,
    fieldnames: tuple[str, ...] = AIRPORT_COLUMNS,
) -> str:
    with (repository / "airports.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                column: row.get(column, "")
                for column in fieldnames
            }
            for row in rows
        )
    _git(repository, "add", "airports.csv")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_AUTHOR_DATE": committed_at,
        "GIT_COMMITTER_DATE": committed_at,
    }
    _git(repository, "commit", "-m", message, env=env)
    return _git(repository, "rev-parse", "HEAD")


def _commit_non_airport_file(
    repository: Path,
    committed_at: str,
) -> str:
    (repository / "README.md").write_text(committed_at, encoding="utf-8")
    _git(repository, "add", "README.md")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_AUTHOR_DATE": committed_at,
        "GIT_COMMITTER_DATE": committed_at,
    }
    _git(repository, "commit", "-m", "non-airport change", env=env)
    return _git(repository, "rev-parse", "HEAD")


def _history_repository(tmp_path: Path) -> tuple[Path, list[str]]:
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "-b", "main")

    commits = [
        _commit_airports(
            repository,
            [
                _airport(1, "KOLD"),
                _airport(2, "KCLO", "closed"),
            ],
            "2023-12-31T12:00:00+00:00",
            "baseline",
        ),
        _commit_airports(
            repository,
            [
                _airport(1, "KNEW", name="Renamed Airport"),
                _airport(3, "KADD", "heliport"),
            ],
            "2024-01-02T12:00:00+00:00",
            "rename remove and add",
        ),
        _commit_airports(
            repository,
            [
                _airport(1, "KNEW", "closed", name="Renamed Airport"),
                _airport(3, "KADD", "heliport"),
            ],
            "2024-02-01T12:00:00+00:00",
            "close airport",
        ),
    ]
    return repository, commits


@pytest.mark.parametrize("baseline", [date(2024, 1, 1), date(2021, 1, 1)])
def test_dates_before_history_use_earliest_snapshot(tmp_path, baseline):
    from airports.catalog import AirportCatalogView

    repository, commits = _history_repository(tmp_path)
    paths = HistoryPaths(history_dir=tmp_path / "derived", repository=repository)
    build_history(paths, baseline_date=baseline)
    history = AirportHistory(paths)
    early = date(2021, 2, 1)
    first = date(2023, 12, 31)

    assert history.source_commit_sha(early) == commits[0]
    assert history.as_of(early).equals(history.as_of(first))
    assert history.aliases_as_of(early).equals(history.aliases_as_of(first))
    assert history.aliases_through(early, date(2021, 3, 1)).equals(
        history.aliases_as_of(first)
    )
    view = AirportCatalogView.from_history(
        history, flight_date=early, catalog_available_date=date(2024, 2, 1)
    )
    assert view.get_airport_by_ident("KOLD") is not None
    assert view.get_airport_by_ident("KNEW") is None
    assert view.provenance.flight_date == early
    assert view.provenance.observed_commit_sha == commits[0]
    assert view.provenance.knowledge_commit_sha == commits[0]
    assert history.source_commit_sha(date(2024, 1, 3)) == commits[1]


def test_builds_change_only_temporal_catalog(tmp_path):
    repository, commits = _history_repository(tmp_path)
    paths = HistoryPaths(
        history_dir=tmp_path / "derived",
        repository=repository,
    )

    build_history(paths)
    history = AirportHistory.open(
        paths.history_dir,
        repository_path=paths.repository,
    )

    baseline = history.airports_as_of(date(2024, 1, 1)).sort("id")
    assert baseline.select("id", "ident", "type").rows() == [
        (1, "KOLD", "small_airport"),
        (2, "KCLO", "closed"),
    ]
    assert "test_extra_field" in baseline.columns

    january = history.airports_as_of(date(2024, 1, 3)).sort("id")
    assert january.select("id", "ident", "type").rows() == [
        (1, "KNEW", "small_airport"),
        (3, "KADD", "heliport"),
    ]

    february = history.airports_as_of(date(2024, 2, 1)).sort("id")
    assert february.select("id", "ident", "type").rows() == [
        (1, "KNEW", "closed"),
        (3, "KADD", "heliport"),
    ]

    versions = pl.read_parquet(paths.airport_versions)
    airport_one = versions.filter(pl.col("id") == 1).sort("observed_at")
    assert airport_one.height == 3
    assert airport_one["observed_until"].to_list() == [
        datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
        datetime(2024, 2, 1, 12, tzinfo=timezone.utc),
        None,
    ]
    assert airport_one["source_commit_sha"].to_list() == commits

    airport_two = versions.filter(pl.col("id") == 2).sort("observed_at")
    assert airport_two["present"].to_list() == [True, False]
    assert airport_two["source_commit_sha"].to_list() == commits[:2]


def test_aliases_are_versioned_only_when_codes_change(tmp_path):
    repository, _ = _history_repository(tmp_path)
    paths = HistoryPaths(tmp_path / "derived", repository)
    build_history(paths)
    history = AirportHistory.open(paths.history_dir, repository_path=repository)

    aliases = pl.read_parquet(paths.alias_versions)
    ident_aliases = aliases.filter(
        (pl.col("airport_id") == 1) & (pl.col("code_type") == "ident")
    ).sort("observed_at")
    assert ident_aliases["code"].to_list() == ["KOLD", "KNEW"]
    assert ident_aliases["observed_until"].to_list() == [
        datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
        None,
    ]
    assert (
        history.aliases_as_of(date(2024, 1, 1))
        .filter(pl.col("airport_id") == 1)["code"]
        .to_list()
        .count("KOLD")
        == 3
    )
    aliases_during_grace = history.aliases_through(
        date(2024, 1, 1),
        date(2024, 1, 3),
    ).filter(
        (pl.col("airport_id") == 1) & (pl.col("code_type") == "ident")
    )
    assert aliases_during_grace["code"].to_list() == ["KOLD", "KNEW"]


def test_commit_manifest_metadata_and_latest_properties(tmp_path):
    repository, commits = _history_repository(tmp_path)
    paths = HistoryPaths(tmp_path / "derived", repository)
    build_history(paths)
    history = AirportHistory.open(paths.history_dir, repository_path=repository)

    manifest = history.commits()
    assert manifest["commit_sha"].to_list() == commits
    assert manifest["is_baseline"].to_list() == [True, False, False]
    assert history.commit_as_of(date(2024, 1, 3)).sha == commits[1]
    assert history.source_commit_sha(date(2024, 1, 3)) == commits[1]
    assert history.as_of(date(2024, 1, 3)).equals(
        history.airports_as_of(date(2024, 1, 3))
    )
    assert history.latest_available_commit == commits[-1]
    assert history.latest_available_date == datetime.now(timezone.utc).date()

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["baseline_date"] == "2024-01-01"
    assert metadata["commit_count"] == 3
    assert metadata["airport_version_count"] == 6
    assert metadata["latest_commit_sha"] == commits[-1]


def test_open_clones_and_builds_when_history_is_missing(tmp_path):
    source, commits = _history_repository(tmp_path)
    ourairports_dir = tmp_path / "output" / "data" / "raw" / "ourairports"

    history = AirportHistory.open(
        ourairports_dir / "history",
        repository_path=ourairports_dir / "repository",
        repository_url=str(source),
    )

    assert history.paths.repository.is_dir()
    assert all(path.exists() for path in history.paths.required_files)
    assert history.latest_available_commit == commits[-1]


def test_open_rebuilds_an_old_history_schema(tmp_path):
    repository, commits = _history_repository(tmp_path)
    paths = HistoryPaths(tmp_path / "derived", repository)
    build_history(paths)
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    metadata["schema_version"] = 1
    paths.metadata.write_text(json.dumps(metadata), encoding="utf-8")

    history = AirportHistory.open(paths.history_dir, repository_path=repository)

    assert history.metadata["schema_version"] == SCHEMA_VERSION
    assert history.commits()["commit_sha"].to_list() == commits


def test_open_refreshes_stale_repository_and_rebuilds_history(tmp_path):
    source, _ = _history_repository(tmp_path)
    ourairports_dir = tmp_path / "output" / "data" / "raw" / "ourairports"
    history = AirportHistory.open(
        ourairports_dir / "history",
        repository_path=ourairports_dir / "repository",
        repository_url=str(source),
    )
    metadata = history.metadata
    metadata["repository_checked_at"] = "2024-01-01T00:00:00+00:00"
    history.paths.metadata.write_text(json.dumps(metadata), encoding="utf-8")

    latest_sha = _commit_airports(
        source,
        [_airport(1, "KFINAL")],
        "2024-03-01T12:00:00+00:00",
        "latest",
    )

    refreshed = AirportHistory.open(
        ourairports_dir / "history",
        repository_path=ourairports_dir / "repository",
        repository_url=str(source),
    )

    assert refreshed.latest_available_commit == latest_sha
    assert refreshed.as_of(date(2024, 3, 1))["ident"].to_list() == ["KFINAL"]


def test_concurrent_open_refreshes_shared_repository_once(tmp_path, monkeypatch):
    source, _ = _history_repository(tmp_path)
    ourairports_dir = tmp_path / "output" / "data" / "raw" / "ourairports"
    history = AirportHistory.open(
        ourairports_dir / "history",
        repository_path=ourairports_dir / "repository",
        repository_url=str(source),
    )
    metadata = history.metadata
    metadata["repository_checked_at"] = "2024-01-01T00:00:00+00:00"
    history.paths.metadata.write_text(json.dumps(metadata), encoding="utf-8")

    refresh_count = 0
    count_lock = threading.Lock()
    refresh_repository = history_store.refresh_repository

    def track_refresh(repository):
        nonlocal refresh_count
        with count_lock:
            refresh_count += 1
        time.sleep(0.05)
        refresh_repository(repository)

    monkeypatch.setattr(history_store, "refresh_repository", track_refresh)

    def open_history(_):
        return AirportHistory.open(
            ourairports_dir / "history",
            repository_path=ourairports_dir / "repository",
            repository_url=str(source),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        histories = list(executor.map(open_history, range(4)))

    assert refresh_count == 1
    assert len({item.latest_available_commit for item in histories}) == 1


@pytest.mark.parametrize(
    ("checked_at", "now"),
    [
        (
            datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 24, 11, 0, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 9, 24, 23, 45, tzinfo=timezone.utc),
            datetime(2026, 9, 25, 0, 15, tzinfo=timezone.utc),
        ),
    ],
    ids=("same-day-ttl", "utc-day-rollover"),
)
def test_open_refreshes_when_freshness_window_expires(
    tmp_path,
    monkeypatch,
    checked_at,
    now,
):
    source, _ = _history_repository(tmp_path)
    ourairports_dir = tmp_path / "output" / "data" / "raw" / "ourairports"
    history = AirportHistory.open(
        ourairports_dir / "history",
        repository_path=ourairports_dir / "repository",
        repository_url=str(source),
    )
    metadata = history.metadata
    metadata["repository_checked_at"] = checked_at.isoformat()
    history.paths.metadata.write_text(json.dumps(metadata), encoding="utf-8")
    latest_sha = _commit_airports(
        source,
        [_airport(1, "KFINAL")],
        "2024-03-01T12:00:00+00:00",
        "latest",
    )
    monkeypatch.setattr(history_store, "_utc_now", lambda: now)

    refreshed = AirportHistory.open(
        ourairports_dir / "history",
        repository_path=ourairports_dir / "repository",
        repository_url=str(source),
    )

    assert refreshed.latest_available_commit == latest_sha
    assert refreshed.metadata["repository_checked_at"] == now.isoformat()


def test_non_airport_repository_commit_does_not_rebuild_history(
    tmp_path,
    monkeypatch,
):
    source, commits = _history_repository(tmp_path)
    ourairports_dir = tmp_path / "output" / "data" / "raw" / "ourairports"
    history = AirportHistory.open(
        ourairports_dir / "history",
        repository_path=ourairports_dir / "repository",
        repository_url=str(source),
    )
    metadata = history.metadata
    original_built_at = metadata["built_at"]
    metadata["repository_checked_at"] = "2026-09-24T10:00:00+00:00"
    history.paths.metadata.write_text(json.dumps(metadata), encoding="utf-8")
    repository_head = _commit_non_airport_file(
        source,
        "2026-09-24T10:30:00+00:00",
    )
    now = datetime(2026, 9, 24, 11, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(history_store, "_utc_now", lambda: now)

    refreshed = AirportHistory.open(
        ourairports_dir / "history",
        repository_path=ourairports_dir / "repository",
        repository_url=str(source),
    )

    assert _git(refreshed.paths.repository, "rev-parse", "HEAD") == repository_head
    assert refreshed.latest_available_commit == commits[-1]
    assert refreshed.metadata["built_at"] == original_built_at
    assert refreshed.metadata["repository_checked_at"] == now.isoformat()


def test_update_reads_only_new_airport_snapshots(tmp_path, monkeypatch):
    repository, commits = _history_repository(tmp_path)
    paths = HistoryPaths(tmp_path / "derived", repository)
    build_history(paths)
    original_versions = pl.read_parquet(paths.airport_versions)

    latest_sha = _commit_airports(
        repository,
        [
            _airport(1, "KFINAL"),
            _airport(3, "KADD", "heliport"),
        ],
        "2024-03-01T12:00:00+00:00",
        "latest",
    )
    read_commits: list[str] = []
    read_airports_csv = history_build._read_airports_csv

    def track_read(repository_path, commit_sha):
        read_commits.append(commit_sha)
        return read_airports_csv(repository_path, commit_sha)

    monkeypatch.setattr(history_build, "_read_airports_csv", track_read)

    update_history(paths)

    assert read_commits == [latest_sha]
    manifest = pl.read_parquet(paths.commits)
    assert manifest["commit_sha"].to_list() == [*commits, latest_sha]
    versions = pl.read_parquet(paths.airport_versions)
    assert versions.height == original_versions.height + 1
    assert versions.filter(pl.col("id") == 3).equals(
        original_versions.filter(pl.col("id") == 3)
    )
    airport_one = versions.filter(pl.col("id") == 1).sort("observed_at")
    assert airport_one.tail(2)["observed_until"].to_list() == [
        datetime(2024, 3, 1, 12, tzinfo=timezone.utc),
        None,
    ]
    assert airport_one.tail(1)["ident"].item() == "KFINAL"
    assert json.loads(paths.metadata.read_text(encoding="utf-8"))[
        "latest_commit_sha"
    ] == latest_sha

    read_commits.clear()
    parquet_contents = paths.airport_versions.read_bytes()
    update_history(paths)
    assert read_commits == []
    assert paths.airport_versions.read_bytes() == parquet_contents


def test_zero_row_snapshot_is_recorded_without_changing_state(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    initial_rows = [_airport(1, "KONE"), _airport(2, "KTWO")]
    initial_sha = _commit_airports(
        repository,
        initial_rows,
        "2023-12-31T12:00:00+00:00",
        "baseline",
    )
    paths = HistoryPaths(tmp_path / "derived", repository)
    build_history(paths)

    zero_sha = _commit_airports(
        repository,
        [],
        "2024-01-02T12:00:00+00:00",
        "empty export",
    )
    update_history(paths)

    history = AirportHistory.open(paths.history_dir, repository_path=repository)
    assert history.as_of(date(2024, 1, 2)).sort("id")["id"].to_list() == [1, 2]
    assert history.commit_as_of(date(2024, 1, 2)).sha == initial_sha
    metadata = history.metadata
    assert metadata["latest_commit_sha"] == zero_sha
    assert metadata["latest_state_commit_sha"] == initial_sha

    recovery_sha = _commit_airports(
        repository,
        initial_rows,
        "2024-01-03T12:00:00+00:00",
        "restore export",
    )
    update_history(paths)

    manifest = pl.read_parquet(paths.commits)
    assert manifest["commit_sha"].to_list() == [
        initial_sha,
        zero_sha,
        recovery_sha,
    ]
    assert manifest["state_applied"].to_list() == [True, False, True]
    assert manifest["airport_count"].to_list() == [2, 0, 2]
    assert manifest["airport_change_count"].to_list() == [2, 0, 0]
    assert manifest["alias_change_count"].to_list() == [8, 0, 0]
    assert pl.read_parquet(paths.airport_versions).height == 2
    assert pl.read_parquet(paths.alias_versions).height == 8


def test_baseline_seed_precedes_start_of_first_flight_day(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    commits = [
        _commit_airports(
            repository,
            [_airport(1, "KPRE")],
            "2023-12-31T23:00:00+00:00",
            "pre-range seed",
        ),
        _commit_airports(
            repository,
            [_airport(1, "KMID")],
            "2024-01-01T06:00:00+00:00",
            "morning state",
        ),
        _commit_airports(
            repository,
            [_airport(1, "KEND")],
            "2024-01-01T18:00:00+00:00",
            "evening state",
        ),
    ]
    paths = HistoryPaths(tmp_path / "derived", repository)

    build_history(paths)

    manifest = pl.read_parquet(paths.commits)
    assert manifest["commit_sha"].to_list() == commits
    assert manifest["is_baseline"].to_list() == [True, False, False]
    versions = pl.read_parquet(paths.airport_versions).sort("observed_at")
    assert versions["observed_at"].to_list() == [
        datetime(2023, 12, 31, 23, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 6, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 18, tzinfo=timezone.utc),
    ]
    assert versions["observed_until"].to_list() == [
        datetime(2024, 1, 1, 6, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 18, tzinfo=timezone.utc),
        None,
    ]
    history = AirportHistory.open(paths.history_dir, repository_path=repository)
    assert history.as_of(date(2024, 1, 1))["ident"].item() == "KEND"
    assert (
        history.aliases_through(date(2024, 1, 1), date(2024, 1, 1))
        .filter(pl.col("code_type") == "ident")
        .sort("observed_at")["code"]
        .to_list()
        == ["KPRE", "KMID", "KEND"]
    )


def test_added_empty_column_does_not_create_airport_versions(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    initial_rows = [_airport(1, "KONE"), _airport(2, "KTWO")]
    _commit_airports(
        repository,
        initial_rows,
        "2023-12-31T12:00:00+00:00",
        "baseline without added column",
        fieldnames=AIRPORT_COLUMNS[:-1],
    )
    paths = HistoryPaths(tmp_path / "derived", repository)
    build_history(paths)

    empty_added_column_rows = [
        {**row, "test_extra_field": ""}
        for row in initial_rows
    ]
    _commit_airports(
        repository,
        empty_added_column_rows,
        "2024-01-02T12:00:00+00:00",
        "add empty column",
    )
    update_history(paths)

    versions = pl.read_parquet(paths.airport_versions).sort("id")
    assert "test_extra_field" in versions.columns
    assert versions.height == 2
    assert versions["test_extra_field"].to_list() == [None, None]
    assert pl.read_parquet(paths.commits).tail(1)[
        "airport_change_count"
    ].item() == 0

    changed_column_rows = [
        {
            **row,
            "test_extra_field": "enabled" if row["id"] == "1" else "",
        }
        for row in initial_rows
    ]
    _commit_airports(
        repository,
        changed_column_rows,
        "2024-01-03T12:00:00+00:00",
        "populate added column",
    )
    update_history(paths)

    versions = pl.read_parquet(paths.airport_versions)
    assert versions.height == 3
    assert (
        versions.filter(pl.col("id") == 1)
        .sort("observed_at")["test_extra_field"]
        .to_list()
        == [None, "enabled"]
    )
    assert versions.filter(pl.col("id") == 2).height == 1
    latest_manifest = pl.read_parquet(paths.commits).tail(1)
    assert latest_manifest["airport_change_count"].item() == 1
    assert latest_manifest["alias_change_count"].item() == 0
