from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import polars as pl

from airports.history.build import (
    build_history,
    latest_airports_commit,
    refresh_repository,
    update_history,
)
from airports.history.schema import (
    BASELINE_DATE,
    Commit,
    HistoryPaths,
    OURAIRPORTS_REPOSITORY_URL,
    SCHEMA_VERSION,
)


REPOSITORY_REFRESH_INTERVAL = timedelta(hours=1)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _repository_refresh_due(checked_at: datetime, now: datetime) -> bool:
    return (
        checked_at.date() < now.date()
        or now - checked_at >= REPOSITORY_REFRESH_INTERVAL
    )


def _record_repository_check(history: "AirportHistory", checked_at: datetime) -> None:
    metadata = history.metadata
    metadata["repository_checked_at"] = checked_at.isoformat()
    history.paths.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _next_midnight(day: date) -> datetime:
    return datetime.combine(
        day + timedelta(days=1),
        time.min,
        tzinfo=timezone.utc,
    )


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


@dataclass(frozen=True)
class AirportHistory:
    paths: HistoryPaths

    @classmethod
    def open(
        cls,
        history_dir: Path | None = None,
        *,
        repository_path: Path | None = None,
        repository_url: str = OURAIRPORTS_REPOSITORY_URL,
        baseline_date: date = BASELINE_DATE,
    ) -> "AirportHistory":
        defaults = HistoryPaths.defaults()
        selected_history_dir = history_dir or defaults.history_dir
        paths = HistoryPaths(
            history_dir=selected_history_dir,
            repository=(
                repository_path
                or (
                    selected_history_dir.parent / "repository"
                    if history_dir is not None
                    else defaults.repository
                )
            ),
        )
        history_exists = all(path.exists() for path in paths.required_files)
        schema_is_current = history_exists and (
            json.loads(paths.metadata.read_text(encoding="utf-8"))[
                "schema_version"
            ]
            == SCHEMA_VERSION
        )
        if not schema_is_current:
            build_history(
                paths,
                baseline_date=baseline_date,
                repository_url=repository_url,
            )
        history = cls(paths)
        checked_at = datetime.fromisoformat(
            str(history.metadata["repository_checked_at"])
        )
        now = _utc_now()
        if _repository_refresh_due(checked_at, now):
            refresh_repository(paths.repository)
            if (
                latest_airports_commit(paths.repository)
                != history.latest_available_commit
            ):
                update_history(paths)
                history = cls(paths)
            _record_repository_check(history, now)
        return history

    @property
    def metadata(self) -> dict[str, object]:
        return json.loads(self.paths.metadata.read_text(encoding="utf-8"))

    @property
    def latest_available_at(self) -> datetime:
        return datetime.fromisoformat(
            str(self.metadata["repository_checked_at"])
        )

    @property
    def latest_available_date(self) -> date:
        return self.latest_available_at.date()

    @property
    def latest_available_commit(self) -> str:
        return str(self.metadata["latest_commit_sha"])

    def airports_as_of(self, day: date) -> pl.DataFrame:
        boundary = _next_midnight(day)
        return (
            pl.scan_parquet(self.paths.airport_versions)
            .filter(
                pl.col("present")
                & (pl.col("observed_at") < boundary)
                & (
                    pl.col("observed_until").is_null()
                    | (pl.col("observed_until") >= boundary)
                )
            )
            .collect()
        )

    def as_of(self, day: date) -> pl.DataFrame:
        return self.airports_as_of(day)

    def aliases_as_of(self, day: date) -> pl.DataFrame:
        boundary = _next_midnight(day)
        return (
            pl.scan_parquet(self.paths.alias_versions)
            .filter(
                (pl.col("observed_at") < boundary)
                & (
                    pl.col("observed_until").is_null()
                    | (pl.col("observed_until") >= boundary)
                )
            )
            .collect()
        )

    def aliases_through(
        self,
        start_date: date,
        cutoff_date: date,
    ) -> pl.DataFrame:
        start = _midnight(start_date)
        end = _next_midnight(cutoff_date)
        return (
            pl.scan_parquet(self.paths.alias_versions)
            .filter(
                (pl.col("observed_at") < end)
                & (
                    pl.col("observed_until").is_null()
                    | (pl.col("observed_until") > start)
                )
            )
            .collect()
        )

    def commit_as_of(self, day: date) -> Commit:
        boundary = _next_midnight(day)
        row = (
            pl.scan_parquet(self.paths.commits)
            .filter(
                pl.col("state_applied")
                & (pl.col("committed_at") < boundary)
            )
            .sort("committed_at", descending=True)
            .select("commit_sha", "committed_at")
            .first()
            .collect()
            .row(0, named=True)
        )
        return Commit(sha=row["commit_sha"], committed_at=row["committed_at"])

    def source_commit_sha(self, day: date) -> str:
        return self.commit_as_of(day).sha

    def commits(self) -> pl.DataFrame:
        return pl.read_parquet(self.paths.commits)
