from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Self

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource


EVALUATION_METADATA_COLUMNS = (
    "test_date",
    "adsb_src",
    "test_src",
    "gold_src",
    "target_day_adsb",
)
_METRIC_COLUMNS = (
    "gold_icao_count",
    "gold_icao_missing_from_test_count",
    "gold_icao_missing_from_test_pct",
    "true_positive",
    "false_positive",
    "false_negative",
    "precision",
    "recall",
    "f1",
    "fully_correct_precision",
    "fully_correct_recall",
    "fully_correct_f1",
    "takeoff_airport_id_total_count",
    "takeoff_airport_id_match_count",
    "takeoff_airport_id_incorrect_match_count",
    "takeoff_airport_id_match_pct",
    "takeoff_airport_id_null_count",
    "takeoff_airport_id_null_pct",
    "landing_airport_id_total_count",
    "landing_airport_id_match_count",
    "landing_airport_id_incorrect_match_count",
    "landing_airport_id_match_pct",
    "landing_airport_id_null_count",
    "landing_airport_id_null_pct",
    "airport_id_total_count",
    "airport_id_match_count",
    "airport_id_incorrect_match_count",
    "airport_id_match_pct",
)
_RATE_COLUMNS = {
    "takeoff_airport_id_null_pct",
    "landing_airport_id_null_pct",
    "gold_icao_missing_from_test_pct",
    "precision",
    "recall",
    "f1",
    "fully_correct_precision",
    "fully_correct_recall",
    "fully_correct_f1",
    "takeoff_airport_id_match_pct",
    "landing_airport_id_match_pct",
    "airport_id_match_pct",
}
EVALUATION_STATS_SCHEMA = {
    **dict.fromkeys(EVALUATION_METADATA_COLUMNS, pl.String),
    "target_day_adsb": pl.Boolean,
    **{
        column: pl.Float64 if column in _RATE_COLUMNS else pl.Int64
        for column in _METRIC_COLUMNS
    },
}


def _count(row: Mapping[str, Any], column: str) -> int:
    value = float(row[column])
    if not value.is_integer():
        raise ValueError(f"{column} must be an integer count")
    return int(value)


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationStats:
    """Typed counts and derived metrics for one daily source comparison."""

    test_date: date
    adsb_source: ADSBSource | None
    test_source: str
    gold_source: str
    gold_icao_count: int
    gold_icao_missing_from_test_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    takeoff_airport_id_match_count: int
    landing_airport_id_match_count: int
    airport_id_match_count: int
    takeoff_airport_id_null_count: int = 0
    landing_airport_id_null_count: int = 0
    target_day_adsb: bool = False

    def __post_init__(self) -> None:
        counts = (
            self.true_positive,
            self.false_positive,
            self.false_negative,
            self.takeoff_airport_id_match_count,
            self.landing_airport_id_match_count,
            self.airport_id_match_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("Evaluation counts cannot be negative")
        if any(value > self.true_positive for value in counts[3:]):
            raise ValueError("Airport match counts cannot exceed true_positive")
        for null_count in (
            self.takeoff_airport_id_null_count,
            self.landing_airport_id_null_count,
        ):
            if not 0 <= null_count <= self.true_positive + self.false_positive:
                raise ValueError(
                    "Airport null counts must be between zero and test flight count"
                )
        if not 0 <= self.gold_icao_missing_from_test_count <= self.gold_icao_count:
            raise ValueError(
                "Missing gold ICAO count must be between zero and gold_icao_count"
            )
        if not self.test_source or not self.gold_source:
            raise ValueError("Evaluation sources cannot be empty")

    @classmethod
    def from_comparison(
        cls,
        comparison: pl.DataFrame,
        *,
        test_date: date,
        adsb_source: ADSBSource | None,
        test_source: str,
        gold_source: str,
        gold_icao_count: int,
        gold_icao_missing_from_test_count: int,
        target_day_adsb: bool = False,
    ) -> Self:
        def count(expression: pl.Expr) -> int:
            return comparison.filter(expression).height

        return cls(
            test_date=test_date,
            adsb_source=adsb_source,
            test_source=test_source,
            gold_source=gold_source,
            target_day_adsb=target_day_adsb,
            gold_icao_count=gold_icao_count,
            gold_icao_missing_from_test_count=gold_icao_missing_from_test_count,
            true_positive=count(pl.col("match_status") == "both"),
            false_positive=count(pl.col("match_status") == "df0_only"),
            false_negative=count(pl.col("match_status") == "df1_only"),
            takeoff_airport_id_match_count=count(
                pl.col("same_takeoff_airport_id")
            ),
            landing_airport_id_match_count=count(
                pl.col("same_landing_airport_id")
            ),
            airport_id_match_count=count(pl.col("same_airport_id")),
            takeoff_airport_id_null_count=count(
                pl.col("match_status").is_in(["both", "df0_only"])
                & pl.col("takeoff_airport_id").is_null()
            ),
            landing_airport_id_null_count=count(
                pl.col("match_status").is_in(["both", "df0_only"])
                & pl.col("landing_airport_id").is_null()
            ),
        )

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Self:
        raw_date = row["test_date"]
        if raw_date == "average":
            raise ValueError("Average rows are not daily EvaluationStats")
        if isinstance(raw_date, datetime):
            test_date = raw_date.date()
        elif isinstance(raw_date, date):
            test_date = raw_date
        else:
            test_date = date.fromisoformat(str(raw_date))
        raw_adsb_source = row.get("adsb_src")
        return cls(
            test_date=test_date,
            adsb_source=(
                ADSBSource(str(raw_adsb_source)) if raw_adsb_source else None
            ),
            test_source=str(row["test_src"]),
            gold_source=str(row["gold_src"]),
            target_day_adsb=row.get("target_day_adsb", False),
            gold_icao_count=_count(row, "gold_icao_count"),
            gold_icao_missing_from_test_count=_count(
                row, "gold_icao_missing_from_test_count"
            ),
            true_positive=_count(row, "true_positive"),
            false_positive=_count(row, "false_positive"),
            false_negative=_count(row, "false_negative"),
            takeoff_airport_id_match_count=_count(
                row, "takeoff_airport_id_match_count"
            ),
            landing_airport_id_match_count=_count(
                row, "landing_airport_id_match_count"
            ),
            airport_id_match_count=_count(row, "airport_id_match_count"),
            takeoff_airport_id_null_count=_count(
                row, "takeoff_airport_id_null_count"
            ),
            landing_airport_id_null_count=_count(
                row, "landing_airport_id_null_count"
            ),
        )

    @property
    def takeoff_airport_id_null_pct(self) -> float:
        """Fraction of retained test flights with no resolved takeoff airport."""
        return _ratio(
            self.takeoff_airport_id_null_count,
            self.true_positive + self.false_positive,
        )

    @property
    def landing_airport_id_null_pct(self) -> float:
        """Fraction of retained test flights with no resolved landing airport."""
        return _ratio(
            self.landing_airport_id_null_count,
            self.true_positive + self.false_positive,
        )

    @property
    def gold_icao_missing_from_test_pct(self) -> float:
        """Fraction of distinct daily gold ICAOs absent from the test input."""
        return _ratio(self.gold_icao_missing_from_test_count, self.gold_icao_count)

    @property
    def precision(self) -> float:
        return _ratio(
            self.true_positive,
            self.true_positive + self.false_positive,
        )

    @property
    def recall(self) -> float:
        return _ratio(
            self.true_positive,
            self.true_positive + self.false_negative,
        )

    @property
    def f1(self) -> float:
        return _ratio(
            2 * self.precision * self.recall,
            self.precision + self.recall,
        )

    @property
    def fully_correct_precision(self) -> float:
        """Precision requiring correct segmentation and both airport IDs."""
        return _ratio(
            self.airport_id_match_count,
            self.true_positive + self.false_positive,
        )

    @property
    def fully_correct_recall(self) -> float:
        """Recall requiring correct segmentation and both airport IDs."""
        return _ratio(
            self.airport_id_match_count,
            self.true_positive + self.false_negative,
        )

    @property
    def fully_correct_f1(self) -> float:
        return _ratio(
            2 * self.fully_correct_precision * self.fully_correct_recall,
            self.fully_correct_precision + self.fully_correct_recall,
        )

    @property
    def takeoff_airport_id_match_pct(self) -> float:
        return _ratio(
            self.takeoff_airport_id_match_count,
            self.true_positive,
        )

    @property
    def landing_airport_id_match_pct(self) -> float:
        return _ratio(
            self.landing_airport_id_match_count,
            self.true_positive,
        )

    @property
    def airport_id_match_pct(self) -> float:
        return _ratio(self.airport_id_match_count, self.true_positive)

    def to_row(self) -> dict[str, str | int | float | None]:
        return {
            "test_date": self.test_date.isoformat(),
            "adsb_src": self.adsb_source.value if self.adsb_source else None,
            "test_src": self.test_source,
            "gold_src": self.gold_source,
            "target_day_adsb": self.target_day_adsb,
            "gold_icao_count": self.gold_icao_count,
            "gold_icao_missing_from_test_count": (
                self.gold_icao_missing_from_test_count
            ),
            "gold_icao_missing_from_test_pct": self.gold_icao_missing_from_test_pct,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "fully_correct_precision": self.fully_correct_precision,
            "fully_correct_recall": self.fully_correct_recall,
            "fully_correct_f1": self.fully_correct_f1,
            "takeoff_airport_id_total_count": self.true_positive,
            "takeoff_airport_id_match_count": self.takeoff_airport_id_match_count,
            "takeoff_airport_id_incorrect_match_count": (
                self.true_positive - self.takeoff_airport_id_match_count
            ),
            "landing_airport_id_total_count": self.true_positive,
            "landing_airport_id_match_count": self.landing_airport_id_match_count,
            "landing_airport_id_incorrect_match_count": (
                self.true_positive - self.landing_airport_id_match_count
            ),
            "airport_id_total_count": self.true_positive,
            "airport_id_match_count": self.airport_id_match_count,
            "airport_id_incorrect_match_count": (
                self.true_positive - self.airport_id_match_count
            ),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "takeoff_airport_id_match_pct": self.takeoff_airport_id_match_pct,
            "takeoff_airport_id_null_count": self.takeoff_airport_id_null_count,
            "takeoff_airport_id_null_pct": self.takeoff_airport_id_null_pct,
            "landing_airport_id_match_pct": self.landing_airport_id_match_pct,
            "landing_airport_id_null_count": self.landing_airport_id_null_count,
            "landing_airport_id_null_pct": self.landing_airport_id_null_pct,
            "airport_id_match_pct": self.airport_id_match_pct,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    comparison: pl.DataFrame
    stats: EvaluationStats


@dataclass(frozen=True, slots=True)
class EvaluationResults:
    entries: tuple[EvaluationResult, ...]

    @property
    def stats(self) -> tuple[EvaluationStats, ...]:
        return tuple(entry.stats for entry in self.entries)

    @property
    def comparisons(self) -> tuple[pl.DataFrame, ...]:
        return tuple(entry.comparison for entry in self.entries)

    def to_frame(
        self,
        *,
        include_averages: bool | None = None,
    ) -> pl.DataFrame:
        frame = evaluation_stats_frame(self.stats)
        if include_averages is None:
            include_averages = len({item.test_date for item in self.stats}) > 1
        if not include_averages or frame.is_empty():
            return frame

        frame = frame.with_columns(pl.col(_METRIC_COLUMNS).cast(pl.Float64))
        averages = (
            frame.group_by(["adsb_src", "test_src", "gold_src", "target_day_adsb"])
            .agg([pl.col(column).mean() for column in _METRIC_COLUMNS])
            .with_columns(pl.lit("average").alias("test_date"))
            .select(frame.columns)
        )
        return pl.concat([frame, averages])


def evaluation_stats_frame(
    stats: Sequence[EvaluationStats],
) -> pl.DataFrame:
    return pl.DataFrame(
        [item.to_row() for item in stats],
        schema=EVALUATION_STATS_SCHEMA,
    )
