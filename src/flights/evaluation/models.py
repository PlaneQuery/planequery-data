from __future__ import annotations

import math
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
    "takeoff_airport_id_non_null_count",
    "takeoff_airport_id_non_null_match_pct",
    "takeoff_airport_id_null_count",
    "takeoff_airport_id_null_pct",
    "landing_airport_id_total_count",
    "landing_airport_id_match_count",
    "landing_airport_id_incorrect_match_count",
    "landing_airport_id_match_pct",
    "landing_airport_id_non_null_count",
    "landing_airport_id_non_null_match_pct",
    "landing_airport_id_null_count",
    "landing_airport_id_null_pct",
    "airport_id_total_count",
    "airport_id_match_count",
    "airport_id_incorrect_match_count",
    "airport_id_match_pct",
    "median_max_endpoint_time_error_minutes",
    "p90_max_endpoint_time_error_minutes",
    "both_endpoints_within_5m_recall",
    "both_endpoints_within_15m_recall",
    "both_endpoints_within_30m_recall",
    "takeoff_time_bias_minutes",
    "landing_time_bias_minutes",
)
_TIMING_METRIC_COLUMNS = (
    "median_max_endpoint_time_error_minutes",
    "p90_max_endpoint_time_error_minutes",
    "both_endpoints_within_5m_recall",
    "both_endpoints_within_15m_recall",
    "both_endpoints_within_30m_recall",
    "takeoff_time_bias_minutes",
    "landing_time_bias_minutes",
)
_RATE_COLUMNS = {
    "takeoff_airport_id_non_null_match_pct",
    "landing_airport_id_non_null_match_pct",
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
    "both_endpoints_within_5m_recall",
    "both_endpoints_within_15m_recall",
    "both_endpoints_within_30m_recall",
}
_FLOAT_METRIC_COLUMNS = _RATE_COLUMNS | set(_TIMING_METRIC_COLUMNS)
EVALUATION_STATS_SCHEMA = {
    **dict.fromkeys(EVALUATION_METADATA_COLUMNS, pl.String),
    "target_day_adsb": pl.Boolean,
    **{
        column: pl.Float64 if column in _FLOAT_METRIC_COLUMNS else pl.Int64
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


def _optional_float(row: Mapping[str, Any], column: str) -> float | None:
    value = row[column]
    return None if value is None else float(value)


def _time_difference_minutes(test_column: str, gold_column: str) -> pl.Expr:
    return (
        (pl.col(test_column) - pl.col(gold_column))
        .dt.total_milliseconds()
        .cast(pl.Float64)
        / 60_000.0
    )


def _timing_metric_expressions() -> list[pl.Expr]:
    matched = pl.col("match_status") == "both"
    retained_gold = pl.col("match_status").is_in(["both", "df1_only"])
    takeoff_error = _time_difference_minutes(
        "takeoff_time", "takeoff_time_df1"
    )
    landing_error = _time_difference_minutes(
        "landing_time", "landing_time_df1"
    )
    max_endpoint_error = pl.max_horizontal(
        takeoff_error.abs(),
        landing_error.abs(),
    )

    def within_threshold_recall(minutes: int) -> pl.Expr:
        within_threshold = pl.when(matched).then(
            (max_endpoint_error <= minutes).fill_null(False)
        ).otherwise(False)
        return (
            pl.when(retained_gold)
            .then(within_threshold)
            .otherwise(None)
            .mean()
            .alias(f"both_endpoints_within_{minutes}m_recall")
        )

    return [
        max_endpoint_error.filter(matched).median().alias(
            "median_max_endpoint_time_error_minutes"
        ),
        max_endpoint_error.filter(matched).quantile(
            0.9,
            interpolation="linear",
        ).alias("p90_max_endpoint_time_error_minutes"),
        within_threshold_recall(5),
        within_threshold_recall(15),
        within_threshold_recall(30),
        takeoff_error.filter(matched).mean().alias(
            "takeoff_time_bias_minutes"
        ),
        landing_error.filter(matched).mean().alias(
            "landing_time_bias_minutes"
        ),
    ]


def _timing_metrics(comparison: pl.DataFrame) -> dict[str, float | None]:
    return comparison.select(_timing_metric_expressions()).row(0, named=True)


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
    takeoff_airport_id_non_null_count: int
    landing_airport_id_non_null_count: int
    takeoff_airport_id_null_count: int = 0
    landing_airport_id_null_count: int = 0
    target_day_adsb: bool = False
    median_max_endpoint_time_error_minutes: float | None = None
    p90_max_endpoint_time_error_minutes: float | None = None
    both_endpoints_within_5m_recall: float = 0.0
    both_endpoints_within_15m_recall: float = 0.0
    both_endpoints_within_30m_recall: float = 0.0
    takeoff_time_bias_minutes: float | None = None
    landing_time_bias_minutes: float | None = None

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
        for match_count, non_null_count in (
            (self.takeoff_airport_id_match_count, self.takeoff_airport_id_non_null_count),
            (self.landing_airport_id_match_count, self.landing_airport_id_non_null_count),
        ):
            if not match_count <= non_null_count <= self.true_positive:
                raise ValueError(
                    "Airport non-null counts must be between match count and true_positive"
                )
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
        time_errors = (
            self.median_max_endpoint_time_error_minutes,
            self.p90_max_endpoint_time_error_minutes,
        )
        if any(value is not None and value < 0 for value in time_errors):
            raise ValueError("Absolute timing errors cannot be negative")
        if any(
            value is not None and not math.isfinite(value)
            for value in (
                *time_errors,
                self.takeoff_time_bias_minutes,
                self.landing_time_bias_minutes,
            )
        ):
            raise ValueError("Timing metrics must be finite when present")
        timing_recalls = (
            self.both_endpoints_within_5m_recall,
            self.both_endpoints_within_15m_recall,
            self.both_endpoints_within_30m_recall,
        )
        if any(not 0 <= value <= 1 for value in timing_recalls):
            raise ValueError("Timing recalls must be between zero and one")
        if tuple(sorted(timing_recalls)) != timing_recalls:
            raise ValueError("Timing recalls must not decrease at wider thresholds")

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

        timing = _timing_metrics(comparison)

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
            takeoff_airport_id_non_null_count=count(
                (pl.col("match_status") == "both")
                & pl.col("takeoff_airport_id").is_not_null()
            ),
            landing_airport_id_non_null_count=count(
                (pl.col("match_status") == "both")
                & pl.col("landing_airport_id").is_not_null()
            ),
            takeoff_airport_id_null_count=count(
                pl.col("match_status").is_in(["both", "df0_only"])
                & pl.col("takeoff_airport_id").is_null()
            ),
            landing_airport_id_null_count=count(
                pl.col("match_status").is_in(["both", "df0_only"])
                & pl.col("landing_airport_id").is_null()
            ),
            median_max_endpoint_time_error_minutes=timing[
                "median_max_endpoint_time_error_minutes"
            ],
            p90_max_endpoint_time_error_minutes=timing[
                "p90_max_endpoint_time_error_minutes"
            ],
            both_endpoints_within_5m_recall=timing[
                "both_endpoints_within_5m_recall"
            ]
            or 0.0,
            both_endpoints_within_15m_recall=timing[
                "both_endpoints_within_15m_recall"
            ]
            or 0.0,
            both_endpoints_within_30m_recall=timing[
                "both_endpoints_within_30m_recall"
            ]
            or 0.0,
            takeoff_time_bias_minutes=timing["takeoff_time_bias_minutes"],
            landing_time_bias_minutes=timing["landing_time_bias_minutes"],
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
            takeoff_airport_id_non_null_count=_count(
                row, "takeoff_airport_id_non_null_count"
            ),
            landing_airport_id_non_null_count=_count(
                row, "landing_airport_id_non_null_count"
            ),
            takeoff_airport_id_null_count=_count(
                row, "takeoff_airport_id_null_count"
            ),
            landing_airport_id_null_count=_count(
                row, "landing_airport_id_null_count"
            ),
            median_max_endpoint_time_error_minutes=_optional_float(
                row, "median_max_endpoint_time_error_minutes"
            ),
            p90_max_endpoint_time_error_minutes=_optional_float(
                row, "p90_max_endpoint_time_error_minutes"
            ),
            both_endpoints_within_5m_recall=float(
                row["both_endpoints_within_5m_recall"]
            ),
            both_endpoints_within_15m_recall=float(
                row["both_endpoints_within_15m_recall"]
            ),
            both_endpoints_within_30m_recall=float(
                row["both_endpoints_within_30m_recall"]
            ),
            takeoff_time_bias_minutes=_optional_float(
                row, "takeoff_time_bias_minutes"
            ),
            landing_time_bias_minutes=_optional_float(
                row, "landing_time_bias_minutes"
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
    def takeoff_airport_id_non_null_match_pct(self) -> float:
        """Accuracy among matched flights with a non-null test takeoff airport."""
        return _ratio(
            self.takeoff_airport_id_match_count,
            self.takeoff_airport_id_non_null_count,
        )

    @property
    def landing_airport_id_non_null_match_pct(self) -> float:
        """Accuracy among matched flights with a non-null test landing airport."""
        return _ratio(
            self.landing_airport_id_match_count,
            self.landing_airport_id_non_null_count,
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
            "takeoff_airport_id_non_null_count": self.takeoff_airport_id_non_null_count,
            "takeoff_airport_id_non_null_match_pct": (
                self.takeoff_airport_id_non_null_match_pct
            ),
            "takeoff_airport_id_null_count": self.takeoff_airport_id_null_count,
            "takeoff_airport_id_null_pct": self.takeoff_airport_id_null_pct,
            "landing_airport_id_match_pct": self.landing_airport_id_match_pct,
            "landing_airport_id_non_null_count": self.landing_airport_id_non_null_count,
            "landing_airport_id_non_null_match_pct": (
                self.landing_airport_id_non_null_match_pct
            ),
            "landing_airport_id_null_count": self.landing_airport_id_null_count,
            "landing_airport_id_null_pct": self.landing_airport_id_null_pct,
            "airport_id_match_pct": self.airport_id_match_pct,
            "median_max_endpoint_time_error_minutes": (
                self.median_max_endpoint_time_error_minutes
            ),
            "p90_max_endpoint_time_error_minutes": (
                self.p90_max_endpoint_time_error_minutes
            ),
            "both_endpoints_within_5m_recall": (
                self.both_endpoints_within_5m_recall
            ),
            "both_endpoints_within_15m_recall": (
                self.both_endpoints_within_15m_recall
            ),
            "both_endpoints_within_30m_recall": (
                self.both_endpoints_within_30m_recall
            ),
            "takeoff_time_bias_minutes": self.takeoff_time_bias_minutes,
            "landing_time_bias_minutes": self.landing_time_bias_minutes,
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
        group_columns = [
            "adsb_src",
            "test_src",
            "gold_src",
            "target_day_adsb",
        ]
        non_timing_metric_columns = [
            column
            for column in _METRIC_COLUMNS
            if column not in _TIMING_METRIC_COLUMNS
        ]
        averages = (
            frame.group_by(group_columns)
            .agg([pl.col(column).mean() for column in non_timing_metric_columns])
        )
        comparison_frames = [
            entry.comparison.with_columns(
                pl.lit(
                    entry.stats.adsb_source.value
                    if entry.stats.adsb_source is not None
                    else None,
                    dtype=pl.String,
                ).alias("adsb_src"),
                pl.lit(entry.stats.test_source).alias("test_src"),
                pl.lit(entry.stats.gold_source).alias("gold_src"),
                pl.lit(entry.stats.target_day_adsb).alias("target_day_adsb"),
            )
            for entry in self.entries
        ]
        timing_averages = (
            pl.concat(comparison_frames, how="diagonal_relaxed")
            .group_by(group_columns)
            .agg(_timing_metric_expressions())
        )
        averages = (
            averages.join(
                timing_averages,
                on=group_columns,
                how="left",
                nulls_equal=True,
            )
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
