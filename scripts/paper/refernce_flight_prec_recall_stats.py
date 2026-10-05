from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
from flights.flight_source import FlightSource
from flights.sources import FLIGHT_SOURCES_BY_NAME


@dataclass(frozen=True, slots=True)
class EvaluationStats:
    evaluation_date: date
    adsb_source: ADSBSource | None
    algorithm_source: FlightSource
    gold_source: FlightSource

    true_positive: int
    false_positive: int
    false_negative: int

    takeoff_airport_match_count: int
    landing_airport_match_count: int
    airport_match_count: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return (
            2 * self.precision * self.recall / denominator
            if denominator
            else 0.0
        )

    @property
    def takeoff_airport_match_pct(self) -> float:
        return self._airport_percentage(self.takeoff_airport_match_count)

    @property
    def landing_airport_match_pct(self) -> float:
        return self._airport_percentage(self.landing_airport_match_count)

    @property
    def airport_match_pct(self) -> float:
        return self._airport_percentage(self.airport_match_count)

    def _airport_percentage(self, match_count: int) -> float:
        return (
            match_count / self.true_positive
            if self.true_positive
            else 0.0
        )

    @classmethod
    def from_csv_row(
        cls,
        row: Mapping[str, Any],
    ) -> "EvaluationStats":
        if row["test_date"] == "average":
            raise ValueError(
                "Average rows do not contain actual integer counts"
            )

        return cls(
            evaluation_date=date.fromisoformat(row["test_date"]),
            adsb_source=parse_optional_adsb_source(row["adsb_src"]),
            algorithm_source=resolve_flight_source(row["test_src"]),
            gold_source=resolve_flight_source(row["gold_src"]),
            true_positive=parse_count(row["true_positive"]),
            false_positive=parse_count(row["false_positive"]),
            false_negative=parse_count(row["false_negative"]),
            takeoff_airport_match_count=parse_count(
                row["takeoff_airport_id_match_count"]
            ),
            landing_airport_match_count=parse_count(
                row["landing_airport_id_match_count"]
            ),
            airport_match_count=parse_count(
                row["airport_id_match_count"]
            ),
        )

def parse_count(value: int | float) -> int:
    """Convert CSV values such as 12457.0 into validated integer counts."""
    number = float(value)

    if not number.is_integer():
        raise ValueError(f"Expected an integer count, got {value!r}")

    return int(number)


def parse_optional_adsb_source(
    value: str | None,
) -> ADSBSource | None:
    if value is None or not value.strip():
        return None

    return ADSBSource(value)


def resolve_flight_source(name: str) -> FlightSource:
    try:
        return FLIGHT_SOURCES_BY_NAME[name]
    except KeyError:
        raise ValueError(f"Unknown flight source: {name!r}") from None
# Data Analysis

bts_csv_path = "/Volumes/T2-SSD/planequery/data/experiments/comparison/2026-10-03_20-56-20-039148_18884de_flights_evaluation.csv" 
sfpds_csv_path = "/Volumes/T2-SSD/planequery/data/experiments/comparison/2026-10-04_19-02-38-491861_18884de_flights_evaluation.csv"
eurocontrol_csv_path = "/Volumes/T2-SSD/planequery/data/experiments/comparison/2026-10-04_19-05-37-057977_bbf4675_flights_evaluation.csv"  # needs OpenSky and actually make it general for 3 adsb sources. 

def main(path):
    df = pl.read_csv(path)
    stats = [
        EvaluationStats.from_csv_row(row)
        for row in df.filter(
            pl.col("test_date") != "average"
        ).to_dicts()
    ]
    stats[0].evaluation_date
    stats_by_day: dict[date, list] = dict()
    for stat in stats:
        stats_by_day.setdefault(stat.evaluation_date, []).append(stat)

    metrics_by_day = dict()

    @dataclass
    class DayMetrics:
        date: date
        adsblol_stats: EvaluationStats # | None? 
        adsbx_stats: EvaluationStats

    for date_key, stats in stats_by_day.items():
        dm = DayMetrics(
            date=date_key,
            adsblol_stats=next(
                stat for stat in stats if stat.adsb_source == ADSBSource.ADSBLOL
            ),
            adsbx_stats=next(
                stat for stat in stats if stat.adsb_source == ADSBSource.ADSBX
            ),
        )
        metrics_by_day[date_key] = dm
    precision_diffs = []
    recall_diffs = []
    for dm in metrics_by_day.values():
        precision_diff = dm.adsblol_stats.precision - dm.adsbx_stats.precision
        recall_diff = dm.adsblol_stats.recall - dm.adsbx_stats.recall
        precision_diffs.append(precision_diff)
        recall_diffs.append(recall_diff)

    print("Avg Precision Diff:", sum(precision_diffs) / len(precision_diffs))
    print("Avg Recall Diff:", sum(recall_diffs) / len(recall_diffs))

print("BTS")        
main(bts_csv_path)
print("SFPDS")
main(sfpds_csv_path)
print("Eurocontrol")
main(eurocontrol_csv_path)

# TODO: Write down all dates in this file. I shoudl have 2 days for both.
