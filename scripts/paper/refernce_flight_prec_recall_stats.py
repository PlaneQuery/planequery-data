from __future__ import annotations

from datetime import date
from statistics import fmean

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
from data_engineering.utils import OUTPUT_DIR
from flights.evaluation.evaluation import evaluate
from flights.evaluation.models import EvaluationStats
from flights.sources import (
    ALGORITHM_NN_SOURCE,
    BTS_SOURCE,
    EUROCONTROL_SOURCE,
    SFDPS_SOURCE,
)


TEST_SOURCE = ALGORITHM_NN_SOURCE
LEFT_ADSB_SOURCE = ADSBSource.ADSBLOL
RIGHT_ADSB_SOURCE = ADSBSource.ADSBX
OUTPUT_PATH = (
    OUTPUT_DIR
    / "data/experiments/paper"
    / "reference_flight_precision_recall_adsb_source_comparison.csv"
)

SFDPS_DAYS = [date(2026, 2, 1), date(2026, 3, 1)]
BTS_DAYS = [*SFDPS_DAYS, date(2025, 6, 1)]
EUROCONTROL_DAYS = [
    date(2025, 6, 1),
    date(2025, 7, 1),
    date(2025, 8, 1),
]
GOLD_SOURCE_DAYS = {
    BTS_SOURCE: BTS_DAYS,
    SFDPS_SOURCE: SFDPS_DAYS,
    # EUROCONTROL_SOURCE: EUROCONTROL_DAYS,
}


def percent_change(value: float, baseline: float) -> float:
    """Return the signed percentage change from baseline to value."""
    if baseline == 0:
        return 0.0 if value == 0 else float("inf")
    return (value - baseline) / baseline * 100


def comparison_row(
    left: EvaluationStats,
    right: EvaluationStats,
) -> dict[str, str | float]:
    assert left.test_date == right.test_date
    assert left.gold_source == right.gold_source
    assert left.adsb_source == LEFT_ADSB_SOURCE
    assert right.adsb_source == RIGHT_ADSB_SOURCE
    return {
        "gold_source": left.gold_source,
        "test_date": left.test_date.isoformat(),
        "adsblol_precision": left.precision,
        "adsbx_precision": right.precision,
        "precision_diff_adsblol_minus_adsbx": (
            left.precision - right.precision
        ),
        "adsblol_recall": left.recall,
        "adsbx_recall": right.recall,
        "recall_diff_adsblol_minus_adsbx": left.recall - right.recall,
        "precision_pct_diff_adsblol_vs_adsbx": percent_change(
            left.precision,
            right.precision,
        ),
        "recall_pct_diff_adsblol_vs_adsbx": percent_change(
            left.recall,
            right.recall,
        ),
    }


def average_row(
    daily_rows: list[dict[str, str | float]],
) -> dict[str, str | float]:
    """Average all numeric columns for one gold source."""
    assert daily_rows
    return {
        column: (
            value
            if column == "gold_source"
            else "average"
            if column == "test_date"
            else fmean(float(row[column]) for row in daily_rows)
        )
        for column, value in daily_rows[0].items()
    }


def main() -> None:
    rows: list[dict[str, str | float]] = []
    for gold_source, test_dates in GOLD_SOURCE_DAYS.items():
        results = evaluate(
            test_source=TEST_SOURCE,
            gold_source=gold_source,
            test_dates=test_dates,
            adsb_source=[LEFT_ADSB_SOURCE, RIGHT_ADSB_SOURCE],
        )
        stats = {
            (item.test_date, item.adsb_source): item for item in results.stats
        }
        daily_rows = [
            comparison_row(
                stats[(test_date, LEFT_ADSB_SOURCE)],
                stats[(test_date, RIGHT_ADSB_SOURCE)],
            )
            for test_date in test_dates
        ]
        rows.extend([*daily_rows, average_row(daily_rows)])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUTPUT_PATH, float_precision=9)
    print(f"Comparison CSV saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
