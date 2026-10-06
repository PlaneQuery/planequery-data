from __future__ import annotations

import sys
from datetime import date

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
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
        "precision_pct_diff_adsblol_vs_adsbx": percent_change(
            left.precision,
            right.precision,
        ),
        "adsblol_recall": left.recall,
        "adsbx_recall": right.recall,
        "recall_diff_adsblol_minus_adsbx": left.recall - right.recall,
        "recall_pct_diff_adsblol_vs_adsbx": percent_change(
            left.recall,
            right.recall,
        ),
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
        rows.extend(
            comparison_row(
                stats[(test_date, LEFT_ADSB_SOURCE)],
                stats[(test_date, RIGHT_ADSB_SOURCE)],
            )
            for test_date in test_dates
        )

    pl.DataFrame(rows).write_csv(sys.stdout, float_precision=9)


if __name__ == "__main__":
    main()
