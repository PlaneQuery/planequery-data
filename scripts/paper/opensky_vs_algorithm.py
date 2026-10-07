from datetime import date
from statistics import fmean

import polars as pl

from data_engineering.adsb.canonical import ADSBSource
from data_engineering.utils import OUTPUT_DIR
from flights.evaluation.evaluation import evaluate
from flights.evaluation.models import EvaluationStats
from flights.sources import ALGORITHM_NN_SOURCE, BTS_SOURCE, OPENSKY_SOURCE


TEST_DATES = [date(2026, 3, 1), date(2024, 6, 1)]
OUTPUT_PATH = (
    OUTPUT_DIR
    / "data/experiments/paper"
    / "opensky_vs_algorithm_precision_recall_f1.csv"
)

CONFIGS = {
    "algorithm_nn_adsblol": (
        ALGORITHM_NN_SOURCE.name,
        ADSBSource.ADSBLOL,
    ),
    "algorithm_nn_opensky": (
        ALGORITHM_NN_SOURCE.name,
        ADSBSource.OPENSKY,
    ),
    "opensky": (OPENSKY_SOURCE.name, None),
}
COMPARISONS = (
    ("algorithm_nn_adsblol", "algorithm_nn_opensky"),
    ("algorithm_nn_adsblol", "opensky"),
    ("algorithm_nn_opensky", "opensky"),
)
METRICS = ("precision", "recall", "f1")


def comparison_row(
    test_date: date,
    stats: dict[tuple[date, str, ADSBSource | None], EvaluationStats],
) -> dict[str, str | float]:
    config_stats = {
        config: stats[(test_date, test_source, adsb_source)]
        for config, (test_source, adsb_source) in CONFIGS.items()
    }
    row: dict[str, str | float] = {
        "gold_source": BTS_SOURCE.name,
        "test_date": test_date.isoformat(),
    }

    for config, item in config_stats.items():
        for metric in METRICS:
            row[f"{config}_{metric}"] = getattr(item, metric)

    for left, right in COMPARISONS:
        for metric in METRICS:
            row[f"{metric}_diff_{left}_minus_{right}"] = (
                getattr(config_stats[left], metric)
                - getattr(config_stats[right], metric)
            )
    return row


def average_row(
    daily_rows: list[dict[str, str | float]],
) -> dict[str, str | float]:
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
    results = evaluate(
        test_source=[ALGORITHM_NN_SOURCE, OPENSKY_SOURCE],
        gold_source=BTS_SOURCE,
        test_dates=TEST_DATES,
        adsb_source=[ADSBSource.ADSBLOL, ADSBSource.OPENSKY],
    )
    stats = {
        (item.test_date, item.test_source, item.adsb_source): item
        for item in results.stats
    }
    rows = [comparison_row(test_date, stats) for test_date in TEST_DATES]
    rows.append(average_row(rows))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUTPUT_PATH, float_precision=9)
    print(f"Comparison CSV saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
