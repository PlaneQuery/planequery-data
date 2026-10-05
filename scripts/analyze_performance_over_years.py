"""Evaluate Algorithm NN performance across several dates and plot the results."""

from datetime import UTC, date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import polars as pl

from data_engineering.utils import OUTPUT_DIR
from flights.evaluation.evaluation import evaluate
from flights.sources import ALGORITHM_NN_SOURCE, BTS_SOURCE
from planequery_data.utils import current_commit_hash


TEST_DATES = [
    date(2024, 3, 1),
    date(2024, 6, 1),
    date(2025, 3, 1),
    date(2025, 6, 1),
    date(2026, 3, 1),
    date(2026, 6, 1),
]
EUROCONTROL_TEST_DATES = TEST_DATES[:-2]  # Exclude 2026 dates, which are outside the EUROCONTROL dataset range.
EXPERIMENT_DIR = (
    OUTPUT_DIR
    / "data"
    / "experiments"
    / "algorithm-nn-adsblol-performance-over-years"
)


def make_output_dir() -> Path:
    """Create a timestamped, commit-specific directory for this run."""
    run_time = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M")
    output_dir = EXPERIMENT_DIR / f"{run_time}_{current_commit_hash()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def make_performance_chart(daily_metrics: pl.DataFrame, output_path: Path) -> None:
    """Plot flight detection and airport assignment performance over time."""
    daily_metrics = daily_metrics.sort("test_date")
    x_values = [date.fromisoformat(value) for value in daily_metrics["test_date"]]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    metric_groups = (
        (
            axes[0],
            (
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("f1", "F1"),
            ),
            "Flight detection performance",
        ),
        (
            axes[1],
            (
                ("takeoff_airport_id_match_pct", "Takeoff airport"),
                ("landing_airport_id_match_pct", "Landing airport"),
                ("airport_id_match_pct", "Both airports"),
            ),
            "Airport assignment performance",
        ),
    )

    for axis, metrics, title in metric_groups:
        for column, label in metrics:
            axis.plot(
                x_values,
                daily_metrics[column].to_list(),
                marker="o",
                label=label,
            )
        axis.set_title(title)
        axis.set_ylabel("Rate")
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Evaluation date")
    fig.autofmt_xdate()
    fig.suptitle("Algorithm NN performance over time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_raw_comparisons(
    comparisons: list[pl.DataFrame],
    test_dates: list[date],
    output_dir: Path,
) -> None:
    """Write the detailed flight comparison for each evaluation date."""
    if len(comparisons) != len(test_dates):
        raise ValueError(
            "Expected one raw comparison DataFrame per test date, but got "
            f"{len(comparisons)} comparisons for {len(test_dates)} dates."
        )

    raw_data_dir = output_dir / "raw_data"
    raw_data_dir.mkdir()
    for test_date, comparison in zip(test_dates, comparisons, strict=True):
        comparison.write_parquet(
            raw_data_dir / f"{test_date.isoformat()}_flight_comparison.parquet"
        )


def main() -> tuple[Path, Path]:
    output_dir = make_output_dir()
    results = evaluate(
        test_source=ALGORITHM_NN_SOURCE,
        gold_source=BTS_SOURCE,
        test_dates=TEST_DATES,
        write_outputs=False,
    )

    daily_metrics = results.to_frame(include_averages=False).sort("test_date")
    metrics_path = output_dir / "daily_metrics.csv"
    chart_path = output_dir / "performance_over_years.png"

    daily_metrics.write_csv(metrics_path)
    save_raw_comparisons(list(results.comparisons), TEST_DATES, output_dir)
    make_performance_chart(daily_metrics, chart_path)

    print(f"Saved experiment data to {output_dir}")
    print(f"Saved daily metrics to {metrics_path}")
    print(f"Saved performance chart to {chart_path}")
    return metrics_path, chart_path


if __name__ == "__main__":
    main()
