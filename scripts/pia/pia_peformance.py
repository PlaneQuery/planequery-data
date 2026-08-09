"""Measure PIA resolver performance as the evaluation window grows.

Runs windows starting on 2026-01-01 with monthly end dates through 2026-07-01,
then saves the run summary and chart below OUTPUT_DIR.
"""

from datetime import date, datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from data_engineering.utils import OUTPUT_DIR
from tests.pia.test_pia import pia_tester


START_DATE = date(2026, 1, 1)
END_DATES = [
    date(2026, 2, 1),
    date(2026, 3, 1),
    date(2026, 4, 1),
    date(2026, 5, 1),
    date(2026, 6, 1),
    date(2026, 7, 1),
]


def make_performance_chart(results: pl.DataFrame, output_path: Path) -> None:
    """Plot accuracy and traceability against the end of each test window."""
    end_dates = results.get_column("end_date").to_list()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        end_dates,
        results.get_column("accuracy").to_list(),
        marker="o",
        label="Mapping accuracy",
    )
    ax.plot(
        end_dates,
        results.get_column("traceability_index").to_list(),
        marker="o",
        label="Traceability index",
    )
    ax.set_title("PIA performance by evaluation-window length")
    ax.set_xlabel("Evaluation end date (start: 2026-01-01)")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> tuple[Path, Path]:
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        OUTPUT_DIR
        / "data"
        / "expierments"
        / "pia"
        / "pia_peformance"
        / current_datetime
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for end_date in END_DATES:
        print(f"Running PIA evaluation: {START_DATE} to {end_date}")
        stats = pia_tester(
            START_DATE,
            end_date,
            correct_number=0,
            filter=False,
            test_name=f"pia_peformance_{START_DATE}_{end_date}",
        )
        rows.append({
            "start_date": START_DATE.isoformat(),
            "end_date": end_date.isoformat(),
            "window_months": end_date.month - START_DATE.month,
            **stats,
        })

    results = pl.DataFrame(rows)
    csv_path = output_dir / "result.csv"
    png_path = output_dir / "result.png"
    results.write_csv(csv_path)
    make_performance_chart(results, png_path)
    print(f"Saved PIA performance results to {csv_path}")
    print(f"Saved PIA performance chart to {png_path}")
    return csv_path, png_path


if __name__ == "__main__":
    main()
