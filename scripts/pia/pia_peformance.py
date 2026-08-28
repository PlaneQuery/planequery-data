"""Run PIA performance sweeps over location radius, date window, or both."""

import argparse
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
DEFAULT_PIA_LOCATION_RADIUS_KM = 10
PIA_LOCATION_RADII_KM = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
END_DATES = [
    date(2026, 2, 1),
    date(2026, 3, 1),
    date(2026, 4, 1),
    date(2026, 5, 1),
    date(2026, 6, 1),
    date(2026, 7, 1),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        "--test",
        dest="sweep",
        choices=("radius", "end-date", "both"),
        default="end-date",
        help=(
            "Run the radius sweep, the end-date sweep, or their Cartesian "
            "product (default: end-date)."
        ),
    )
    return parser.parse_args()


def make_performance_chart(
    results: pl.DataFrame,
    output_path: Path,
    sweep: str,
) -> None:
    """Plot accuracy and traceability for the selected experiment dimensions."""
    x_column = "pia_location_radius_km" if sweep == "radius" else "end_date"
    x_label = (
        "PIA location radius (km)"
        if sweep == "radius"
        else "Evaluation end date (start: 2026-01-01)"
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for metric, axis, title in zip(
        ("accuracy", "traceability_index"),
        axes,
        ("Mapping accuracy", "Traceability index"),
    ):
        if sweep == "both":
            for radius in PIA_LOCATION_RADII_KM:
                series = results.filter(pl.col("pia_location_radius_km") == radius)
                axis.plot(
                    series.get_column(x_column).to_list(),
                    series.get_column(metric).to_list(),
                    marker="o",
                    label=f"{radius} km",
                )
            axis.legend(title="Radius", ncols=2)
        else:
            axis.plot(
                results.get_column(x_column).to_list(),
                results.get_column(metric).to_list(),
                marker="o",
            )
        axis.set_title(title)
        axis.set_ylabel("Rate")
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter("{x:.0%}")
        axis.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel(x_label)
    if sweep != "radius":
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(sweep: str) -> tuple[Path, Path]:
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

    radii = (
        PIA_LOCATION_RADII_KM
        if sweep in ("radius", "both")
        else [DEFAULT_PIA_LOCATION_RADIUS_KM]
    )
    end_dates = END_DATES if sweep in ("end-date", "both") else [END_DATES[-1]]

    rows = []
    for radius in radii:
        for end_date in end_dates:
            print(
                "Running PIA evaluation: "
                f"radius={radius} km, {START_DATE} to {end_date}"
            )
            stats = pia_tester(
                START_DATE,
                end_date,
                correct_number=0,
                filter=False,
                test_name=f"pia_peformance_radius_{radius}_{START_DATE}_{end_date}",
                pia_location_radius_km=radius,
            )
            rows.append({
                "sweep": sweep,
                "start_date": START_DATE.isoformat(),
                "end_date": end_date.isoformat(),
                "window_months": end_date.month - START_DATE.month,
                "pia_location_radius_km": radius,
                **stats,
            })

    results = pl.DataFrame(rows).sort(["pia_location_radius_km", "end_date"])
    csv_path = output_dir / "result.csv"
    png_path = output_dir / "result.png"
    results.write_csv(csv_path)
    make_performance_chart(results, png_path, sweep)
    print(f"Saved PIA performance results to {csv_path}")
    print(f"Saved PIA performance chart to {png_path}")
    return csv_path, png_path


if __name__ == "__main__":
    main(parse_args().sweep)
