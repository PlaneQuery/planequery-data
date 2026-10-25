"""Plot validation accuracy versus distinct training days in an airport-NN sweep.

Each point uses the checkpoint selected by minimum mean daily validation loss.
No model loading or inference is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


STEM = "airport_nn_training_curve"


def collect_models(sweep_dir: Path) -> tuple[list[dict], dict]:
    model_dirs = sorted(sweep_dir.glob("days-*"))
    if len(model_dirs) < 2:
        raise ValueError(f"Expected multiple days-* model directories in {sweep_dir}")
    rows = []
    reference = None
    reference_daily = None
    for model_dir in model_dirs:
        training = json.loads((model_dir / "training.json").read_text())
        history = json.loads((model_dir / "history.json").read_text())
        train_dates = set(training["train_dates"])
        validation_dates = set(training["validation_dates"])
        if not history or not train_dates or not validation_dates:
            raise ValueError(f"Empty history or training/validation dates: {model_dir}")
        if train_dates & validation_dates:
            raise ValueError(f"Training and validation dates overlap: {model_dir}")
        if reference is None:
            reference = training
        for key in ("network", "validation_dates", "validation_label_sources"):
            if training[key] != reference[key]:
                raise ValueError(f"Inconsistent {key} across models: {model_dir}")
        best = min(history, key=lambda record: record["validation_loss"])
        if set(best["validation"]) != validation_dates:
            raise ValueError(f"Incomplete selected-checkpoint validation: {model_dir}")
        daily = []
        for day in sorted(validation_dates):
            metrics = json.loads(
                (model_dir / "validation" / day / "metrics.json").read_text()
            )["primary"]
            if metrics != best["validation"][day]["primary"]:
                raise ValueError(f"Saved validation differs from selected epoch: {model_dir}, {day}")
            if metrics["endpoints"] <= 0:
                raise ValueError(f"Empty validation day: {model_dir}, {day}")
            daily.append(metrics)
        # All model sizes must be measured on the same validation population.
        population = [(day["endpoints"], day["nearest_accuracy"]) for day in daily]
        if reference_daily is None:
            reference_daily = population
        elif population != reference_daily:
            raise ValueError(f"Validation population differs across models: {model_dir}")
        endpoints = sum(day["endpoints"] for day in daily)
        rows.append({
            "training_days": len(train_dates),
            "best_epoch": best["epoch"],
            "epochs_run": len(history),
            "validation_loss": best["validation_loss"],
            "validation_days": len(daily),
            "validation_endpoints": endpoints,
            "nn_accuracy_pct": 100 * sum(
                day["nn_accuracy"] * day["endpoints"] for day in daily
            ) / endpoints,
            "nearest_accuracy_pct": 100 * sum(
                day["nearest_accuracy"] * day["endpoints"] for day in daily
            ) / endpoints,
            "model_dir": str(model_dir.resolve()),
        })
    rows.sort(key=lambda row: row["training_days"])
    if len({row["training_days"] for row in rows}) != len(rows):
        raise ValueError("Sweep contains duplicate training-day counts.")
    return rows, reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ["OUTPUT_DIR"]).expanduser() if os.environ.get("OUTPUT_DIR") else None
    parser.add_argument(
        "--sweep-dir", type=Path,
        default=root / "data/experiments/flights_airport_nn/airport-nn-sweep-59" if root else None,
        help="Sweep directory containing days-*/training.json, history.json and validation reports.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "paper/figures" / STEM if root else None,
        help="Output directory for PDF, PNG, SVG, CSV and a LaTeX figure snippet "
             "(default: $OUTPUT_DIR/paper/figures/airport_nn_training_curve).",
    )
    args = parser.parse_args()
    if args.sweep_dir is None or args.output_dir is None:
        parser.error("Set OUTPUT_DIR or supply both --sweep-dir and --output-dir.")
    rows, training = collect_models(args.sweep_dir.expanduser())
    validation_dates = set(training["validation_dates"])

    # JOAS uses Linux Libertine; register the TeX installation's font when available.
    font_family = "DejaVu Serif"
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich:
        font_path = subprocess.run(
            [kpsewhich, "LinLibertine_R.otf"], capture_output=True, text=True, check=False,
        ).stdout.strip()
        if font_path and Path(font_path).is_file():
            font_manager.fontManager.addfont(font_path)
            font_family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [font_family], "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # B5 paper (176 mm) minus the JOAS template's two 17 mm side margins.
    fig, ax = plt.subplots(figsize=(142 / 25.4, 65 / 25.4), layout="constrained")
    training_days = [row["training_days"] for row in rows]
    ax.plot(training_days, [row["nn_accuracy_pct"] for row in rows],
            color="#0072B2", marker="o", markersize=3.5, linewidth=1.2,
            label="Neural network")
    ax.plot(training_days, [row["nearest_accuracy_pct"] for row in rows],
            color="0.4", linestyle="--", linewidth=1.1, label="Nearest airport")
    # Regular ticks avoid overlapping the adjacent 1-day and 2-day labels.
    ax.set_xlim(0, max(training_days) + 1)
    ax.set_xlabel("Number of training days")
    ax.set_ylabel("Validation accuracy (%)")
    ax.tick_params(width=0.6, length=3)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.legend(frameon=False, loc="center right")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        path = output_dir / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=300)
        print(path.resolve())
    plt.close(fig)
    csv_path = output_dir / f"{STEM}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path.resolve())
    dates = sorted(validation_dates)
    source_names = {"BTS": "BTS", "SFDPS": "FAA SFDPS"}
    label_sources = " and ".join(
        source_names.get(source, source)
        for source in sorted(training["validation_label_sources"])
    )
    network = {"adsblol": "ADSB.lol"}.get(training["network"], training["network"])
    latex_path = output_dir / f"{STEM}.tex"
    latex_path.write_text(
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{figures/{STEM}/{STEM}.pdf}}\n"
        "  \\caption{Airport prediction accuracy versus number of training days. "
        "Each point represents a separately trained model, using the checkpoint "
        "with the lowest mean daily validation cross-entropy. "
        f"Predicted takeoff and landing airports from {network} observations "
        f"are validated against reference airport labels from {label_sources}. "
        f"Accuracy is pooled over {rows[0]['validation_endpoints']:,} endpoints "
        f"across the same {len(dates)} validation dates ({dates[0]} to {dates[-1]}), "
        "with equal weight per endpoint. "
        "The dashed line shows the nearest-airport baseline.}\n"
        "  \\label{fig:airport-nn-training-curve}\n"
        "\\end{figure}\n"
    )
    print(latex_path.resolve())
    print(f"Figure: 142 x 65 mm; font: {font_family}")


if __name__ == "__main__":
    main()
