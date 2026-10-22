"""Export a LaTeX-ready airport accuracy figure from cached evaluations only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from flights.evaluation.cache import EvaluationCache


REPO_ROOT = Path(__file__).resolve().parents[2]
STEM = "airport_accuracy_vs_distance_opensky"
MODELS = ("algorithm-nearest", "algorithm-nn")
BREAKS = [1, 2, 5, 10, 20, 50, 100]
LABELS = ["0-1", "1-2", "2-5", "5-10", "10-20", "20-50", "50-100", "100+"]
ENDPOINTS = (
    ("takeoff", "first_message_distance_to_takeoff_airport_km", "same_takeoff_airport_id"),
    ("landing", "last_message_distance_to_landing_airport_km", "same_landing_airport_id"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="Override the evaluation cache root.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "paper" / "figures" / STEM,
    )
    parser.add_argument("--us-date", type=date.fromisoformat, default=date(2026, 3, 1))
    parser.add_argument("--eurocontrol-date", type=date.fromisoformat, default=date(2024, 6, 1))
    parser.add_argument(
        "--cache-policy", choices=("latest", "error"), default="latest",
        help="Select the newest configuration, or reject ambiguous cache entries.",
    )
    return parser.parse_args()


def collect_bins(args: argparse.Namespace) -> tuple[pl.DataFrame, list[dict]]:
    cache = EvaluationCache(args.cache_dir)
    frames = []
    provenance = []
    for gold, day in (
        ("bts", args.us_date),
        ("sfdps", args.us_date),
        ("eurocontrol", args.eurocontrol_date),
    ):
        for model in MODELS:
            entries = [
                entry for entry in cache.find_stats(
                    test_source=model, gold_source=gold,
                    adsb_source="opensky", test_dates=day,
                )
                if not entry.stats.target_day_adsb
            ]
            if not entries:
                raise ValueError(f"Missing standard-window OpenSky cache: {gold}, {day}, {model}")
            if len(entries) > 1 and args.cache_policy == "error":
                raise ValueError(f"Ambiguous cache: {gold}, {day}, {model}: {len(entries)} entries")
            entry = max(entries, key=lambda item: (item.created_at, str(item.path)))
            result_path = entry.path / "result.parquet"
            matched = pl.read_parquet(result_path).filter(pl.col("match_status") == "both")
            observations = pl.concat([
                matched.select(
                    pl.lit(endpoint).alias("endpoint"),
                    pl.col(distance).alias("distance_km"),
                    pl.col(correct).alias("correct"),
                )
                for endpoint, distance, correct in ENDPOINTS
            ]).filter(
                pl.col("distance_km").is_finite()
                & (pl.col("distance_km") >= 0)
                & pl.col("correct").is_not_null()
            )
            if observations.is_empty():
                raise ValueError(f"No eligible endpoints: {gold}, {day}, {model}")
            summary = (
                observations.with_columns(
                    pl.col("distance_km")
                    .cut(BREAKS, labels=LABELS, left_closed=True)
                    .cast(pl.String).alias("distance_bin")
                )
                .group_by("endpoint", "distance_bin")
                .agg(
                    pl.len().alias("n"),
                    pl.col("correct").sum().alias("correct_count"),
                    (100 * pl.col("correct").mean()).alias("accuracy_pct"),
                )
            )
            # Explicit empty bins keep curves from bridging unavailable observations.
            summary = (
                pl.DataFrame({"bin_order": range(len(LABELS)), "distance_bin": LABELS})
                .join(pl.DataFrame({"endpoint": [item[0] for item in ENDPOINTS]}), how="cross")
                .join(summary, on=["endpoint", "distance_bin"], how="left")
                .with_columns(
                    pl.col("n", "correct_count").fill_null(0),
                    pl.lit(gold).alias("gold_source"),
                    pl.lit(day).alias("test_date"),
                    pl.lit(model).alias("test_source"),
                    pl.lit("opensky").alias("adsb_source"),
                )
                .sort("endpoint", "bin_order")
            )
            frames.append(summary)
            with result_path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            provenance.append({
                "cache_path": str(entry.path),
                "key": entry.key.to_dict(),
                "created_at": entry.created_at.isoformat(),
                "result_sha256": digest,
                "matching_configurations": len(entries),
                "matched_flights": matched.height,
                "eligible_endpoints": observations.height,
                "excluded_endpoints": 2 * matched.height - observations.height,
                "eligible_endpoints_by_type": {
                    endpoint: observations.filter(pl.col("endpoint") == endpoint).height
                    for endpoint, _, _ in ENDPOINTS
                },
            })
            print(f"{gold} {day} {model}: {observations.height:,} endpoints "
                  f"(selected newest of {len(entries)} cached configurations)")
    return pl.concat(frames), provenance


def plot_bins(binned: pl.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.84,
                        wspace=0.12, hspace=0.22)
    styles = {
        "algorithm-nearest": dict(color="#D55E00", marker="s", linestyle="--"),
        "algorithm-nn": dict(color="#0072B2", marker="o", linestyle="-"),
    }
    for row, (endpoint, _, _) in enumerate(ENDPOINTS):
        for column, gold in enumerate(("bts", "sfdps", "eurocontrol")):
            ax = axes[row, column]
            panel = binned.filter(
                (pl.col("gold_source") == gold) & (pl.col("endpoint") == endpoint)
            )
            for model in MODELS:
                rows = panel.filter(pl.col("test_source") == model).sort("bin_order")
                ax.plot(
                    rows["bin_order"].to_numpy(), rows["accuracy_pct"].to_numpy(),
                    label=model, linewidth=1.1, markersize=3,
                    markerfacecolor="white", markeredgewidth=0.8, **styles[model],
                )
            if row == 0:
                day = panel["test_date"][0]
                ax.set_title(f"{gold.upper()}\n{day}", pad=7)
            ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
            ax.set_ylim(0, 103)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", color="0.88", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(width=0.6, length=3)
        axes[row, 0].set_ylabel(f"{endpoint.title()}\nairport accuracy (%)")
    fig.supxlabel("OpenSky ADS-B endpoint distance to gold airport (km)", y=0.025, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.savefig(output_dir / f"{STEM}.pdf", metadata={
        "Title": "OpenSky ADS-B takeoff and landing airport accuracy versus gold airport distance",
        "Creator": "scripts/paper/airport_accuracy_vs_distance.py",
    })
    fig.savefig(output_dir / f"{STEM}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    binned, provenance = collect_bins(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_bins(binned, args.output_dir)
    binned.write_csv(args.output_dir / f"{STEM}.csv")
    manifest = {
        "adsb_source": "opensky", "target_day_adsb": False,
        "cache_policy": args.cache_policy,
        "bin_lower_bounds_km": [0, *BREAKS],
        "bin_upper_bounds_km": [*BREAKS, None],
        "bins_left_closed": True,
        "accuracy_definition": "100 * correct airport IDs / eligible matched endpoints; "
            "takeoff and landing evaluated separately, null predictions count as incorrect",
        "distance_definition": "First/last OpenSky ADS-B position to gold departure/arrival airport",
        "exclusions": "Unmatched flights and endpoints with missing/nonfinite/negative distance "
            "or missing correctness flag",
        "inputs": provenance,
    }
    (args.output_dir / f"{STEM}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    latex = rf"""% Requires \usepackage{{graphicx}}. Copy this folder under figures/.
% For a two-column paper, change figure to figure*.
\begin{{figure}}[t]
  \centering
  \includegraphics[width=\linewidth]{{figures/{STEM}/{STEM}.pdf}}
  \caption{{Airport assignment accuracy using OpenSky ADS-B messages against
  BTS and SFDPS on {args.us_date}, and EUROCONTROL on {args.eurocontrol_date}.
  Takeoff (top row) and landing (bottom row) are evaluated separately.
  Distance is measured from the first OpenSky ADS-B position to the gold
  takeoff airport, or from the last position to the gold landing airport.
  Accuracy is the percentage of correct airport IDs among matched flights
  in each distance bin for the indicated endpoint; missing predicted
  airports count as incorrect. Endpoints without a finite distance are excluded.
  Bins include their lower bound and exclude their upper bound; the final bin
  includes distances of at least 100 km. Input windows use the provider defaults.
  This measures airport assignment accuracy conditional on flight matching.}}
  \label{{fig:airport-accuracy-vs-distance-opensky}}
\end{{figure}}
"""
    (args.output_dir / f"{STEM}.tex").write_text(latex, encoding="utf-8")
    print(f"Saved PDF, PNG, CSV, LaTeX snippet and provenance to {args.output_dir}")


if __name__ == "__main__":
    main()
