"""Paired uncertainty and calibration for a frozen broad airport test report."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def run(report_dir):
    report_dir = Path(report_dir)
    df = pl.read_parquet(report_dir / "test_predictions.parquet").with_columns(
        (pl.col("target") >= 0).alias("retrieved"),
        ((pl.col("target") >= 0) & (pl.col("nearest") == pl.col("target"))).alias(
            "nearest_correct"
        ),
        ((pl.col("target") >= 0) & (pl.col("new") == pl.col("target"))).alias(
            "new_correct"
        ),
        pl.col("flight_id").str.split("_").list.first().alias("icao"),
    )
    rng = np.random.default_rng(20260917)
    groups = {
        "all": df,
        "large_airport": df.filter(pl.col("truth_type") == "large_airport"),
        "medium_airport": df.filter(pl.col("truth_type") == "medium_airport"),
        "small_airport": df.filter(pl.col("truth_type") == "small_airport"),
        "unseen_small_airport": df.filter(
            (pl.col("truth_type") == "small_airport") & ~pl.col("seen_airport")
        ),
    }
    for day in df["date"].unique():
        groups[str(day) + "_small"] = df.filter(
            (pl.col("date") == day) & (pl.col("truth_type") == "small_airport")
        )
    results = {}
    for name, frame in groups.items():
        if frame.is_empty():
            continue
        wins = frame.filter(pl.col("new_correct") & ~pl.col("nearest_correct")).height
        losses = frame.filter(pl.col("nearest_correct") & ~pl.col("new_correct")).height
        by_aircraft = frame.group_by("icao").agg(
            pl.len().alias("n"),
            (
                pl.col("new_correct").cast(pl.Int64)
                - pl.col("nearest_correct").cast(pl.Int64)
            )
            .sum()
            .alias("gain"),
        )
        n = by_aircraft["n"].to_numpy()
        gain = by_aircraft["gain"].to_numpy()
        draws = []
        for _ in range(20):
            ix = rng.integers(len(n), size=(100, len(n)))
            draws.extend((gain[ix].sum(1) / n[ix].sum(1)).tolist())
        results[name] = {
            "endpoints": frame.height,
            "aircraft": len(n),
            "nn_correct_nearest_wrong": wins,
            "nearest_correct_nn_wrong": losses,
            "difference": (wins - losses) / frame.height,
            "paired_aircraft_bootstrap_95_ci": np.quantile(
                draws, [0.025, 0.975]
            ).tolist(),
        }
    calibration = {}
    for name, frame in groups.items():
        if frame.is_empty():
            continue
        calibration[name] = {
            "endpoints": frame.height,
            "accuracy": frame["new_correct"].mean(),
            "mean_confidence": frame["new_confidence"].mean(),
            "ece_10_bins": sum(
                b.height * abs(b["new_correct"].mean() - b["new_confidence"].mean())
                for i in range(10)
                if not (
                    b := frame.filter(
                        (pl.col("new_confidence") * 10).floor().clip(0, 9) == i
                    )
                ).is_empty()
            )
            / frame.height,
        }
    summary = {
        "replicates": 2000,
        "seed": 20260917,
        "paired_results": results,
        "calibration": calibration,
        "limitation": "Bootstrap resamples aircraft within the observed dates. It does not address date, airport, or FAA-label uncertainty.",
    }
    (report_dir / "paired_analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    run(parser.parse_args().report_dir)
