"""Compare ADSBx nearest airport with ADSB.lol model on identical labeled segments."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import add_flight_id_col
from flights_airport_nn.benchmark import evaluate_dates
from flights_airport_nn.features.features import create_features
from flights_airport_nn.features.ground_truth_join_segmentation_algorithm import (
    flights_features_df,
)
from flights_airport_nn.inference import load_model

ADSB_COLUMNS = [
    "time",
    "icao",
    "lat",
    "lon",
    "baro_altitude_ft",
    "ground_speed_kt",
    "track_deg",
    "baro_rate_fpm",
    "on_ground",
]


def rates(frame):
    n = frame.height
    covered = frame.filter(pl.col("adsbx_covered"))
    return {
        "endpoints": n,
        "adsbx_covered": covered.height,
        "adsbx_coverage": covered.height / n if n else None,
        "adsbx_nearest_accuracy_all": frame["adsbx_correct"].mean() if n else None,
        "adsblol_model_accuracy_all": frame["model_correct"].mean() if n else None,
        "adsblol_nearest_accuracy_all": frame["adsblol_nearest_correct"].mean()
        if n
        else None,
        "adsbx_nearest_accuracy_covered": covered["adsbx_correct"].mean()
        if covered.height
        else None,
        "adsblol_model_accuracy_covered": covered["model_correct"].mean()
        if covered.height
        else None,
    }


def bootstrap_difference(frame, seed=20260917, replicates=2000):
    """Paired model minus ADSBx-nearest difference, resampling aircraft."""
    groups = frame.group_by("icao").agg(
        pl.len().alias("endpoints"),
        (pl.col("model_correct").cast(pl.Int8) - pl.col("adsbx_correct").cast(pl.Int8))
        .sum()
        .alias("difference_sum"),
    )
    n = groups["endpoints"].to_numpy()
    d = groups["difference_sum"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for i in range(replicates):
        index = rng.integers(0, len(n), len(n))
        draws[i] = d[index].sum() / n[index].sum()
    return {
        "aircraft": groups.height,
        "difference": float(d.sum() / n.sum()),
        "aircraft_bootstrap_95_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "replicates": replicates,
        "seed": seed,
    }


def run(day, model_path, output, batch_aircraft=64):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    model, config = load_model(model_path)
    if day in set(model.training_dates) | set(model.validation_dates):
        raise ValueError("Comparison day overlaps model training or validation")
    flights = add_flight_id_col(flights_features_df(day))
    model_results, _, _ = evaluate_dates(
        model, [day], config, temperature=model.temperature
    )
    if flights.height * 2 != model_results.height:
        raise ValueError("Model cache and matched flight count differ")
    icaos = sorted(flights["icao"].unique().to_list())
    print(f"Reading ADSBx observations for {len(icaos)} aircraft", flush=True)
    messages = read_adsb(
        day, columns=ADSB_COLUMNS, icaos=icaos, source="adsbx", lazy=True
    ).collect(engine="streaming")
    print(f"Read {messages.height:,} ADSBx observations", flush=True)
    nearest_rows = []
    for start in range(0, len(icaos), batch_aircraft):
        block = icaos[start : start + batch_aircraft]
        subset = flights.filter(pl.col("icao").is_in(block)).with_columns(
            pl.lit(None, dtype=pl.String).alias("takeoff_airport_ident"),
            pl.lit(None, dtype=pl.String).alias("landing_airport_ident"),
        )
        tracks = messages.filter(pl.col("icao").is_in(block))
        for endpoint in create_features(subset, tracks).endpoints:
            nearest_rows.append(
                {
                    "flight_id": endpoint.flight_id,
                    "endpoint": endpoint.endpoint,
                    "adsbx_nearest": endpoint.candidates[0].predicted_airport.ident
                    if endpoint.candidates
                    else None,
                    "adsbx_endpoint_points": len(endpoint.candidates[0].features)
                    if endpoint.candidates
                    else 0,
                }
            )
        print(f"Aircraft {min(start + batch_aircraft, len(icaos))}/{len(icaos)}", flush=True)
    nearest = pl.DataFrame(nearest_rows)
    if nearest.height != model_results.height:
        raise ValueError("ADSBx and model endpoint counts differ")
    result = model_results.rename(
        {
            "prediction": "adsblol_model",
            "correct": "model_correct",
            "nearest_correct": "adsblol_nearest_correct",
        }
    ).join(nearest, on=["flight_id", "endpoint"], how="left", validate="1:1")
    if result.height != model_results.height or result["adsbx_endpoint_points"].null_count():
        raise ValueError("ADSBx/model endpoint alignment failed")
    result = result.with_columns(
        pl.col("adsbx_nearest").is_not_null().alias("adsbx_covered"),
        (pl.col("adsbx_nearest") == pl.col("truth"))
        .fill_null(False)
        .alias("adsbx_correct"),
    )
    result.write_parquet(output / "paired_predictions.parquet")
    all_metrics = rates(result)
    covered = result.filter(pl.col("adsbx_covered"))
    discordance = {
        "model_correct_adsbx_wrong": result.filter(
            pl.col("model_correct") & ~pl.col("adsbx_correct")
        ).height,
        "adsbx_correct_model_wrong": result.filter(
            pl.col("adsbx_correct") & ~pl.col("model_correct")
        ).height,
        "both_wrong": result.filter(
            ~pl.col("model_correct") & ~pl.col("adsbx_correct")
        ).height,
    }
    metrics = {
        "date": day.isoformat(),
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "definition": "Same BTS/SFDPS labels and ADSB.lol segmented flights; model uses ADSB.lol endpoint observations, nearest baseline uses ADSBx endpoint observations. No usable ADSBx airport candidate within the model's 10-minute endpoint window counts as an ADSBx failure.",
        "source_scope": "Original filtered commercial ground-truth population, not all raw flights or all airports.",
        "adsbx_observations": messages.height,
        "all": all_metrics,
        "takeoff": rates(result.filter(pl.col("endpoint") == "takeoff")),
        "landing": rates(result.filter(pl.col("endpoint") == "landing")),
        "paired_discordance": discordance,
        "covered_paired_difference": bootstrap_difference(covered),
        "all_paired_difference": bootstrap_difference(result),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    lines = [
        f"# ADSBx nearest versus ADSB.lol model, {day}",
        "",
        metrics["definition"],
        "",
        "| Cohort | Endpoints | ADSBx nearest | ADSB.lol model | ADSB.lol nearest | ADSBx coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("all", "takeoff", "landing"):
        row = metrics[key]
        lines.append(
            f"| {key} | {row['endpoints']:,} | {row['adsbx_nearest_accuracy_all']:.2%} | "
            f"{row['adsblol_model_accuracy_all']:.2%} | "
            f"{row['adsblol_nearest_accuracy_all']:.2%} | "
            f"{row['adsbx_coverage']:.2%} |"
        )
    lines += [
        "",
        (
            f"On the {covered.height:,} ADSBx-covered endpoints, ADSBx nearest was "
            f"{all_metrics['adsbx_nearest_accuracy_covered']:.2%} and the ADSB.lol model "
            f"was {all_metrics['adsblol_model_accuracy_covered']:.2%}."
        ),
        (
            f"The model alone was right on {discordance['model_correct_adsbx_wrong']} "
            f"endpoints; ADSBx nearest alone was right on "
            f"{discordance['adsbx_correct_model_wrong']}."
        ),
        "",
        metrics["source_scope"],
        "The ADSBx sample contains March 1 observations only; overnight flight endpoints before midnight may lack ADSBx data.",
        "These are agreement scores with the saved ground truth; the model and nearest baseline use different observation networks.",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-aircraft", type=int, default=64)
    args = parser.parse_args()
    run(args.date, args.model, args.output_dir, args.batch_aircraft)
