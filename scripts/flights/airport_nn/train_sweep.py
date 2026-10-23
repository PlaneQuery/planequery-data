"""Sweep chronological training prefixes, holding June 1–5 fixed for validation."""

import argparse
import csv
import json
import os
from datetime import date, timedelta
from pathlib import Path


TRAIN_DAYS = [date(2026, 2, 1) + timedelta(days=i) for i in range(59)]
VALIDATION_DAYS = [date(2026, 6, day) for day in range(1, 6)]


def summarize(output, days):
    history = json.loads((output / "history.json").read_text())
    best = min(history, key=lambda row: row["validation_loss"])
    daily = []
    for day in VALIDATION_DAYS:
        metrics = json.loads(
            (output / "validation" / str(day) / "metrics.json").read_text()
        )["primary"]
        daily.append({
            "training_days": days,
            "validation_date": str(day),
            "validation_loss": best["validation_daily_losses"][str(day)],
            **metrics,
        })
    endpoints = sum(row["endpoints"] for row in daily)
    if not endpoints:
        raise ValueError(f"No eligible validation endpoints in {output}")
    summary = {
        "training_days": days,
        "training_start": str(TRAIN_DAYS[0]),
        "training_end": str(TRAIN_DAYS[days - 1]),
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_loss": best["train_loss"],
        "validation_loss": best["validation_loss"],
        "validation_endpoints": endpoints,
        "model_path": str(output / "model.pt"),
    }
    for metric in ("nn_accuracy", "nearest_accuracy", "candidate_recall"):
        summary[metric] = sum(
            row[metric] * row["endpoints"] for row in daily if row["endpoints"]
        ) / endpoints
    return summary, daily


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("data/features/flights_airport_nn/feb-mar-june-2026"),
    )
    parser.add_argument(
        "--output", type=Path,
        help="Defaults to $OUTPUT_DIR/data/experiments/flights_airport_nn",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[1, 2, 5, 10, 15, 20, 30, 40, 50, 59],
        help="Training sizes to run (default: 1 2 5 10 15 20 30 40 50 59).",
    )
    args = parser.parse_args()
    if args.output is None:
        if not os.environ.get("OUTPUT_DIR"):
            parser.error("Set OUTPUT_DIR or pass --output")
        args.output = Path(os.environ["OUTPUT_DIR"]) / "data/experiments/flights_airport_nn"
    if args.epochs < 1 or any(n < 1 or n > 59 for n in args.sizes):
        parser.error("Epochs must be positive; training sizes must be between 1 and 59")

    from flights.airport_nn.evaluate import files_for
    from flights.airport_nn.train import train

    sizes = sorted(set(args.sizes))
    # Fail before training if any requested day has missing prepared parts.
    for day in TRAIN_DAYS[:max(sizes)] + VALIDATION_DAYS:
        if not files_for(args.cache, [day]):
            raise ValueError(f"No prepared parts for {day}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "cache": str(args.cache.resolve()),
        "train_dates": list(map(str, TRAIN_DAYS)),
        "validation_dates": list(map(str, VALIDATION_DAYS)),
        "epochs": args.epochs,
        "seed": args.seed,
    }
    config_path = args.output / "sweep.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Sweep configuration changed; choose a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    summaries, daily_rows = [], []
    for days in sizes:
        output = args.output / f"days-{days:02d}"
        if (output / "model.pt").exists():
            print(f"Reusing completed run: {output}", flush=True)
        else:
            if (output / "best.pt").exists():
                raise FileExistsError(
                    f"Interrupted run at {output}; move that directory aside to "
                    "restart this size. Completed sizes will be reused."
                )
            print(f"\n=== Training on {days} days → {output} ===", flush=True)
            train(
                cache=args.cache, output=output, epochs=args.epochs,
                seed=args.seed, train_days=TRAIN_DAYS[:days],
                validation_days=VALIDATION_DAYS,
            )
        summary, daily = summarize(output, days)
        summaries.append(summary)
        daily_rows.extend(daily)
        write_csv(args.output / "summary.csv", summaries)
        write_csv(args.output / "validation_daily.csv", daily_rows)
    print(f"\nGraph-ready results: {args.output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
