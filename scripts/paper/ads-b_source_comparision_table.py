"""Build table_1_adsb_sources from existing daily evaluations (no evaluation runs).

Compare ADSB.lol, OpenSky, and ADSBX with target-day-only ADS-B: BTS/SFDPS
and verified variants on 2026-03-01; Eurocontrol variants on 2024-06-01.
Reuse the summary table's metric definitions, cache selection, and renderer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

from csvs_to_latex_table import (
    DEFAULT_OUTPUT_DIR, METADATA, RATE_COLUMNS, TABLES, csv_column,
    load_cache_rows, load_rows, prepare_rows, reference_sort_key, row_key,
    summary_latex,
)

METHODS = ("algorithm-nearest", "algorithm-nn")
PROVIDERS = ("adsblol", "opensky", "adsbx")
REFERENCE_DATES = {
    "bts": "2026-03-01", "sfdps": "2026-03-01",
    "eurocontrol": "2024-06-01", "bts-verified": "2026-03-01",
    "sfdps-verified": "2026-03-01", "eurocontrol-verified": "2024-06-01",
}
METRICS = tuple(metric for metric in TABLES["reference_summary"].metrics
                if metric.key in {"gold_flight_count", "test_flight_count",
                                  "precision", "recall", "f1",
                                  "takeoff_airport_id_match_pct",
                                  "landing_airport_id_match_pct"})
NOTE = (
    "G/T = retained gold/test flights. "
    "P/R/F1 = flight segmentation precision/recall/F1; F1 is their harmonic mean. "
    "TO/LD = takeoff/landing airport accuracy among matched flights; null airport "
    "IDs are incorrect. All rates are percentages. Nearest/NN denote "
    "algorithm-nearest/algorithm-nn. OpenSky denotes the ADS-B message provider. "
    "Retained flight counts can differ across providers and methods because of "
    "boundary filtering. Exact denominators are included in the CSV."
)


def validate_comparisons(rows: list[dict], methods: list[str]) -> None:
    """Require every requested reference/date/provider/method before writing."""
    present = {row_key(row) for row in rows}
    expected = {(day, reference, provider, "true", method)
                for reference, day in REFERENCE_DATES.items()
                for provider in PROVIDERS for method in methods}
    if missing := expected - present:
        details = "\n".join("  " + " | ".join(key) for key in sorted(missing))
        raise ValueError(f"Missing {len(missing)} of {len(expected)} requested comparisons "
                         f"(date | reference | ADS-B | target-day | method):\n{details}\n"
                         "Run the missing evaluations first, or supply explicit evaluation CSVs.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="*", type=Path,
                        help="Evaluation CSVs; relative paths use OUTPUT_DIR/data/experiments/comparison. Default: current-version cache.")
    parser.add_argument("--test-source", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cache-policy", choices=("latest", "error"), default="latest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output parent directory (default: $OUTPUT_DIR/paper/tables).")
    parser.add_argument("--prefix", default="table_1_adsb_sources")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args(argv)
    if args.csvs and args.cache_dir:
        parser.error("Use either CSV inputs or --cache-dir, not both.")
    if not 0 <= args.decimals <= 6:
        parser.error("--decimals must be between 0 and 6.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prefix):
        parser.error("--prefix must contain only letters, numbers, underscores, and hyphens.")
    include = {"test_src": set(args.test_source), "gold_src": set(REFERENCE_DATES),
               "adsb_src": set(PROVIDERS), "target_day_adsb": {"true"},
               "test_date": set(REFERENCE_DATES.values())}
    try:
        if args.output_dir is None:
            raise ValueError("Set OUTPUT_DIR or supply --output-dir for generated tables.")
        provenance = {"input_mode": "csv" if args.csvs else "evaluation_cache"}
        entries = {}
        if args.csvs:
            paths = [path.expanduser() for path in args.csvs]
            if any(not path.is_absolute() for path in paths):
                if not os.environ.get("OUTPUT_DIR"):
                    raise ValueError("Set OUTPUT_DIR for relative CSV filenames.")
                root = Path(os.environ["OUTPUT_DIR"]).expanduser() / "data/experiments/comparison"
                paths = [path if path.is_absolute() else root / path for path in paths]
            rows = load_rows(paths, include=include)
            provenance["inputs"] = [{"path": str(path.resolve()),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                    for path in paths]
        else:
            if not os.environ.get("OUTPUT_DIR"):
                raise ValueError("Set OUTPUT_DIR to read evaluation caches.")
            from flights.evaluation.cache import EvaluationCache
            cache = EvaluationCache(args.cache_dir.expanduser() if args.cache_dir else None)
            rows, entries = load_cache_rows(cache, include, args.cache_policy)
            provenance.update(cache_root=str(cache.root.resolve()), cache_policy=args.cache_policy)
        rows = [row for row in rows if row["test_date"] == REFERENCE_DATES[row["gold_src"]]]
        rows.sort(key=reference_sort_key)
        validate_comparisons(rows, args.test_source)
        prepare_rows(rows, {"adsb_sources": METRICS}, require_complete=True)
        # Preserve all exact denominators in the CSV, including matched flights.
        denominators = TABLES["reference_summary"].denominators
        if any(row.get(key) is None for row in rows for key in denominators):
            raise ValueError("Missing denominator counts; supply full evaluation CSVs.")
        latex = summary_latex(rows, METRICS, args.prefix, args.decimals,
                              caption="Flight segmentation and airport accuracy by ADS-B source",
                              note=NOTE, label_suffix="")
        provenance.update(
            cache_entries=[{"path": str(entry.path.resolve()),
                            "created_at": entry.created_at.isoformat(), "key": entry.key.to_dict(),
                            "stats_sha256": hashlib.sha256((entry.path / "stats.parquet").read_bytes()).hexdigest()}
                           for row in rows if (entry := entries.get(row_key(row))) is not None],
            filters={key: sorted(values) for key, values in include.items()},
            reference_dates=REFERENCE_DATES, daily_rows=len(rows), csv_percent_scale="0-100",
            latex_decimals=args.decimals, metrics=[metric.key for metric in METRICS], definition=NOTE,
        )
        output = args.output_dir.expanduser() / args.prefix
        output.mkdir(parents=True, exist_ok=True)
        csv_path = output / f"{args.prefix}.csv"
        latex_path = output / f"{args.prefix}.latex"
        manifest = output / f"{args.prefix}_manifest.json"
        columns = list(dict.fromkeys([*METADATA, *denominators, *(metric.key for metric in METRICS)]))
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[csv_column(key) for key in columns])
            writer.writeheader()
            for row in rows:
                writer.writerow({csv_column(key): row[key] * 100 if key in RATE_COLUMNS else
                                 row[key] if key in METADATA else int(row[key]) for key in columns})
        latex_path.write_text(latex, encoding="utf-8")
        provenance["outputs"] = [str(path.resolve()) for path in (csv_path, latex_path)]
        manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Exported all {len(rows)} requested comparisons:")
    for path in (csv_path, latex_path, manifest):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
