"""Compare three flight algorithms using existing OpenSky evaluation results.

Reuse the paper's grouped booktabs/longtable layout; never run evaluations.
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
    DEFAULT_OUTPUT_DIR, METADATA, RATE_COLUMNS, REFERENCE_ORDER, TABLES,
    csv_column, load_cache_rows, load_rows, prepare_rows, row_key, summary_latex,
)

METHOD_LABELS = {
    "junzis": "Sun",
    "plane-notify": "plane-notify",
    "openap": "OpenAP",
}
REFERENCE_DATES = {
    "bts": "2026-03-01",
    "sfdps": "2026-03-01",
    "eurocontrol": "2024-06-01",
}
METRICS = tuple(metric for metric in TABLES["reference_summary"].metrics
                if metric.key in {"precision", "recall", "f1",
                                  "takeoff_airport_id_match_pct",
                                  "landing_airport_id_match_pct"})
NOTE = (
    "P/R/F1 = flight segmentation precision/recall/F1; F1 is their harmonic mean. "
    "TO/LD = takeoff/landing airport match percentages among matched flights; "
    "null airport IDs are incorrect. All rates are percentages. "
    "All three methods use OpenSky ADS-B messages. Sun = junzis; "
    "OpenAP = openap. Retained flight counts can differ by method because "
    "of boundary filtering. Exact denominators are included in the CSV."
)


def select_comparisons(rows: list[dict], references: list[str]) -> list[dict]:
    """Require every method, and never accept another provider for an algorithm."""
    selected = [row for row in rows
                if row["gold_src"] in references
                and row["test_date"] == REFERENCE_DATES[row["gold_src"]]
                and row["test_src"] in METHOD_LABELS
                and row["adsb_src"] == "opensky"]
    present = set()
    for row in selected:
        key = (row["gold_src"], row["test_src"])
        if key in present:
            raise ValueError(f"Multiple comparisons for {key}; select one evaluation run.")
        present.add(key)
    expected = {(reference, method) for reference in references for method in METHOD_LABELS}
    if missing := expected - present:
        details = "\n".join(f"  {REFERENCE_DATES[reference]} | {reference} | {method}"
                            for reference, method in sorted(missing))
        raise ValueError(f"Missing {len(missing)} of {len(expected)} requested comparisons "
                         f"(date | reference | method):\n{details}\n"
                         "Run the missing OpenSky evaluations first, or supply evaluation CSVs.")
    return sorted(selected, key=lambda row: (REFERENCE_ORDER.index(row["gold_src"]),
                                            tuple(METHOD_LABELS).index(row["test_src"])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="*", type=Path,
                        help="Evaluation CSVs; relative paths use $OUTPUT_DIR/data/experiments/comparison. Default: current-version cache.")
    parser.add_argument("--gold-source", nargs="+", choices=REFERENCE_DATES,
                        default=list(REFERENCE_DATES))
    parser.add_argument("--target-day-adsb", choices=("true", "false"), default="false")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cache-policy", choices=("latest", "error"), default="latest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output parent directory (default: $OUTPUT_DIR/paper/tables).")
    parser.add_argument("--prefix", default="algorithm_comparison")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args(argv)
    if args.csvs and args.cache_dir:
        parser.error("Use either CSV inputs or --cache-dir, not both.")
    if not 0 <= args.decimals <= 6:
        parser.error("--decimals must be between 0 and 6.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prefix):
        parser.error("--prefix must contain only letters, numbers, underscores, and hyphens.")
    include = {"test_src": set(METHOD_LABELS), "gold_src": set(args.gold_source),
               "adsb_src": {"opensky"}, "target_day_adsb": {args.target_day_adsb},
               "test_date": {REFERENCE_DATES[source] for source in args.gold_source}}
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
        rows = select_comparisons(rows, args.gold_source)
        prepare_rows(rows, {"algorithm_comparison": METRICS}, require_complete=True)
        denominators = TABLES["reference_summary"].denominators
        if any(row.get(key) is None for row in rows for key in denominators):
            raise ValueError("Missing denominator counts; supply full evaluation CSVs.")
        latex = summary_latex(rows, METRICS, args.prefix, args.decimals,
                              caption="Algorithm comparison using OpenSky ADS-B",
                              note=NOTE, label_suffix="", method_labels=METHOD_LABELS)
        provenance.update(
            cache_entries=[{"path": str(entry.path.resolve()),
                            "created_at": entry.created_at.isoformat(), "key": entry.key.to_dict(),
                            "stats_sha256": hashlib.sha256((entry.path / "stats.parquet").read_bytes()).hexdigest()}
                           for row in rows if (entry := entries.get(row_key(row))) is not None],
            filters={key: sorted(values) for key, values in include.items()},
            reference_dates={source: REFERENCE_DATES[source] for source in args.gold_source},
            method_labels=METHOD_LABELS, daily_rows=len(rows), csv_percent_scale="0-100",
            latex_decimals=args.decimals, metrics=[metric.key for metric in METRICS], definition=NOTE,
        )
        output = args.output_dir.expanduser() / args.prefix
        output.mkdir(parents=True, exist_ok=True)
        csv_path = output / f"{args.prefix}.csv"
        latex_path = output / f"{args.prefix}.tex"
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
