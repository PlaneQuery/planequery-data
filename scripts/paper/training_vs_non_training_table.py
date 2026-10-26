"""Export Nearest/NN airport accuracy on training and nontraining airports.

Read evaluation caches by default, or pass *_airport_exposure.csv files.
Existing evaluator metrics are reused; missing evaluations are never run.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import re
from typing import Any

from csvs_to_latex_table import (
    DEFAULT_OUTPUT_DIR, METADATA, REFERENCE_LABELS, latex_escape,
    load_cache_rows, number, reference_sort_key, row_key, select_reference_dates,
)

METHODS = ("algorithm-nearest", "algorithm-nn")
GROUPS = ("training", "nontraining")
ENDPOINTS = ("takeoff", "landing")
RATES = ("airport_accuracy", "airport_macro_accuracy", "flight_match_coverage",
         "correct_endpoint_recall")
NOTE = (
    "TO/LD = takeoff/landing. Accuracy (%) = correct airport IDs divided by "
    "matched endpoints with known gold airport IDs; null predictions are incorrect. "
    "n gives the corresponding matched endpoint counts (TO/LD). Training airports "
    "are eligible positive labels used for NN training, combining takeoff and "
    "landing labels; validation labels are excluded. Both methods use the same "
    "training airport set. Unknown gold airport IDs are excluded. A dash denotes "
    "missing data or an undefined accuracy when n is zero. CSV counts are exact "
    "and percentages are unrounded. Drop (pp) is training minus nontraining "
    "accuracy in percentage points, calculated separately for each method and "
    "endpoint; a negative drop means higher nontraining accuracy."
)


def exposure_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (*row_key(row), row["airport_group"], row["endpoint"])


def add_accuracy_drops(rows: list[dict]) -> list[dict]:
    """Pair groups within the same comparison/endpoint before rounding rates."""
    lookup = {exposure_key(row): row for row in rows}
    enriched = []
    for row in rows:
        values = [lookup.get((*row_key(row), group, row["endpoint"]), {}).get("airport_accuracy")
                  for group in GROUPS]
        drop = None if any(value is None for value in values) else 100 * (values[0] - values[1])
        enriched.append({**row, "accuracy_drop_pp": drop})
    return enriched


def load_exposure_csvs(paths: list[Path], include: dict[str, set[str]]) -> list[dict]:
    unique = {}
    required = {*METADATA, "airport_group", "endpoint", "airport_accuracy",
                "matched_endpoint_count", "correct_count", "training_manifest_sha256"}
    text_columns = {*METADATA, "airport_group", "endpoint", "training_manifest",
                    "training_manifest_sha256"}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if missing := required - set(reader.fieldnames or []):
                raise ValueError(f"{path}: missing exposure columns {sorted(missing)}. "
                                 "Use an *_airport_exposure.csv, not the summary CSV.")
            for raw in reader:
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError(f"{path}: malformed CSV row")
                raw["target_day_adsb"] = raw["target_day_adsb"].lower()
                if any(raw[key] not in values for key, values in include.items()):
                    continue
                if raw["airport_group"] not in GROUPS:
                    continue
                date.fromisoformat(raw["test_date"])
                row = {key: value if key in text_columns else number(value)
                       for key, value in raw.items()}
                key = exposure_key(row)
                if key in unique and unique[key] != row:
                    raise ValueError(f"Conflicting exposure rows for {key}; select one run.")
                unique[key] = row
    return list(unique.values())


def validate_rows(rows: list[dict], methods: list[str], gold_sources: list[str] | None,
                  require_complete: bool) -> None:
    hashes = {row.get("training_manifest_sha256") for row in rows}
    if len(hashes) != 1 or not next(iter(hashes), None):
        raise ValueError("Selected rows must use one training manifest; do not mix training sets.")
    for row in rows:
        if row["endpoint"] not in ENDPOINTS:
            raise ValueError(f"Unknown endpoint: {row['endpoint']}")
        count, correct, accuracy = (row.get(key) for key in
                                    ("matched_endpoint_count", "correct_count", "airport_accuracy"))
        if count is None or correct is None or int(count) != count or int(correct) != correct:
            raise ValueError("Exposure rows require integer matched/correct counts.")
        if not 0 <= correct <= count:
            raise ValueError("Exposure counts must satisfy 0 <= correct <= matched.")
        if count == 0:
            if accuracy is not None:
                raise ValueError("Accuracy must be undefined when matched count is zero.")
        elif accuracy is None or abs(accuracy - correct / count) > 1e-8:
            raise ValueError("Accuracy must be a 0–1 fraction matching correct/matched counts.")
    if not require_complete:
        return
    present = {exposure_key(row) for row in rows}
    missing = []
    for row in rows:
        for method in methods:
            for group in GROUPS:
                for endpoint in ENDPOINTS:
                    expected = (*row_key({**row, "test_src": method}), group, endpoint)
                    if expected not in present:
                        missing.append(expected)
    absent_sources = set(gold_sources or []) - {row["gold_src"] for row in rows}
    if missing or absent_sources:
        raise ValueError(f"Incomplete selection. Missing reference sources: {sorted(absent_sources)}; "
                         f"missing method/group/endpoint rows: {sorted(set(missing))}")


def render_latex(rows: list[dict], prefix: str, decimals: int) -> str:
    comparisons = {row_key(row): row for row in rows}
    lookup = {exposure_key(row): row for row in rows}
    providers = {row["adsb_src"] for row in rows}
    labels = {"opensky": "OpenSky", "adsblol": "ADSB.lol", "adsbx": "ADSBX"}
    heading = (r"Method & \shortstack{Training $n$\\TO / LD} & "
               r"\shortstack{Training\\accuracy (\%)\\TO / LD} & "
               r"\shortstack{Nontraining $n$\\TO / LD} & "
               r"\shortstack{Nontraining\\accuracy (\%)\\TO / LD} & "
               r"\shortstack{Drop (pp)\\TO / LD} \\")
    lines = [r"% Requires \usepackage{booktabs,longtable}.", r"\begingroup",
             r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{@{}lrrrrr@{}}",
             r"\caption{Airport accuracy on training and nontraining airports}",
             r"\label{tab:" + prefix.replace("_", "-") + r"-training-vs-non-training}\\",
             r"\toprule", heading, r"\midrule", r"\endfirsthead",
             r"\toprule", heading, r"\midrule", r"\endhead",
             r"\midrule\multicolumn{6}{r}{Continued on next page}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    ordered = sorted(comparisons.values(), key=reference_sort_key)
    for context, group in groupby(ordered, lambda r: (r["gold_src"], r["test_date"], r["target_day_adsb"])):
        source, day, window = context
        title = f"{REFERENCE_LABELS.get(source, source)} | {day}"
        if len({row["target_day_adsb"] for row in rows}) > 1:
            title += f" | Target-day ADS-B: {window}"
        lines += [r"\addlinespace[4pt]", r"\multicolumn{6}{@{}l}{\textbf{" + latex_escape(title) + r"}}\\*"]
        for row in group:
            method = "Nearest" if row["test_src"] == METHODS[0] else "NN"
            if len(providers) > 1:
                method += " / " + labels.get(row["adsb_src"], row["adsb_src"])
            cells = [latex_escape(method)]
            for airport_group in GROUPS:
                values = [lookup.get((*row_key(row), airport_group, endpoint), {}) for endpoint in ENDPOINTS]
                cells.append(" / ".join("---" if value.get("matched_endpoint_count") is None
                                         else f"{int(value['matched_endpoint_count']):,}" for value in values))
                cells.append(" / ".join("---" if value.get("airport_accuracy") is None
                                         else f"{100 * value['airport_accuracy']:.{decimals}f}" for value in values))
            drops = [lookup.get((*row_key(row), "training", endpoint), {}).get("accuracy_drop_pp")
                     for endpoint in ENDPOINTS]
            cells.append(" / ".join("---" if drop is None else f"{drop:.{decimals}f}" for drop in drops))
            lines.append(" & ".join(cells) + r" \\")
    note = NOTE
    if len(providers) == 1:
        provider = next(iter(providers))
        note += f" {labels.get(provider, provider)} ADS-B."
    windows = {row["target_day_adsb"] for row in rows}
    if len(windows) == 1:
        note += f" Target-day-only ADS-B input: {next(iter(windows))}."
    lines += [r"\end{longtable}", r"\noindent\begin{minipage}{\linewidth}\footnotesize " +
              latex_escape(note) + r"\end{minipage}\par", r"\endgroup", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="*", type=Path, help="Exposure CSVs; relative filenames resolve under OUTPUT_DIR/data/experiments/comparison.")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cache-policy", choices=("latest", "error"), default="latest")
    parser.add_argument("--airport-nn-training", type=Path, help="Cache mode: training.json (relative to OUTPUT_DIR). Default: beside the current default NN model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output parent directory (default: $OUTPUT_DIR/paper/tables).")
    parser.add_argument("--prefix", help="Output subfolder/filename prefix; default includes the date range.")
    parser.add_argument("--reference-date", action="append", default=[], metavar="SOURCE=YYYY-MM-DD")
    parser.add_argument("--test-source", nargs="+", action="extend", choices=METHODS)
    for flag in ("gold-source", "adsb-source", "date"):
        parser.add_argument("--" + flag, nargs="+", action="extend")
    parser.add_argument("--target-day-adsb", choices=("true", "false", "unknown"), default="false")
    parser.add_argument("--require-complete", action="store_true", help="Require selected sources, both methods, and both endpoints/groups; zero denominators are allowed.")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args(argv)
    if args.csvs and (args.cache_dir or args.airport_nn_training):
        parser.error("CSV inputs already contain training membership; omit cache/training options.")
    if not 0 <= args.decimals <= 6:
        parser.error("--decimals must be between 0 and 6.")
    methods = args.test_source or list(METHODS)
    include = {"test_src": set(methods), "target_day_adsb": {args.target_day_adsb}}
    for column, values in (("gold_src", args.gold_source), ("adsb_src", args.adsb_source), ("test_date", args.date)):
        if values:
            include[column] = set(values)
    try:
        if args.output_dir is None:
            raise ValueError("Set OUTPUT_DIR or supply --output-dir for generated tables.")
        provenance: dict[str, Any] = {"input_mode": "csv" if args.csvs else "evaluation_cache"}
        if args.csvs:
            paths = [path.expanduser() for path in args.csvs]
            if any(not path.is_absolute() for path in paths):
                if not os.environ.get("OUTPUT_DIR"):
                    raise ValueError("Set OUTPUT_DIR for relative CSV filenames.")
                root = Path(os.environ["OUTPUT_DIR"]).expanduser() / "data/experiments/comparison"
                paths = [path if path.is_absolute() else root / path for path in paths]
            rows = load_exposure_csvs(paths, include)
            provenance["inputs"] = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
        else:
            if not os.environ.get("OUTPUT_DIR"):
                raise ValueError("Set OUTPUT_DIR to read evaluation caches and training metadata.")
            from flights.airport_nn import DEFAULT_MODEL_PATH
            from flights.evaluation.airport_exposure import airport_exposure_report, load_training_airports
            from flights.evaluation.cache import EvaluationCache
            from flights.evaluation.models import EvaluationResults
            cache = EvaluationCache(args.cache_dir.expanduser() if args.cache_dir else None)
            selected, entries = load_cache_rows(cache, include, args.cache_policy)
            if not selected:
                raise ValueError("No matching cached evaluations. Run evaluation first or supply exposure CSVs.")
            selected = select_reference_dates(selected, args.reference_date)
            training_path = args.airport_nn_training or DEFAULT_MODEL_PATH.resolve().with_name("training.json")
            training = load_training_airports(training_path)
            rows, cached_provenance = [], []
            for selection in selected:
                entry = entries[row_key(selection)]
                result = cache.load(entry.key)
                if result is None:
                    raise ValueError(f"Cannot load cached comparison: {entry.path}")
                report = airport_exposure_report(EvaluationResults((result,)), training)
                for row in report.to_dicts():
                    row["target_day_adsb"] = str(row["target_day_adsb"]).lower()
                    row["adsb_src"] = row["adsb_src"] or ""
                    if row["airport_group"] in GROUPS:
                        rows.append(row)
                cached_provenance.append({"path": str(entry.path), "key": entry.key.to_dict(),
                                          "created_at": entry.created_at.isoformat()})
            provenance.update(cache_root=str(cache.root), cache_policy=args.cache_policy,
                              cache_entries=cached_provenance, training_manifest=str(training.manifest_path))
        if not rows:
            raise ValueError("No matching airport exposure rows.")
        rows = select_reference_dates(rows, args.reference_date)
        validate_rows(rows, methods, args.gold_source, args.require_complete)
        rows = add_accuracy_drops(rows)
        rows.sort(key=lambda row: (*reference_sort_key(row), GROUPS.index(row["airport_group"]), ENDPOINTS.index(row["endpoint"])))
        dates = sorted({row["test_date"] for row in rows})
        prefix = args.prefix or "airport_training_" + (dates[0] if len(dates) == 1 else dates[0] + "_to_" + dates[-1])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", prefix):
            raise ValueError("--prefix must contain only letters, numbers, underscores, and hyphens.")
        latex = render_latex(rows, prefix, args.decimals)
        output = args.output_dir.expanduser() / prefix
        output.mkdir(parents=True, exist_ok=True)
        csv_path = output / f"{prefix}_training_vs_non_training.csv"
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[key + "_percent" if key in RATES else key for key in columns])
            writer.writeheader()
            for row in rows:
                writer.writerow({key + "_percent" if key in RATES else key:
                                 value * 100 if key in RATES and value is not None else
                                 int(value) if key.endswith("_count") and value is not None else value
                                 for key, value in row.items()})
        latex_path = csv_path.with_suffix(".tex")
        latex_path.write_text(latex, encoding="utf-8")
        manifest = output / f"{prefix}_training_vs_non_training_manifest.json"
        provenance.update(training_manifest_sha256=rows[0]["training_manifest_sha256"],
                          filters={key: sorted(value) for key, value in include.items()},
                          reference_dates={row["gold_src"]: row["test_date"] for row in rows},
                          csv_percent_scale="0-100", latex_decimals=args.decimals,
                          definition=NOTE, exposure_rows=len(rows),
                          outputs=[str(csv_path.resolve()), str(latex_path.resolve())])
        manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for path in (csv_path, latex_path, manifest):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
