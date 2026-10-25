"""Export daily evaluation CSVs as numeric CSVs and booktabs/longtable LaTeX.

Run with --help; see scripts/paper/README.md for metric definitions and examples.
Reads the evaluation cache by default, or explicit CSV inputs. No evaluations
or network requests are made. CSV mode uses only the standard library.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
from itertools import groupby
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

DEFAULT_OUTPUT_DIR = (Path(os.environ["OUTPUT_DIR"]).expanduser() / "paper/tables"
                      if os.environ.get("OUTPUT_DIR") else None)
METADATA = ("test_date", "gold_src", "adsb_src", "target_day_adsb", "test_src")


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    kind: str = "count"
    rate: str | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.key, self.rate) if self.rate else (self.key,)


@dataclass(frozen=True)
class Table:
    title: str
    metrics: tuple[Metric, ...]
    note: str
    denominators: tuple[str, ...] = ()


TABLES = {
    "reference_summary": Table(
        "Flight evaluation by reference dataset",
        (
            Metric("gold_flight_count", "Gold flights"),
            Metric("test_flight_count", "Test flights"),
            Metric("gold_icao_count", "Gold ICAOs"),
            Metric("precision", "Precision (%)", "percent"),
            Metric("recall", "Recall (%)", "percent"),
            Metric("f1", "F1 (%)", "percent"),
            Metric("gold_icao_missing_from_test_pct", "Zero flights (%)", "percent"),
            Metric("takeoff_airport_id_match_pct", "Takeoff accuracy (%)", "percent"),
            Metric("landing_airport_id_match_pct", "Landing accuracy (%)", "percent"),
            Metric("takeoff_airport_id_null_pct", "Takeoff null (%)", "percent"),
            Metric("landing_airport_id_null_pct", "Landing null (%)", "percent"),
            Metric("takeoff_time_bias_minutes", "Takeoff bias (min)", "minutes"),
            Metric("landing_time_bias_minutes", "Landing bias (min)", "minutes"),
        ),
        "G/T = retained gold/test flights; P/R/F1 = segmentation precision/recall/F1; "
        "F1 is the harmonic mean of precision and recall. "
        "TO/LD = takeoff/landing. All rates are percentages. Zero flights is the "
        "percentage of gold ICAOs with no test flights. Airport accuracy divides "
        "by matched flights; null airports divide by T. Matched flights = G times "
        "R/100, false negatives = G minus matched, and false positives = T minus "
        "matched. Other counts follow from their percentages and denominators "
        "(approximately, due to rounding). G varies by method because of boundary "
        "filtering. Bias is mean signed test-minus-gold time among matched flights; "
        "negative is early. Nearest/NN denote algorithm-nearest/algorithm-nn. "
        "OpenSky denotes the published "
        "flight dataset; OpenSky-NN uses those same flights with NN airport predictions. "
        "Exact denominators are included in the CSV.",
        ("gold_icao_count", "gold_flight_count", "test_flight_count", "true_positive"),
    ),
    "segmentation": Table(
        "Flight segmentation",
        (
            Metric("true_positive", "TP"),
            Metric("false_positive", "FP"),
            Metric("false_negative", "FN"),
            Metric("precision", "Precision (%)", "percent"),
            Metric("recall", "Recall (%)", "percent"),
            Metric("f1", "F1 (%)", "percent"),
        ),
        "TP = matched flights; FP = unmatched test flights; FN = unmatched gold flights. "
        "Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 is their harmonic mean. "
        "Counts use the retained gold-aircraft cohort after boundary filtering.",
        ("test_flight_count", "gold_flight_count"),
    ),
    "aircraft_coverage": Table(
        "Gold-aircraft coverage",
        (
            Metric("gold_icao_count", "Gold ICAOs"),
            Metric("gold_icao_missing_from_test_count", "Zero test flights\nn (%)",
                   rate="gold_icao_missing_from_test_pct"),
        ),
        "Distinct daily gold ICAOs with zero test flights, as a count and percentage "
        "of all gold ICAOs. Coverage is measured before flight matching/boundary filtering.",
    ),
    "airport_accuracy": Table(
        "Airport identification accuracy and missing outputs",
        (
            Metric("takeoff_airport_id_match_count", "Takeoff correct\nn (%)",
                   rate="takeoff_airport_id_match_pct"),
            Metric("landing_airport_id_match_count", "Landing correct\nn (%)",
                   rate="landing_airport_id_match_pct"),
            Metric("airport_id_match_count", "Both correct\nn (%)", rate="airport_id_match_pct"),
            Metric("takeoff_airport_id_null_count", "Takeoff null\nn (%)",
                   rate="takeoff_airport_id_null_pct"),
            Metric("landing_airport_id_null_count", "Landing null\nn (%)",
                   rate="landing_airport_id_null_pct"),
        ),
        "Correct-airport percentages divide by matched flights (TP); null IDs are incorrect. "
        "Both correct requires both airport IDs to match. Null-output percentages divide "
        "by all retained test flights (TP+FP), including unmatched test flights.",
        ("true_positive", "test_flight_count"),
    ),
    "timing_accuracy": Table(
        "Takeoff and landing timing accuracy",
        (
            Metric("true_positive", "Matched\nn"),
            Metric("takeoff_time_bias_minutes", "Takeoff bias\n(min)", "minutes"),
            Metric("landing_time_bias_minutes", "Landing bias\n(min)", "minutes"),
            Metric("median_max_endpoint_time_error_minutes", "Max error\nmedian (min)", "minutes"),
            Metric("p90_max_endpoint_time_error_minutes", "Max error\nP90 (min)", "minutes"),
            Metric("both_endpoints_within_5m_recall", "Within 5 min\nrecall (%)", "percent"),
            Metric("both_endpoints_within_15m_recall", "Within 15 min\nrecall (%)", "percent"),
        ),
        "Bias = mean signed test-minus-gold time difference among matched flights; "
        "negative is early, positive is late. Bias is not mean absolute error. "
        "Max error is the larger absolute takeoff/landing error for each matched flight. "
        "Threshold recall requires both endpoints within the threshold and divides by "
        "all retained gold flights (TP+FN). Matching itself limits each endpoint error "
        "to one hour. A dash denotes an unavailable or undefined value.",
        ("gold_flight_count",),
    ),
}
# These metric families depend on the flight segmentation, not on the airport
# assignment method. Nearest/NN share them for the same ADS-B input;
# OpenSky/OpenSky-NN share the published flight boundaries.
# Airport-dependent metrics (including fully-correct flight scores) stay separate.
SHARED_FLIGHT_METRICS = {
    column
    for name in ("segmentation", "aircraft_coverage", "timing_accuracy")
    for column in (
        *TABLES[name].denominators,
        *(column for metric in TABLES[name].metrics for column in metric.columns),
    )
}
REFERENCE_ORDER = ("bts", "sfdps", "eurocontrol", "bts-verified", "sfdps-verified", "eurocontrol-verified")
# These sources have no selectable ADS-B provider in evaluation metadata.
# Keep the CSV exporter usable with only the standard library.
PROVIDER_INDEPENDENT_SOURCES = {*REFERENCE_ORDER, "opensky", "opensky-nn", "adsbx", "mike"}
REFERENCE_LABELS = {
    "bts": "BTS", "bts-verified": "BTS (verified)",
    "sfdps": "SFDPS", "sfdps-verified": "SFDPS (verified)",
    "eurocontrol": "Eurocontrol", "eurocontrol-verified": "Eurocontrol (verified)",
}
SUMMARY_COLUMNS = (
    ("Flights", "G / T", ("gold_flight_count", "test_flight_count")),
    ("Segmentation", "P / R / F1 (%)", ("precision", "recall", "f1")),
    ("Zero flights", "ICAOs (%)", ("gold_icao_missing_from_test_pct",)),
    ("Airport acc.", "TO / LD (%)", ("takeoff_airport_id_match_pct", "landing_airport_id_match_pct")),
    ("Null airports", "TO / LD (%)", ("takeoff_airport_id_null_pct", "landing_airport_id_null_pct")),
    ("Time bias", "TO / LD (min)", ("takeoff_time_bias_minutes", "landing_time_bias_minutes")),
)
RATE_COLUMNS = {
    column for table in TABLES.values() for metric in table.metrics
    for column in ((metric.key,) if metric.kind == "percent" else (metric.rate,))
    if column is not None
}


def row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in METADATA)


def select_reference_dates(rows: list[dict[str, Any]], overrides: list[str]) -> list[dict[str, Any]]:
    """Choose one shared date per reference family, without pooling daily metrics."""
    requested = {}
    for item in overrides:
        source, separator, day = item.partition("=")
        if not separator or not source:
            raise ValueError("--reference-date expects SOURCE=YYYY-MM-DD")
        date.fromisoformat(day)
        family = source.removesuffix("-verified")
        if family in requested and requested[family] != day:
            raise ValueError(f"Conflicting dates for reference family {family}")
        requested[family] = day
    families: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        source = row["gold_src"]
        families.setdefault(source.removesuffix("-verified"), {}).setdefault(source, set()).add(row["test_date"])
    unknown = requested.keys() - families.keys()
    if unknown:
        raise ValueError(f"No rows for requested reference families: {sorted(unknown)}")
    chosen = {}
    for family, sources in families.items():
        common = set.intersection(*sources.values())
        if not common:
            raise ValueError(f"No shared date for {family} and its verified variant.")
        chosen[family] = requested.get(family, max(common))
        if chosen[family] not in common:
            raise ValueError(f"Date {chosen[family]} unavailable for every {family} variant; available: {sorted(common)}")
    return [row for row in rows if row["test_date"] == chosen[row["gold_src"].removesuffix("-verified")]]


def reference_sort_key(row: dict[str, Any]) -> tuple:
    source = row["gold_src"]
    rank = REFERENCE_ORDER.index(source) if source in REFERENCE_ORDER else len(REFERENCE_ORDER)
    # The published OpenSky dataset has no ADS-B provider metadata, but belongs
    # beside the methods evaluated on OpenSky messages in the paper table.
    provider = row["adsb_src"] or ("opensky" if row["test_src"] in {"opensky", "opensky-nn"} else "")
    provider_rank = {"adsblol": 0, "opensky": 1, "adsbx": 2}.get(provider, 3)
    method_rank = {"opensky": 0, "opensky-nn": 1, "algorithm-nearest": 2, "algorithm-nn": 3}.get(row["test_src"], 4)
    return (rank, source, row["test_date"], row["target_day_adsb"], provider_rank, provider, method_rank, row["test_src"])


def validate_completeness(rows: list[dict[str, Any]], include: dict[str, set[str]]) -> None:
    """Require the selected method/reference/provider grid for every selected day."""
    methods = include.get("test_src", {row["test_src"] for row in rows})
    references = include.get("gold_src", {row["gold_src"] for row in rows})
    absent_references = references - {row["gold_src"] for row in rows}
    if absent_references:
        raise ValueError("Missing evaluations for requested reference sources: " +
                         ", ".join(sorted(absent_references)) + ". Run those evaluations first.")
    providers = include.get("adsb_src", {row["adsb_src"] for row in rows}) - {""}
    present = {row_key(row) for row in rows}
    missing = []
    for reference in sorted(references):
        contexts = {(row["test_date"], row["target_day_adsb"]) for row in rows
                    if row["gold_src"] == reference}
        if "test_date" in include:
            windows = include.get("target_day_adsb", {window for _, window in contexts})
            contexts = {(day, window) for day in include["test_date"] for window in windows}
        for day, window in sorted(contexts):
            for method in sorted(methods):
                pair_providers = ({""} if method in PROVIDER_INDEPENDENT_SOURCES
                                  and reference in PROVIDER_INDEPENDENT_SOURCES
                                  else providers or {"<ADS-B provider required>"})
                for provider in sorted(pair_providers):
                    key = (day, reference, provider, window, method)
                    if key not in present:
                        missing.append(" | ".join((day, reference, provider or "none", window, method)))
    if missing:
        raise ValueError(f"Missing {len(missing)} requested evaluations "
                         "(date | reference | ADS-B | target-day | method):\n  " +
                         "\n  ".join(missing) + "\nRun the missing evaluations first. "
                         "Cache mode accepts only the current evaluation cache version.")


def number(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return value
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric: {value}")
    return result


def load_rows(paths: list[Path], *, include: dict[str, set[str]] | None = None) -> list[dict[str, Any]]:
    """Filter daily rows, then deduplicate and reject conflicting selected results."""
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = set(METADATA) - {"target_day_adsb"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: missing metadata columns {sorted(required - set(reader.fieldnames or []))}")
            for line, raw in enumerate(reader, 2):
                if raw["test_date"] == "average":
                    continue
                if None in raw:
                    raise ValueError(f"{path}:{line}: malformed CSV row")
                date.fromisoformat(raw["test_date"])
                if not raw["test_src"] or not raw["gold_src"]:
                    raise ValueError(f"{path}:{line}: empty source name")
                row = {key: (value or "" if key in METADATA else number(value)) for key, value in raw.items()}
                row["target_day_adsb"] = raw.get("target_day_adsb", "unknown").lower()
                if row["target_day_adsb"] not in {"true", "false", "unknown"}:
                    raise ValueError(f"{path}:{line}: invalid target_day_adsb")
                if any(row[column] not in values for column, values in (include or {}).items()):
                    continue
                key = row_key(row)
                if key in unique and unique[key] != row:
                    raise ValueError(f"Conflicting daily rows for {key}; select one evaluation run.")
                unique[key] = row
    return sorted(unique.values(), key=row_key)


def load_cache_rows(cache: Any, include: dict[str, set[str]], policy: str = "latest") -> tuple[list[dict[str, Any]], dict]:
    """Read typed stats through the evaluator's cache API, without evaluating flights."""
    entries = cache.find_stats(
        test_source=sorted(include["test_src"]) if "test_src" in include else None,
        gold_source=sorted(include["gold_src"]) if "gold_src" in include else None,
        test_dates=[date.fromisoformat(day) for day in sorted(include["test_date"])] if "test_date" in include else None,
    )
    selected = {}
    for entry in entries:
        row = entry.stats.to_row()
        row["target_day_adsb"] = str(row["target_day_adsb"]).lower()
        row["adsb_src"] = row["adsb_src"] or ""
        if any(row[column] not in values for column, values in include.items()):
            continue
        key = row_key(row)
        if key in selected:
            previous = selected[key][1]
            if policy == "error":
                raise ValueError(f"Multiple cached runs for {key}: {previous.path} and {entry.path}. Use --cache-policy latest or supply an explicit evaluation CSV.")
            if previous.created_at == entry.created_at and previous.path != entry.path:
                raise ValueError(f"Cached runs have the same creation time for {key}; supply an explicit evaluation CSV.")
            if previous.created_at > entry.created_at:
                continue
        selected[key] = (row, entry)
    return ([selected[key][0] for key in sorted(selected)],
            {key: selected[key][1] for key in selected})


def csv_column(column: str) -> str:
    if column in RATE_COLUMNS:
        return column.removesuffix("_pct") + "_percent"
    return column


def latex_escape(value: Any) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


def format_metric(row: dict[str, Any], metric: Metric, decimals: int) -> str:
    value = row.get(metric.key)
    if value is None:
        return "---"
    if metric.kind == "count":
        formatted = f"{int(value):,}"
    else:
        formatted = f"{value * (100 if metric.kind == 'percent' else 1):.{decimals}f}"
    if metric.rate:
        rate = row.get(metric.rate)
        formatted += " (---)" if rate is None else f" ({100 * rate:.{decimals}f}\\%)"
    return formatted


def summary_latex(rows: list[dict[str, Any]], metrics: tuple[Metric, ...],
                  prefix: str, decimals: int, *,
                  caption: str = "Flight evaluation by reference dataset",
                  note: str | None = None,
                  label_suffix: str = "-reference-summary",
                  method_labels: dict[str, str] | None = None) -> str:
    """One compact table: reference datasets as groups, methods/providers as rows."""
    by_key = {metric.key: metric for metric in metrics}
    columns = []
    for title, subtitle, keys in SUMMARY_COLUMNS:
        active = tuple(key for key in keys if key in by_key)
        if active:
            label = (title, subtitle) if active == keys else (by_key[active[0]].label,)
            if title == "Segmentation":
                abbreviations = {"precision": "P", "recall": "R", "f1": "F1"}
                label = (title, " / ".join(abbreviations[key] for key in active) + " (%)")
            columns.append((label, active))
    width = len(columns) + 1
    providers = {row["adsb_src"] for row in rows if row["adsb_src"]}
    show_provider = len(providers) > 1
    provider_labels = {"adsblol": "ADSB.lol", "opensky": "OpenSky", "adsbx": "ADSBX"}
    headers = ["Method / ADS-B" if show_provider else "Method"] + [
        r"\shortstack{" + r"\\".join(latex_escape(line) for line in label) + "}"
        for label, _ in columns
    ]
    heading = " & ".join(headers) + r" \\"
    lines = [
        "% Requires \\usepackage{booktabs,longtable,multirow}.",
        r"\begingroup", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{@{}l" + "r" * len(columns) + "@{}}",
        r"\caption{" + latex_escape(caption) + "}",
        r"\label{tab:" + prefix.replace("_", "-") + label_suffix + r"}\\",
        r"\toprule", heading, r"\midrule", r"\endfirsthead",
        r"\multicolumn{" + str(width) + r"}{l}{" + latex_escape(caption) + r" (continued)}\\",
        r"\toprule", heading, r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{" + str(width) + r"}{r}{Continued on next page}\\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
    ]
    windows = {row["target_day_adsb"] for row in rows}
    has_shared_cells = False
    for context, group in groupby(rows, key=lambda row: (row["gold_src"], row["test_date"], row["target_day_adsb"])):
        source, day, window = context
        group_rows = list(group)
        icao_counts = {row.get("gold_icao_count") for row in group_rows}
        if len(icao_counts) != 1:
            raise ValueError(f"Gold ICAO totals differ within {source} on {day}; use a consistent cohort.")
        total = next(iter(icao_counts))
        total_text = "unavailable" if total is None else f"{int(total):,}"
        title = f"{REFERENCE_LABELS.get(source, source)} | {day} | Gold ICAOs: {total_text}"
        if len(windows) > 1:
            title += f" | Target-day ADS-B: {window}"
        lines.extend([r"\addlinespace[4pt]", r"\multicolumn{" + str(width) + r"}{@{}l}{\textbf{" + latex_escape(title) + r"}}\\*"])
        shared_starts: dict[int, set[int]] = {}
        shared_ends: dict[int, set[int]] = {}
        for index, (base, nn) in enumerate(zip(group_rows, group_rows[1:])):
            pair = (base["test_src"], nn["test_src"])
            algorithm_pair = (pair == ("algorithm-nearest", "algorithm-nn")
                              and base["adsb_src"] == nn["adsb_src"])
            opensky_pair = (pair == ("opensky", "opensky-nn")
                            and base["adsb_src"] in {"", "opensky"}
                            and nn["adsb_src"] in {"", "opensky"})
            if not (algorithm_pair or opensky_pair):
                continue
            shared = {
                column_index for column_index, (_, keys) in enumerate(columns)
                if set(keys) <= SHARED_FLIGHT_METRICS
                and all(base.get(key) is not None and base[key] == nn.get(key) for key in keys)
            }
            if shared:
                shared_starts[index] = shared
                shared_ends[index + 1] = shared
                has_shared_cells = True
        for index, row in enumerate(group_rows):
            method = {"algorithm-nearest": "Nearest", "algorithm-nn": "NN",
                      "opensky": "OpenSky", "opensky-nn": "OpenSky-NN"}.get(row["test_src"], row["test_src"])
            method = (method_labels or {}).get(row["test_src"], method)
            if show_provider and row["adsb_src"]:
                provider = provider_labels.get(row["adsb_src"], row["adsb_src"])
                method += " / " + provider
            cells = [latex_escape(method)]
            for column_index, (_, keys) in enumerate(columns):
                cell = " / ".join(format_metric(row, by_key[key], decimals) for key in keys)
                if column_index in shared_starts.get(index, set()):
                    cell = r"\multirow{2}{*}{" + cell + "}"
                elif column_index in shared_ends.get(index, set()):
                    cell = ""
                cells.append(cell)
            # Keep the two methods on the same page when they share cells.
            lines.append(" & ".join(cells) + (r" \\*" if index in shared_starts else r" \\"))
    note = TABLES["reference_summary"].note if note is None else note
    if has_shared_cells:
        note += " Cells spanning a method pair report values shared by both methods."
    if len(providers) == 1:
        provider = next(iter(providers))
        note += f" {provider_labels.get(provider, provider)} ADS-B."
    elif show_provider:
        note += " Method suffixes identify the ADS-B provider."
    if len(windows) == 1:
        note += f" Target-day-only ADS-B input: {next(iter(windows))}."
    lines.extend([r"\end{longtable}", r"\noindent\begin{minipage}{\linewidth}\footnotesize " +
                  latex_escape(note) + r"\end{minipage}\par", r"\endgroup", ""])
    return "\n".join(lines)


def latex_table(rows: list[dict[str, Any]], name: str, table: Table,
                metrics: tuple[Metric, ...], prefix: str, decimals: int) -> str:
    count = 1 + len(metrics)
    headers = ["Method"] + [
        r"\shortstack{" + r"\\".join(latex_escape(line) for line in metric.label.splitlines()) + "}"
        for metric in metrics
    ]
    heading = " & ".join(headers) + r" \\"
    lines = [
        "% Requires \\usepackage{booktabs,longtable}. Use in a single-column/full-width area.",
        r"\begingroup", r"\small", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}l" + "r" * len(metrics) + "@{}}",
        r"\caption{" + latex_escape(table.title) + "}",
        r"\label{tab:" + prefix.replace("_", "-") + "-" + name.replace("_", "-") + r"}\\",
        r"\toprule", heading, r"\midrule", r"\endfirsthead",
        r"\multicolumn{" + str(count) + r"}{l}{\small " + latex_escape(table.title) + r" (continued)}\\",
        r"\toprule", heading, r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{" + str(count) + r"}{r}{Continued on next page}\\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
    ]
    for context, group in groupby(rows, key=lambda row: row_key(row)[:-1]):
        day, gold, adsb, window = context
        title = f"{REFERENCE_LABELS.get(gold, gold)} | {day} | ADS-B: {adsb or 'N/A'} | Target-day ADS-B: {window}"
        lines.extend([r"\addlinespace[3pt]", r"\multicolumn{" + str(count) + r"}{@{}l}{\textbf{" + latex_escape(title) + r"}}\\*",
                      r"\nopagebreak"])
        for row in group:
            lines.append(" & ".join([latex_escape(row["test_src"])] + [
                format_metric(row, metric, decimals) for metric in metrics
            ]) + r" \\")
    lines.extend([r"\end{longtable}", r"\noindent\begin{minipage}{\linewidth}\footnotesize " +
                  latex_escape(table.note) + r"\end{minipage}\par", r"\endgroup", ""])
    return "\n".join(lines)


def prepare_rows(rows: list[dict[str, Any]], selected: dict[str, tuple[Metric, ...]],
                 require_complete: bool) -> list[str]:
    missing = set()
    for row in rows:
        for name, other in (("test_flight_count", "false_positive"), ("gold_flight_count", "false_negative")):
            row[name] = (row["true_positive"] + row[other]
                         if row.get("true_positive") is not None and row.get(other) is not None else None)
        for name in selected:
            for column in TABLES[name].denominators if name in TABLES else ():
                value = row.get(column)
                if value is None:
                    missing.add(column)
                elif not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or not float(value).is_integer():
                    raise ValueError(f"Invalid denominator count {column}: {value!r}")
        for metrics in selected.values():
            for metric in metrics:
                for column in metric.columns:
                    value = row.get(column)
                    if value is None:
                        # Matched-only timing statistics are undefined when TP is zero.
                        if not (metric.kind == "minutes" and row.get("true_positive") == 0):
                            missing.add(column)
                        continue
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"Invalid numeric metric {column}: {value!r}")
                    if column in RATE_COLUMNS and not 0 <= value <= 1:
                        raise ValueError(f"Expected a fraction in [0, 1] for {column}: {value}")
                    if metric.kind == "count" and column == metric.key and (value < 0 or not float(value).is_integer()):
                        raise ValueError(f"Invalid daily count {column}: {value}")
                    if metric.kind == "minutes" and "bias" not in column and value < 0:
                        raise ValueError(f"Negative absolute time error {column}: {value}")
    warnings = []
    if missing:
        message = "Unavailable metrics (blank CSV / dash in LaTeX): " + ", ".join(sorted(missing))
        if require_complete:
            raise ValueError(message + ". Supply full evaluation CSVs.")
        warnings.append(message)
    return warnings


def export_tables(rows: list[dict[str, Any]], output_dir: Path, prefix: str,
                  selected: dict[str, tuple[Metric, ...]], decimals: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, metrics in selected.items():
        table = TABLES[name]
        columns = list(dict.fromkeys([*METADATA, *table.denominators,
                                     *(column for metric in metrics for column in metric.columns)]))
        csv_path = output_dir / f"{prefix}_{name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[csv_column(column) for column in columns])
            writer.writeheader()
            for row in rows:
                values = {}
                for column in columns:
                    value = row.get(column)
                    if value is not None and column not in METADATA:
                        value = value * 100 if column in RATE_COLUMNS else value
                        if column not in RATE_COLUMNS and (column.endswith("_count") or column in {"true_positive", "false_positive", "false_negative"}):
                            value = int(value)
                    values[csv_column(column)] = value
                writer.writerow(values)
        latex_path = csv_path.with_suffix(".latex")
        latex = (summary_latex(rows, metrics, prefix, decimals) if name == "reference_summary"
                 else latex_table(rows, name, table, metrics, prefix, decimals))
        latex_path.write_text(latex, encoding="utf-8")
        outputs.extend([csv_path, latex_path])
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "csvs", nargs="*", type=Path,
        help="Optional CSV filenames relative to $OUTPUT_DIR/data/experiments/comparison, or absolute paths. Omit to read the evaluation cache.",
    )
    parser.add_argument("--cache-dir", type=Path,
                        help="Evaluation cache root (default: $OUTPUT_DIR/data/cache/evaluation). Used when no CSVs are supplied.")
    parser.add_argument("--cache-policy", choices=("latest", "error"), default="latest",
                        help="How to handle multiple cached runs per comparison: newest creation time (default), or fail.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output parent directory; all files go in its <prefix>/ subfolder. Default: $OUTPUT_DIR/paper/tables.")
    parser.add_argument("--prefix", help="Subfolder, filename, and LaTeX-label prefix; default: evaluation_<date or range>.")
    parser.add_argument("--tables", nargs="+", choices=TABLES, action="extend",
                        help="Default: one compact reference_summary table. Other topics are optional.")
    parser.add_argument("--reference-date", action="append", default=[], metavar="SOURCE=YYYY-MM-DD",
                        help="Choose a date for a reference family, including its verified variant. Repeat as needed; default is latest shared date.")
    all_metrics = sorted({metric.key for table in TABLES.values() for metric in table.metrics})
    parser.add_argument("--metrics", nargs="+", choices=all_metrics, action="extend",
                        help="Restrict tables to these metric keys; paired percentages stay with counts.")
    parser.add_argument("--list-metrics", action="store_true")
    for flag in ("test-source", "gold-source", "adsb-source", "date"):
        note = " Use 'none' for published datasets without an ADS-B provider." if flag == "adsb-source" else ""
        parser.add_argument("--" + flag, nargs="+", action="extend",
                            help="Keep only these values; accepts multiple values or repeated options." + note)
    parser.add_argument("--target-day-adsb", choices=("true", "false", "unknown"), default="false")
    parser.add_argument("--require-complete", action="store_true", default=True,
                        help="Require all requested comparisons and selected metrics (now always enabled; retained for compatibility).")
    parser.add_argument("--decimals", type=int, default=2, help="LaTeX decimals (CSV retains numeric precision).")
    args = parser.parse_args(argv)
    args.tables = args.tables or ["reference_summary"]
    if args.list_metrics:
        for name, table in TABLES.items():
            print(name + ":\n" + "\n".join(f"  {metric.key}: {' '.join(metric.label.splitlines())}" for metric in table.metrics))
        return 0
    if args.csvs and args.cache_dir:
        parser.error("Use either explicit CSV inputs or --cache-dir, not both.")
    if not 0 <= args.decimals <= 6:
        parser.error("--decimals must be between 0 and 6.")
    try:
        if args.output_dir is None:
            raise ValueError("Set OUTPUT_DIR or supply --output-dir for generated tables.")
        args.csvs = [path.expanduser() for path in args.csvs]
        if any(not path.is_absolute() for path in args.csvs):
            output_root = os.environ.get("OUTPUT_DIR")
            if not output_root:
                raise ValueError("Set OUTPUT_DIR to use relative CSV filenames, or supply absolute paths.")
            comparison_dir = Path(output_root).expanduser() / "data/experiments/comparison"
            args.csvs = [path if path.is_absolute() else comparison_dir / path for path in args.csvs]
        include = {"target_day_adsb": {args.target_day_adsb}}
        for column, option in (("test_src", "test_source"), ("gold_src", "gold_source"),
                               ("adsb_src", "adsb_source"), ("test_date", "date")):
            values = getattr(args, option)
            if values:
                include[column] = {"" if column == "adsb_src" and value == "none" else value for value in values}
        cache_entries = {}
        cache_root = None
        if args.csvs:
            rows = load_rows(args.csvs, include=include)
        else:
            if not os.environ.get("OUTPUT_DIR"):
                raise ValueError("Set OUTPUT_DIR to use the evaluation cache, as required by the evaluation setup, or supply absolute CSV paths.")
            from flights.evaluation.cache import EvaluationCache
            cache = EvaluationCache(args.cache_dir.expanduser() if args.cache_dir else None)
            cache_root = str(cache.root.resolve())
            if not cache.root.is_dir():
                raise ValueError(f"Evaluation cache not found: {cache.root}. Supply existing evaluation CSVs instead.")
            rows, cache_entries = load_cache_rows(cache, include, args.cache_policy)
        if not rows:
            raise ValueError("No matching daily results. Check the source/date selections or supply existing evaluation CSVs; this script does not run missing evaluations.")
        completeness_filters = dict(include)
        if "reference_summary" in args.tables or args.reference_date:
            rows = select_reference_dates(rows, args.reference_date)
            # Summary date selection intentionally chooses one day per family.
            completeness_filters.pop("test_date", None)
        validate_completeness(rows, completeness_filters)
        rows.sort(key=reference_sort_key)
        dates = sorted({row["test_date"] for row in rows})
        prefix = args.prefix or "evaluation_" + (dates[0] if len(dates) == 1 else dates[0] + "_to_" + dates[-1])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", prefix):
            raise ValueError("--prefix must contain only letters, numbers, underscores, and hyphens.")
        selected = {name: tuple(metric for metric in TABLES[name].metrics if not args.metrics or metric.key in args.metrics)
                    for name in args.tables}
        selected = {name: metrics for name, metrics in selected.items() if metrics}
        if not selected:
            raise ValueError("No metrics selected for the requested tables.")
        warnings = prepare_rows(rows, selected, args.require_complete)
        output_dir = args.output_dir.expanduser() / prefix
        outputs = export_tables(rows, output_dir, prefix, selected, args.decimals)
        manifest = output_dir / f"{prefix}_manifest.json"
        manifest.write_text(json.dumps({
            "input_mode": "csv" if args.csvs else "evaluation_cache",
            "inputs": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in args.csvs],
            "cache_root": cache_root,
            "cache_policy": args.cache_policy if cache_root else None,
            "cache_entries": [
                {"path": str(entry.path.resolve()), "created_at": entry.created_at.isoformat(),
                 "key": entry.key.to_dict(),
                 "stats_sha256": hashlib.sha256((entry.path / "stats.parquet").read_bytes()).hexdigest()}
                for row in rows if (entry := cache_entries.get(row_key(row))) is not None
            ],
            "daily_rows": len(rows), "average_rows": "excluded; no cross-day aggregation",
            "csv_percent_scale": "0-100", "latex_decimals": args.decimals,
            "filters": {key: getattr(args, key) for key in (
                "test_source", "gold_source", "adsb_source", "date", "target_day_adsb",
            )},
            "reference_dates": {row["gold_src"]: row["test_date"] for row in rows} if "reference_summary" in selected or args.reference_date else None,
            "metrics": {name: [metric.key for metric in metrics] for name, metrics in selected.items()},
            "definitions": {name: TABLES[name].note for name in selected},
            "warnings": warnings,
            "outputs": [str(path.resolve()) for path in outputs],
        }, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for warning in warnings:
        print("Warning: " + warning, file=sys.stderr)
    print(f"Exported {len(rows)} daily comparisons to {len(selected)} CSV/LaTeX table pairs:")
    for source in dict.fromkeys(row["gold_src"] for row in rows):
        print(f"  {source}: {', '.join(sorted({row['test_date'] for row in rows if row['gold_src'] == source}))}")
    for path in [*outputs, manifest]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
