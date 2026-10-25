"""Export aircraft-category and airport breakdowns from cached reference flights.

Reads current-version daily flight caches and existing airport history only.
No downloads, flight generation, evaluation matching, or boundary filtering.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import polars as pl

from csvs_to_latex_table import DEFAULT_OUTPUT_DIR, latex_escape

REFERENCE_DATES = {"bts": "2026-03-01", "sfdps": "2026-03-01", "eurocontrol": "2024-06-01"}
LABELS = {"bts": "BTS", "sfdps": "SFDPS", "eurocontrol": "Eurocontrol"}
GROUPED_CATEGORIES = frozenset(
    [f"{letter}{number}" for letter in "BC" for number in range(8)]
    + [f"D{number}" for number in range(3)]
)
CATEGORY_LABELS = {
    "B0--D2": "Other reported categories",
    "A0": "No category information",
    "A1": "Light (e.g., Cessna 172)",
    "A2": "Small (e.g., CRJ-200)",
    "A3": "Large (e.g., Boeing 737 / Airbus A320)",
    "A4": "High-vortex large (Boeing 757)",
    "A5": "Heavy (e.g., Boeing 777)",
    "A6": "High-performance aircraft",
    "A7": "Rotorcraft",
    "B0": "No category information",
    "B1": "Glider / sailplane", "B2": "Lighter-than-air",
    "B3": "Parachutist / skydiver", "B4": "Ultralight / hang-glider / paraglider",
    "B5": "Reserved", "B6": "Unmanned aircraft", "B7": "Space / trans-atmospheric",
    "C0": "No category information", "C1": "Emergency surface vehicle",
    "C2": "Service surface vehicle", "C3": "Point obstacle",
    "C4": "Cluster obstacle", "C5": "Line obstacle", "C6": "Reserved", "C7": "Reserved",
    "missing": "Missing category (null / blank)",
}
SIZE_LABELS = {
    "small_airport": "Small airport", "medium_airport": "Medium airport",
    "large_airport": "Large airport", "heliport": "Heliport",
    "seaplane_base": "Seaplane base", "balloonport": "Balloonport", "closed": "Closed",
    "unknown": "Unknown / unresolved airport",
}
CODE_LABELS = {"both": "Both IATA and ICAO", "iata_only": "IATA only",
               "icao_only": "ICAO only", "neither": "Neither IATA nor ICAO",
               "unknown": "Unknown / unresolved airport"}
SOURCES = {
    "categories": "https://support.adsbexchange.com/hc/en-us/articles/44705224053517-Emitter-Category-ADS-B-DO-260B-2-2-3-2-5-2",
    "airports": "https://ourairports.com/help/data-dictionary.html",
}
POPULATION_NOTE = (
    "All cached reference flights on the indicated UTC takeoff day, before evaluation "
    "matching or boundary filtering; provider preprocessing still applies. "
    "Percentages include missing values in the denominator. CSV counts are exact; "
    "rounding may prevent displayed percentages summing to 100."
)
AIRCRAFT_NOTE = (
    "ADS-B emitter categories use the stored category field, not the aircraft_type "
    "model code (e.g., B738). Maximum takeoff weight in pounds: A1 under 15,500; "
    "A2 15,500 to under 75,000; A3/A4 75,000 to under 300,000; A5 at least 300,000. "
    "A4 identifies high wake vortex aircraft; A6 denotes maneuver capability above "
    "5 g and speed above 400 kt. Examples illustrate "
    "categories; reported classifications are preserved, even when anomalous. "
    "Blank values are not inferred from model codes. Categories come from provider "
    "enrichment (OpenAirframes for BTS/Eurocontrol; ADS-B observations for SFDPS). "
    "A0/B0/C0 mean no category information. B0 through D2 are combined into one "
    "percentage; the CSV retains individual category counts. "
    "Each flight counts once. " + POPULATION_NOTE
)
AIRPORT_NOTE = (
    "Cells show takeoff / landing percentages; each denominator is the dataset's "
    "flight count n. Repeated visits count repeatedly, not once per distinct airport. "
    "Sizes are OurAirports type labels, not thresholds derived from these flights. "
    "Airport sizes use the flight-date catalog and its one-month grace policy. "
    "Code availability uses the explicit iata_code and icao_code fields, not ident "
    "or gps_code. For example, Los Angeles has IATA LAX and ICAO KLAX (both). "
    "No IATA or ICAO means a resolved airport with both fields blank; unresolved "
    "airports or absent code metadata do not contribute to this numerator. "
    "Heliport, seaplane base, balloonport, and closed rows are omitted from the table. "
    "The CSV retains all airport types and code groups; denominators include all flights. " + POPULATION_NOTE
)


def summarize(flights: pl.DataFrame, airports: dict, source: str, day: str,
              code_airports: dict | None = None) -> list[dict]:
    """Count flight-weighted categories and endpoint characteristics, retaining nulls."""
    required = {"category", "takeoff_time", "takeoff_airport_id", "landing_airport_id"}
    if missing := required - set(flights.columns):
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    if flights.is_empty():
        raise ValueError(f"{source}: empty reference flight cache")
    if flights.filter((pl.col("takeoff_time").dt.date() != date.fromisoformat(day)).fill_null(True)).height:
        raise ValueError(f"{source}: flights must have a takeoff time on {day}")
    if code_airports is None:
        code_airports = airports
    counts = {("aircraft_category", "flight"): Counter(
        (value or "").strip().upper() or "missing" for value in flights["category"])}
    for endpoint in ("takeoff", "landing"):
        sizes, codes = Counter(), Counter()
        for airport_id in flights[f"{endpoint}_airport_id"]:
            airport = airports.get(airport_id)
            if airport is None:
                sizes["unknown"] += 1
                codes["unknown"] += 1
                continue
            sizes[str(airport.type)] += 1
            code_airport = code_airports.get(airport_id)
            if code_airport is None:
                codes["unknown"] += 1
                continue
            iata, icao = bool((code_airport.iata_code or "").strip()), bool((code_airport.icao_code or "").strip())
            codes["both" if iata and icao else "iata_only" if iata else "icao_only" if icao else "neither"] += 1
        counts["airport_size", endpoint] = sizes
        counts["airport_codes", endpoint] = codes
    rows = []
    for (dimension, endpoint), counter in counts.items():
        assert sum(counter.values()) == flights.height
        # Include structural zeros, so all datasets have comparable rows.
        keys = (set(f"A{i}" for i in range(8)) | {"missing"} if dimension == "aircraft_category"
                else set(SIZE_LABELS) if dimension == "airport_size" else set(CODE_LABELS))
        for key in sorted(keys | set(counter)):
            count = counter[key]
            rows.append(dict(source=source, date=day, dimension=dimension, endpoint=endpoint,
                             category=key, count=count, denominator=flights.height,
                             percent=100 * count / flights.height))
    return rows


def render_latex(rows: list[dict], prefix: str, kind: str, decimals: int,
                 airport_code_date: str = "unspecified") -> str:
    aircraft = kind == "aircraft"
    title = ("Aircraft categories of reference flights" if aircraft else
             "Airport characteristics of reference flights")
    totals = {r["source"]: r["denominator"] for r in rows}
    lookup = {(r["source"], r["dimension"], r["endpoint"], r["category"]): r["percent"] for r in rows}
    header = ["Category / description" if aircraft else "Airport characteristic"]
    for source, day in REFERENCE_DATES.items():
        header.append(r"\shortstack{" + LABELS[source] + r"\\" + day + r"\\$n=" +
                      f"{totals[source]:,}" + r"$\\" + (r"(\%)" if aircraft else r"TO / LD (\%)") + "}")
    heading = " & ".join(header) + r" \\"
    lines = [r"% Requires \usepackage{booktabs,longtable,array}.",
             "% Definitions: " + SOURCES["categories" if aircraft else "airports"],
             r"\begingroup", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.43\linewidth}rrr@{}}",
             r"\caption{" + title + "}", r"\label{tab:" + prefix.replace("_", "-") + "-" + kind + r"}\\",
             r"\toprule", heading, r"\midrule", r"\endfirsthead",
             r"\toprule", heading, r"\midrule", r"\endhead",
             r"\midrule\multicolumn{4}{r}{Continued on next page}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    if aircraft:
        observed = {r["category"] for r in rows if r["dimension"] == "aircraft_category"}
        if observed & GROUPED_CATEGORIES:
            for source in REFERENCE_DATES:
                count = sum(r["count"] for r in rows if r["source"] == source
                            and r["dimension"] == "aircraft_category"
                            and r["category"] in GROUPED_CATEGORIES)
                lookup[source, "aircraft_category", "flight", "B0--D2"] = 100 * count / totals[source]
            observed = (observed - GROUPED_CATEGORIES) | {"B0--D2"}
        keys = sorted(observed - {"missing"}) + ["missing"]
        sections = [("aircraft_category", "", keys, CATEGORY_LABELS)]
    else:
        sections = [("airport_size", "Airport size / type",
                     ["small_airport", "medium_airport", "large_airport", "unknown"], SIZE_LABELS),
                    ("airport_codes", "Airport code availability", ["neither"],
                     {"neither": "No IATA or ICAO"})]
    for dimension, section, keys, labels in sections:
        if section:
            lines += [r"\addlinespace[4pt]", r"\multicolumn{4}{@{}l}{\textbf{" + section + r"}}\\*"]
        for key in keys:
            label = labels.get(key, "Unrecognized reported code" if aircraft else key)
            if aircraft and key != "missing":
                label = f"{key}: {label}"
            cells = [latex_escape(label)]
            for source in REFERENCE_DATES:
                endpoints = ("flight",) if aircraft else ("takeoff", "landing")
                cells.append(" / ".join(f"{lookup.get((source, dimension, ep, key), 0):.{decimals}f}" for ep in endpoints))
            lines.append(" & ".join(cells) + r" \\")
    note = AIRCRAFT_NOTE if aircraft else (
        f"Code availability for all datasets uses the {airport_code_date} OurAirports snapshot, "
        "because the 2024 schema lacked an explicit ICAO-code field; these code shares "
        "describe that snapshot, not historical code assignments. " + AIRPORT_NOTE)
    lines += [r"\end{longtable}", r"\noindent\begin{minipage}{\linewidth}\footnotesize " +
              latex_escape(note) + r"\end{minipage}\par",
              r"\endgroup", ""]
    return "\n".join(lines)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_inputs(code_date: date | None = None) -> tuple[list[dict], list[dict]]:
    # Import lazily: --help and aggregation tests need no data root or providers.
    from airports.catalog import AirportCatalogView
    from airports.history import AirportHistory
    from airports.history.schema import HistoryPaths
    from flights.flight_manifest import FlightManifest, flight_manifest_path
    from flights.flight_source import flight_path
    from flights.providers.bts import BTSFlightSource
    from flights.providers.eurocontrol import EurocontrolFlightSource
    from flights.providers.sfdps import SFDPSFlightSource

    history = AirportHistory(HistoryPaths.defaults())  # open() may refresh/download.
    code_date = code_date or history.latest_available_date
    if code_date > history.latest_available_date:
        raise ValueError("Airport code date is later than the available local history.")
    code_frame = history.as_of(code_date)
    if code_frame["icao_code"].drop_nulls().is_empty():
        raise ValueError("Selected airport snapshot predates explicit ICAO-code metadata; choose a newer --airport-code-date.")
    code_airports = {row["id"]: SimpleNamespace(iata_code=row["iata_code"], icao_code=row["icao_code"])
                     for row in code_frame.select("id", "iata_code", "icao_code").iter_rows(named=True)}
    rows, inputs = [], []
    for source in (BTSFlightSource(), SFDPSFlightSource(), EurocontrolFlightSource()):
        day = date.fromisoformat(REFERENCE_DATES[source.name])
        path = flight_path(source, day)
        manifest_path = flight_manifest_path(path)
        if not path.is_file() or not manifest_path.is_file():
            raise ValueError(f"Missing current-version reference cache: {path}; generate flights first.")
        catalog = AirportCatalogView.from_history(history, day, history.latest_available_date)
        manifest = FlightManifest.read(manifest_path)
        if manifest.flight_cache_key != source.flight_manifest(catalog.provenance, None).flight_cache_key:
            raise ValueError(f"Stale flight cache/catalog: {path}; regenerate flights first.")
        flights = pl.read_parquet(path)
        rows.extend(summarize(flights, catalog.id_index, source.name, day.isoformat(), code_airports))
        inputs.append(dict(source=source.name, date=day.isoformat(), source_version=source.version,
                           path=str(path.resolve()), sha256=sha256(path), flight_count=flights.height,
                           flight_manifest=asdict(manifest), manifest_sha256=sha256(manifest_path),
                           airport_catalog=asdict(catalog.provenance)))
    inputs.append({"airport_code_date": code_date.isoformat(),
                   "airport_code_commit": history.source_commit_sha(code_date),
                   "airport_history": [{"path": str(path.resolve()), "sha256": sha256(path)}
                                      for path in history.paths.required_files]})
    return rows, inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Parent directory (default: $OUTPUT_DIR/paper/tables); always adds the prefix subfolder.")
    parser.add_argument("--prefix", default="reference_flight_breakdown",
                        help="Output subfolder and filename prefix (default: reference_flight_breakdown).")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--airport-code-date", type=date.fromisoformat,
                        help="Shared code-metadata snapshot date; default: latest in local history.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prefix):
        parser.error("--prefix must contain only letters, numbers, underscores, and hyphens.")
    if not 0 <= args.decimals <= 6:
        parser.error("--decimals must be between 0 and 6.")
    if not os.environ.get("OUTPUT_DIR"):
        parser.error("Set OUTPUT_DIR to the existing flight/airport data root.")
    try:
        rows, inputs = load_inputs(args.airport_code_date)
        output = args.output_dir.expanduser() / args.prefix
        # Render both tables before writing any outputs.
        code_date = inputs[-1]["airport_code_date"]
        tables = {kind: render_latex(rows, args.prefix, kind, args.decimals, code_date) for kind in ("aircraft", "airports")}
        output.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_csv(output / f"{args.prefix}.csv")
        for kind, latex in tables.items():
            (output / f"{args.prefix}_{kind}.latex").write_text(latex, encoding="utf-8")
        provenance = dict(reference_dates=REFERENCE_DATES, inputs=inputs, definitions=SOURCES,
                          aircraft_note=AIRCRAFT_NOTE, airport_note=AIRPORT_NOTE,
                          airport_code_date=code_date, percent_scale="0-100", latex_decimals=args.decimals)
        (output / f"{args.prefix}_manifest.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Exported aircraft and airport breakdowns to {output}")
    for item in inputs[:-1]:
        print(f"{LABELS[item['source']]} | {item['date']} | {item['flight_count']:,} flights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
