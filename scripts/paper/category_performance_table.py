"""Break down the exact evaluations in Table 0 by reference aircraft category.

Reads pinned local caches only; never evaluates or downloads flights.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csvs_to_latex_table import (
    TABLES, RATE_COLUMNS, csv_column, prepare_rows, reference_sort_key,
    summary_latex, validate_opensky_segmentation,
)

CATEGORIES = ("A1", "A2", "A3", "A4", "Other")
METHODS = ("opensky", "opensky-nearest", "opensky-nn", "algorithm-nearest", "algorithm-nn")
NOTE = (
    "Categories are the reference dataset's ADS-B emitter category, assigned once "
    "per ICAO and day using full-day gold flights. Conflicting categories are rejected. "
    "A1, A2, A3 and A4 are separate; Other includes every remaining code, null and blank. "
    "Unmatched test flights inherit their ICAO's reference category, so false positives "
    "remain in the correct denominator. The exact Table 0 evaluations and dates are used. "
    "Gold ICAOs and zero-flight coverage use full-day inputs; flight metrics use the "
    "cached evaluation window. All five groups partition each original evaluation. "
    "Airport accuracy is undefined (dash) when there are no matched flights. "
    + TABLES["reference_summary"].note
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def category_lookup(gold: pl.DataFrame) -> pl.DataFrame:
    if gold["icao"].null_count():
        raise ValueError("Gold inputs contain null ICAOs")
    categories = gold.select("icao", pl.col("category").fill_null("").str.strip_chars()
                             .str.to_uppercase().alias("reference_category")).unique()
    conflicts = categories.group_by("icao").len().filter(pl.col("len") != 1)
    if conflicts.height:
        raise ValueError(f"Conflicting gold categories for {conflicts.height} ICAOs")
    return categories.with_columns(
        pl.when(pl.col("reference_category").is_in(CATEGORIES[:-1]))
        .then(pl.col("reference_category")).otherwise(pl.lit("Other")).alias("category_group")
    )


def categorize(comparison: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    result = comparison.join(lookup, on="icao", how="left", validate="m:1")
    if result["category_group"].null_count():
        raise ValueError("Comparison contains ICAOs outside the full-day gold cohort")
    # Verify that matched/unmatched gold rows agree with the pinned reference cache.
    if "category_df1" in result.columns:
        disagreement = result.filter(
            (pl.col("match_status") != "df0_only") &
            (pl.col("category_df1").fill_null("").str.strip_chars().str.to_uppercase()
             != pl.col("reference_category")))
        if disagreement.height:
            raise ValueError("Comparison and gold cache categories differ")
    return result


def summarize(comparison, lookup, test_icaos, metadata):
    from flights.evaluation.models import EvaluationStats
    tagged = categorize(comparison, lookup)
    rows = []
    for category in CATEGORIES:
        icaos = set(lookup.filter(pl.col("category_group") == category)["icao"])
        stats = EvaluationStats.from_comparison(
            tagged.filter(pl.col("category_group") == category),
            **metadata, gold_icao_count=len(icaos),
            gold_icao_missing_from_test_count=len(icaos - test_icaos),
        )
        row = stats.to_row()
        row.update(category_group=category, adsb_src=row["adsb_src"] or "",
                   target_day_adsb=str(row["target_day_adsb"]).lower())
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-manifest", type=Path, default=ROOT / "paper/tables/table_0/table_0_manifest.json")
    parser.add_argument("--category-manifest", type=Path, default=ROOT / "paper/tables/reference_flight_breakdown/reference_flight_breakdown_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/tables/category_performance")
    args = parser.parse_args()
    from flights.evaluation.models import EvaluationStats
    from data_engineering.adsb.read_adsb import ADSBSource

    manifest = json.loads(args.table_manifest.read_text())
    category_manifest = json.loads(args.category_manifest.read_text())
    gold_inputs = {e["source"]: e for e in category_manifest["inputs"] if "path" in e}
    entries = manifest["cache_entries"]
    expected = {(source, method) for source in manifest["reference_dates"] for method in METHODS}
    if {(e["key"]["gold_source"]["name"], e["key"]["test_source"]["name"]) for e in entries} != expected or len(entries) != len(expected):
        raise ValueError("Expected exactly five methods per reference dataset")
    lookups, provenance, coverage_inputs = {}, [], {}
    for source, entry in gold_inputs.items():
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Gold cache changed: {path}")
        lookups[source] = category_lookup(pl.read_parquet(path, columns=["icao", "category"]))
        provenance.append(dict(role="gold_categories", path=str(path), sha256=entry["sha256"]))

    rows = []
    for entry in entries:
        key = entry["key"]
        source, method = key["gold_source"]["name"], key["test_source"]["name"]
        day = key["test_date"]
        if day != gold_inputs[source]["date"] or key["gold_source"]["version"] != gold_inputs[source]["source_version"]:
            raise ValueError("Gold cache version/date differs from Table 0")
        path = Path(entry["path"])
        if sha256(path / "stats.parquet") != entry["stats_sha256"]:
            raise ValueError(f"Table 0 stats changed: {path}")
        if json.loads((path / "manifest.json").read_text())["key"] != key:
            raise ValueError(f"Cache key differs: {path}")
        cached = EvaluationStats.from_row(pl.read_parquet(path / "stats.parquet").row(0, named=True))
        if cached.target_day_adsb:
            raise ValueError("Expected Table 0's target_day_adsb=false runs")
        comparison = pl.read_parquet(path / "result.parquet")
        # Airport variants preserve their base method's flight identities. Use
        # those full-day inputs for pre-window aircraft coverage, verifying both
        # retained flight identities and cached full-day coverage counts below.
        base_method = "opensky" if method.startswith("opensky") else "algorithm-nearest"
        base = next(e for e in entries if e["key"]["gold_source"]["name"] == source
                    and e["key"]["test_source"]["name"] == base_method)
        base_key = base["key"]
        data_root = Path(manifest["cache_root"]).parent.parent.parent
        flight_path = data_root / "data/flights" / f"source={base_method}"
        if base_method == "algorithm-nearest":
            flight_path /= "adsb_source=opensky"
        flight_path = flight_path / f"version={base_key['test_source']['version']}" / f"year={day[:4]}" / f"month={day[5:7]}" / f"day={day[8:]}" / "data.parquet"
        if flight_path not in coverage_inputs:
            coverage_inputs[flight_path] = pl.read_parquet(flight_path, columns=["icao", "takeoff_time", "landing_time"])
            provenance.append(dict(role="full_day_base_flights", path=str(flight_path), sha256=sha256(flight_path)))
        flights = coverage_inputs[flight_path]
        identities = ["icao", "takeoff_time", "landing_time"]
        retained_test = comparison.filter(pl.col("match_status") != "df1_only").select(identities)
        if retained_test.join(flights, on=identities, how="anti").height:
            raise ValueError(f"Base flights do not contain retained test identities for {source}/{method}")
        test_icaos = set(flights["icao"])
        lookup = lookups[source]
        gold_icaos = set(lookup["icao"])
        if len(gold_icaos) != cached.gold_icao_count or len(gold_icaos - test_icaos) != cached.gold_icao_missing_from_test_count:
            raise ValueError(f"Full-day coverage differs from Table 0 for {source}/{method}")
        metadata = dict(test_date=date.fromisoformat(day), test_source=method,
                        gold_source=source, adsb_source=ADSBSource(cached.adsb_source) if cached.adsb_source else None,
                        target_day_adsb=False)
        recomputed = EvaluationStats.from_comparison(comparison, **metadata,
            gold_icao_count=len(gold_icaos), gold_icao_missing_from_test_count=len(gold_icaos - test_icaos))
        for column, expected_value in cached.to_row().items():
            actual = recomputed.to_row()[column]
            same = (math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12)
                    if isinstance(actual, float) and isinstance(expected_value, float)
                    else actual == expected_value)
            if not same:
                raise ValueError(f"Comparison does not reproduce pinned stats for {source}/{method}/{column}")
        group_rows = summarize(comparison, lookup, test_icaos, metadata)
        count_keys = [k for k, value in cached.to_row().items() if isinstance(value, int) and not isinstance(value, bool)]
        for column in count_keys:
            if sum(r[column] for r in group_rows) != cached.to_row()[column]:
                raise ValueError(f"Category counts do not reconcile: {source}/{method}/{column}")
        rows.extend(group_rows)
        provenance.append(dict(role="evaluation", path=str(path), key=key,
                               result_sha256=sha256(path / "result.parquet"), stats_sha256=entry["stats_sha256"]))

    for category in CATEGORIES:
        validate_opensky_segmentation([r for r in rows if r["category_group"] == category])
    metrics = TABLES["reference_summary"].metrics
    prepare_rows(rows, {"reference_summary": metrics}, require_complete=True)
    for row in rows:
        if not row["true_positive"]:
            for endpoint in ("takeoff", "landing"):
                row[f"{endpoint}_airport_id_match_pct"] = None
    rows.sort(key=lambda r: (*reference_sort_key(r)[:4], CATEGORIES.index(r["category_group"]), *reference_sort_key(r)[4:]))
    latex = summary_latex(rows, metrics, "category_performance", 2,
                          caption="Flight evaluation by reference dataset and aircraft category",
                          note=NOTE, label_suffix="")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.output_dir / "category_performance.tex"
    tex_path.write_text(latex)
    csv_path = args.output_dir / "category_performance.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[csv_column(c) for c in rows[0]])
        writer.writeheader()
        writer.writerows({csv_column(k): (100*v if k in RATE_COLUMNS and v is not None else v)
                         for k, v in row.items()} for row in rows)
    (args.output_dir / "category_performance_manifest.json").write_text(json.dumps(dict(
        table_manifest=str(args.table_manifest), table_manifest_sha256=sha256(args.table_manifest),
        category_manifest=str(args.category_manifest), category_manifest_sha256=sha256(args.category_manifest),
        reference_dates=manifest["reference_dates"], methods=METHODS, categories=CATEGORIES,
        definitions=NOTE, rows=len(rows), inputs=provenance,
        csv_percent_scale="0-100 for columns ending in _percent; other rate columns retain evaluator fractions",
        validation="Category counts sum exactly to pinned Table 0 counts; whole-comparison floating-point stats agree within 1e-12.",
        outputs=[str(tex_path), str(csv_path)],
    ), indent=2) + "\n")
    print(tex_path)
    print(csv_path)
    print(f"Validated {len(entries)} evaluations; wrote {len(rows)} category/method rows.")


if __name__ == "__main__":
    main()
