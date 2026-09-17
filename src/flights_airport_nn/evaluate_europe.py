"""Frozen NN/nearest evaluation on EUROCONTROL and separate OpenSky estimates."""

import argparse
import hashlib
import json
from datetime import date
from functools import partial
from pathlib import Path

import polars as pl

from data_engineering.eurocontrol.read import eurocontrol_flights_parquet_path
from data_engineering.utils import OUTPUT_DIR
from flights_airport_nn.broader_evaluation import load_catalog
from flights_airport_nn.derive_day import DEFAULT_MODEL_PATH
from flights_airport_nn.evaluate_native_sources import KEY, fingerprint, main
from flights_airport_nn.reference_labels import AIRPORT_TYPES
from flights_airport_nn.report_native_evaluation import accuracy, annotate

METADATA_URL = "https://www.eurocontrol.int/sites/default/files/2025-04/eurocontrol-aviation-data-repository-research-metadata.pdf"


def load_european_references(day, catalog, kind):
    path = (
        eurocontrol_flights_parquet_path(day) if kind == "EUROCONTROL"
        else OUTPUT_DIR / f"data/flights/opensky/v1/{day}.parquet"
    )
    original = pl.read_parquet(path)
    fields = ["icao", "takeoff_time", "landing_time", "takeoff_airport_ident", "landing_airport_ident"]
    data = original.unique(subset=fields, maintain_order=True).sort(fields)
    records, excluded = [], []
    for index, r in enumerate(data.iter_rows(named=True)):
        record = {
            "reference_id": f"{day}:{kind}:{index}", "reference_source": kind,
            "date": str(day), "icao": (r["icao"] or "").lower(),
            "takeoff_time": r["takeoff_time"], "landing_time": r["landing_time"],
            "callsign": r.get("callsign") or "", "registration": r.get("registration") or "",
            "aircraft_type": r.get("aircraft_type") or "", "operator": r.get("owner") or "",
            "time_kind": "off_block_to_landing" if kind == "EUROCONTROL" else "first_to_last_seen",
            "start_min_s": -300 if kind == "EUROCONTROL" else -900,
            "start_max_s": 3600 if kind == "EUROCONTROL" else 900,
            "end_tolerance_s": 900,
        }
        for endpoint in ("takeoff", "landing"):
            raw = r[f"{endpoint}_airport_ident"] or ""
            # These derived files already contain canonical idents. Do not use
            # FAA/IATA fallbacks or choose nearby substitutes for unknown labels.
            supported = catalog.get(raw, {}).get("type") in AIRPORT_TYPES
            record[f"{endpoint}_raw"] = raw
            record[f"{endpoint}_truth"] = raw if supported else ""
            record[f"{endpoint}_resolution"] = "canonical_derived_ident" if supported else "unresolved_or_unsupported"
        reason = None
        if not any(catalog.get(r[f"{e}_airport_ident"], {}).get("continent") == "EU" for e in ("takeoff", "landing")):
            reason = "no_european_endpoint"
        elif not record["icao"] or not r["takeoff_time"] or not r["landing_time"] or r["landing_time"] <= r["takeoff_time"]:
            reason = "missing_identity_or_invalid_time"
        elif r["takeoff_time"].date() != day or r["landing_time"].date() != day:
            reason = "cross_day_or_different_day"
        if reason:
            excluded.append({**record, "exclusion": reason})
        else:
            records.append(record)
    return pl.DataFrame(records, infer_schema_length=None), pl.DataFrame(excluded, infer_schema_length=None), [fingerprint(path), fingerprint(Path(__file__))]


def report(output):
    protocol = json.loads((output / "protocol.json").read_text())
    refs = pl.concat([pl.read_parquet(output / f"references-{d}.parquet") for d in protocol["dates"]])
    raw = pl.read_parquet(output / "endpoint_predictions.parquet")
    catalog = load_catalog()
    data, _ = annotate(raw, refs, protocol["lineage"], catalog)
    metadata = pl.DataFrame([{"truth": k, "continent": a["continent"], "airport_name": a["name"]} for k, a in catalog.items()])
    data = data.join(metadata, on="truth", how="left", validate="m:1").join(
        refs.select("reference_id", "time_kind", "registration", "aircraft_type", "operator", pl.col("takeoff_time").alias("reference_departure")),
        on="reference_id", how="left", validate="m:1",
    ).with_columns(
        (pl.col("segment_start") - pl.col("reference_departure")).dt.total_milliseconds().truediv(1000).alias("departure_offset_s")
    )
    normal = data.filter(pl.col("loss_seconds") == 0)
    assert not normal.select(*KEY, "source").is_duplicated().any()
    assert normal.height == refs.height * 2 * len(protocol["sources"])
    assert data.filter((pl.col("match_status") != "matched") & (pl.col("nn_correct") | pl.col("nearest_correct"))).is_empty()
    valid = (~pl.col("reference_time_conflict")) & (pl.col("truth") != "") & (pl.col("continent") == "EU")
    base = normal.filter(valid)
    strict_time = (
        pl.col("departure_offset_s").is_between(-120, 1800)
        if refs["reference_source"][0] == "EUROCONTROL"
        else pl.col("departure_offset_s").abs() <= 300
    ) & (pl.col("end_delta_s") <= 300)
    common = base.group_by(KEY).agg(
        ((pl.col("match_status") == "matched").all() & (pl.col("source").n_unique() == len(protocol["sources"]))).alias("common_matched"),
        pl.col("truth_ground").any().alias("physical_support_any_network"),
        strict_time.fill_null(False).all().alias("strict_time_all_networks"),
    )
    data = data.join(common, on=KEY, how="left", validate="m:1")
    base = data.filter((pl.col("loss_seconds") == 0) & valid)
    strata = {
        "all_european_endpoints": pl.lit(True),
        "large_airports": pl.col("airport_type") == "large_airport",
        "medium_airports": pl.col("airport_type") == "medium_airport",
        "small_airports": pl.col("airport_type") == "small_airport",
        "unseen_aircraft": ~pl.col("seen_aircraft"),
        "unseen_airports": ~pl.col("seen_airport"),
        "ground_corroborated_secondary": pl.col("physical_support_any_network"),
        "without_ground_support": ~pl.col("physical_support_any_network"),
        "takeoff": pl.col("endpoint") == "takeoff",
        "landing": pl.col("endpoint") == "landing",
        "strict_time_subset": pl.col("strict_time_all_networks"),
    }
    metrics = {"dates": protocol["dates"], "reference_source": refs["reference_source"][0], "sources": protocol["sources"], "reference_flights": refs.height, "reference_conflicts": refs.filter(pl.col("reference_id").is_in(data.filter(pl.col("reference_time_conflict"))["reference_id"].unique().implode())).height, "groups": {}}
    for name, condition in strata.items():
        group = base.filter(condition)
        metrics["groups"][name] = {}
        for source in protocol["sources"]:
            source_data = group.filter(pl.col("source") == source)
            metrics["groups"][name][source] = {
                "all_reference": accuracy(source_data),
                "source_matched": accuracy(source_data.filter(pl.col("match_status") == "matched")),
                "common_matched": accuracy(source_data.filter(pl.col("common_matched"))),
            }
    # Strict pairing under stress, including when a network failed to produce a
    # native segment. Normal rows are compared on precisely this stress subset.
    metrics["stress"] = {}
    for loss in (60, 180):
        stress = data.filter(valid & (pl.col("loss_seconds") == loss) & pl.col("common_matched"))
        keys = stress.group_by(KEY).agg(pl.col("source").n_unique().alias("n")).filter(pl.col("n") == len(protocol["sources"])).select(KEY)
        metrics["stress"][str(loss)] = {}
        for source in protocol["sources"]:
            metrics["stress"][str(loss)][source] = {
                "masked": accuracy(stress.filter(pl.col("source") == source).join(keys, on=KEY)),
                "same_endpoints_normal": accuracy(base.filter(pl.col("source") == source).join(keys, on=KEY)),
            }
    data.write_parquet(output / "endpoint_results.parquet")
    data.filter((pl.col("loss_seconds") == 0) & ((~pl.col("nn_correct")) | (~pl.col("nearest_correct")))).write_parquet(output / "errors_and_unmatched.parquet")
    # Save every reference endpoint with side-by-side native flight and match state.
    paired = normal.select(*KEY, "reference_source", "date", "icao", "truth").unique()
    for source in protocol["sources"]:
        fields = ["flight_id", "segment_start", "segment_end", "match_status", "nn", "nearest", "nn_correct", "nearest_correct", "observed", "truth_ground"]
        side = normal.filter(pl.col("source") == source).select(*KEY, *[pl.col(k).alias(f"{source}_{k}") for k in fields])
        paired = paired.join(side, on=KEY, how="left", validate="1:1")
    paired.write_parquet(output / "paired_endpoints.parquet")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "report_policy.json").write_text(json.dumps({"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "primary": "Resolved European endpoints excluding metadata-only overlapping-reference conflicts; no prediction-driven relabeling or ground-evidence filter", "ci": "95% paired aircraft bootstrap; does not include reference-label uncertainty"}, indent=2) + "\n")
    lines = [f"# European evaluation: {metrics['reference_source']}", "", f"Dates: {', '.join(protocol['dates'])}. Frozen model: `{protocol['model_sha256']}`.", "", "Accuracy below is per European endpoint, on native segments uniquely matched in every tested network. Missing/ambiguous references remain in the all-reference metrics and Parquet.", "", "| Subset | Network | Endpoints | Airports | NN | Nearest |", "|---|---|---:|---:|---:|---:|"]
    for name, group in metrics["groups"].items():
        for source, scores in group.items():
            s = scores["common_matched"]
            if s["endpoints"]:
                lines.append(f"| {name} | {source} | {s['endpoints']:,} | {s['airports']:,} | {s['nn_accuracy']:.2%} | {s['nearest_accuracy']:.2%} |")
    lines += ["", "## Coverage", ""]
    for source, scores in metrics["groups"]["all_european_endpoints"].items():
        s = scores["all_reference"]
        lines.append(f"- {source}: {s['matched']:,}/{s['endpoints']:,} reference endpoints matched; {s['ambiguous']:,} ambiguous, {s['unmatched']:,} unmatched. NN/nearest correct across the full reference population: {s['nn_accuracy']:.2%}/{s['nearest_accuracy']:.2%}.")
    lines += ["", "## Interpretation", "", "- EUROCONTROL: independent Network Manager airport records for scheduled and non-scheduled commercial flights. Departure is off-block, arrival is landing. Match native departure between 5 minutes before and 60 minutes after off-block, and arrival within 15 minutes; airport labels/positions never enter matching. Strict subset uses -2/+30 minutes and arrival within 5 minutes.", f"- [EUROCONTROL metadata]({METADATA_URL}). The research sample excludes general aviation, military and other flight categories. ADEP/ADES are ICAO airport codes; ADES records the actual destination after diversion. EUROCONTROL data are acknowledged and are for research use.", "- Daily derived references already require a registration-to-ICAO association and resolvable airport labels. They do not represent every flight in the original NM extract. Flights crossing UTC midnight are excluded. This is historical performance using the current airport catalog, not a contemporaneous 2024 deployment.", "- OpenSky rows are estimated airports and first/last observed times: secondary agreement only, never pooled with EUROCONTROL or described as independent ground truth.", "- Ground support is observational evidence, not an independent label. Primary EUROCONTROL results do not require it; the corroborated subset favors good reception. Wrong predictions do not automatically establish a flawed label.", "- Dates do not overlap fitting/selection, including pretraining. Aircraft and airports can recur; unseen strata are reported. No retraining or checkpoint selection was performed.", "- The common sample compares the same endpoints. Separate per-network matched/full-reference metrics, candidate recall, macro-airport scores, aircraft-bootstrap intervals and 60/180-second reception-loss tests are in metrics.json.", "", "Artifacts: endpoint_results.parquet (all rows/scenarios), paired_endpoints.parquet (reference plus each network's native flight, match_status and predictions), errors_and_unmatched.parquet, references/exclusions and frozen protocol.json."]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:18]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    parser.add_argument("--reference", choices=["EUROCONTROL", "OpenSky"], required=True)
    parser.add_argument("--sources", nargs="+", choices=["adsblol", "adsbx"], default=["adsblol"])
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    roots = None
    if args.prepared_root:
        for source in args.sources:
            for day in args.dates:
                if not (args.prepared_root / f"acquisition-{source}-{day}.json").exists():
                    parser.error(f"Missing completed acquisition manifest for {source} {day}")
        roots = {s: args.prepared_root / "parquet" / s for s in args.sources}
    main(args.dates, args.sources, args.output_dir, args.model, args.workers, args.batch_size,
         reference_loader=partial(load_european_references, kind=args.reference), parquet_roots=roots,
         extra_policy={"labels": args.reference + (" independent NM airport records" if args.reference == "EUROCONTROL" else " estimated airports: agreement only"), "matching": "Unique aircraft/time with explicit per-reference bounds; EUROCONTROL off-block -5/+60min and landing +/-15min; OpenSky first/last-seen +/-15min", "scope": "Flights with at least one European airport, both reference timestamps on same UTC day, identified aircraft; all endpoints retained, headline limited to European airports; exclude metadata overlap conflicts in report"})
    report(args.output_dir)
