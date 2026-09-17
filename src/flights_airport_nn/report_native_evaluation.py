"""Summarize native-source predictions without changing labels from predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from flights_airport_nn.broader_evaluation import load_catalog
from flights_airport_nn.evaluate_native_sources import KEY, temporal_matches


def conflicting_reference_ids(references):
    """Same aircraft cannot fly overlapping flights within one reference source.

    Allow one minute of timestamp rounding. Use only reference metadata, and
    mark both sides of each conflict instead of choosing one from predictions.
    """
    if references.is_empty():
        return set()
    columns = [
        "reference_id",
        "reference_source",
        "icao",
        "date",
        "takeoff_time",
        "landing_time",
    ]
    frame = references.select(columns)
    pairs = frame.join(
        frame, on=["reference_source", "icao", "date"], suffix="_other"
    ).filter(
        (pl.col("reference_id") < pl.col("reference_id_other"))
        & (
            pl.min_horizontal("landing_time", "landing_time_other")
            - pl.max_horizontal("takeoff_time", "takeoff_time_other")
            > pl.duration(seconds=60)
        )
    )
    return set(pairs["reference_id"]) | set(pairs["reference_id_other"])


def crosscheck_bts(references):
    """Independent time/identity linkage to airline-reported airport labels."""
    rows = []
    for day in references["date"].unique():
        refs = references.filter(pl.col("date") == day)
        bts = refs.filter(pl.col("reference_source") == "BTS")
        faa = refs.filter(pl.col("reference_source") == "FAA")
        if bts.is_empty() or faa.is_empty():
            continue
        matches = temporal_matches(
            bts.with_columns(pl.col("reference_id").alias("flight_id")), faa
        ).filter(pl.col("unique_match"))
        labels = {r["reference_id"]: r for r in refs.iter_rows(named=True)}
        for match in matches.iter_rows(named=True):
            a, b = labels[match["reference_id"]], labels[match["flight_id"]]
            for endpoint in ("takeoff", "landing"):
                truth = a[f"{endpoint}_truth"]
                other = b[f"{endpoint}_truth"]
                rows.append(
                    {
                        "reference_id": a["reference_id"],
                        "endpoint": endpoint,
                        "bts_reference_id": b["reference_id"],
                        "bts_truth": other,
                        "bts_agree": bool(truth and other and truth == other),
                        "bts_conflict": bool(truth and other and truth != other),
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "reference_id": pl.String,
            "endpoint": pl.String,
            "bts_reference_id": pl.String,
            "bts_truth": pl.String,
            "bts_agree": pl.Boolean,
            "bts_conflict": pl.Boolean,
        },
    )


def annotate(frame, references, lineage, catalog):
    normal = frame.filter(pl.col("loss_seconds") == 0)
    if normal.select(*KEY, "source").is_duplicated().any():
        raise ValueError("Duplicate source/reference endpoint")
    evidence = normal.group_by(KEY).agg(
        pl.col("truth_ground").any().alias("ground_any"),
        pl.col("truth_ground").all().alias("ground_all"),
        pl.col("source").n_unique().alias("evidence_sources"),
    )
    conflicts = conflicting_reference_ids(references)
    check = (
        crosscheck_bts(references.filter(~pl.col("reference_id").is_in(conflicts)))
        if references.height
        else crosscheck_bts(references)
    )
    data = (
        frame.join(evidence, on=KEY, validate="m:1")
        .join(check, on=KEY, how="left", validate="m:1")
        .with_columns(
            pl.col("bts_agree").fill_null(False),
            pl.col("bts_conflict").fill_null(False),
            pl.col("reference_id").is_in(conflicts).alias("reference_time_conflict"),
        )
    )
    for source in ("adsblol", "adsbx"):
        source_evidence = normal.filter(pl.col("source") == source).select(
            *KEY, pl.col("truth_ground").alias(f"ground_{source}")
        )
        data = data.join(source_evidence, on=KEY, how="left", validate="m:1")
    metadata = pl.DataFrame(
        [
            {
                "truth": ident,
                "airport_type": a["type"],
                "country": a["iso_country"],
                "region": a.get("iso_region", ""),
            }
            for ident, a in catalog.items()
        ]
    )
    data = data.join(metadata, on="truth", how="left", validate="m:1").with_columns(
        pl.col("icao").is_in(lineage["aircraft"]).alias("seen_aircraft"),
        pl.col("truth").is_in(lineage["airports"]).alias("seen_airport"),
        ((pl.col("truth") != "") & (pl.col("nn") == pl.col("truth"))).alias(
            "nn_correct"
        ),
        ((pl.col("truth") != "") & (pl.col("nearest") == pl.col("truth"))).alias(
            "nearest_correct"
        ),
        pl.col("candidates")
        .list.contains(pl.col("truth"))
        .fill_null(False)
        .alias("retrieved"),
        pl.when(pl.col("reference_time_conflict"))
        .then(pl.lit("reference_identity_time_conflict"))
        .when(pl.col("truth") == "")
        .then(pl.lit("unresolved"))
        .when(pl.col("reference_source") == "BTS")
        .then(pl.lit("airline_reported"))
        .when(pl.col("bts_conflict"))
        .then(pl.lit("reference_conflict"))
        .when(pl.col("bts_agree"))
        .then(pl.lit("faa_airline_agreement"))
        .when(pl.col("ground_any"))
        .then(pl.lit("faa_ground_corroborated"))
        .otherwise(pl.lit("faa_uncorroborated"))
        .alias("label_quality"),
    )
    return data, check


def accuracy(frame):
    n = frame.height
    if not n:
        return {"endpoints": 0}
    result = {
        "endpoints": n,
        "flights": frame["reference_id"].n_unique(),
        "aircraft": frame["icao"].n_unique(),
        "airports": frame["truth"].n_unique(),
        "matched": frame.filter(pl.col("match_status") == "matched").height,
        "ambiguous": frame.filter(pl.col("match_status") == "ambiguous").height,
        "unmatched": frame.filter(pl.col("match_status") == "unmatched").height,
        "observed": frame["observed"].sum(),
        "nn_correct": frame["nn_correct"].sum(),
        "nearest_correct": frame["nearest_correct"].sum(),
        "nn_accuracy": frame["nn_correct"].mean(),
        "nearest_accuracy": frame["nearest_correct"].mean(),
        "candidate_recall": frame["retrieved"].mean(),
        "nn_only_correct": frame.filter(
            pl.col("nn_correct") & ~pl.col("nearest_correct")
        ).height,
        "nearest_only_correct": frame.filter(
            ~pl.col("nn_correct") & pl.col("nearest_correct")
        ).height,
        "both_wrong": frame.filter(
            ~pl.col("nn_correct") & ~pl.col("nearest_correct")
        ).height,
    }
    airport_rates = frame.group_by("truth").agg(
        pl.col("nn_correct").mean(), pl.col("nearest_correct").mean()
    )
    result.update(
        nn_macro_airport_accuracy=airport_rates["nn_correct"].mean(),
        nearest_macro_airport_accuracy=airport_rates["nearest_correct"].mean(),
    )
    # Resample whole aircraft, keeping repeated endpoints/flights together.
    blocks = (
        frame.group_by("icao")
        .agg(
            pl.len().alias("n"),
            (
                pl.col("nn_correct").cast(pl.Int32)
                - pl.col("nearest_correct").cast(pl.Int32)
            )
            .sum()
            .alias("delta"),
        )
        .sort("icao")
    )
    rng = np.random.default_rng(20260917)
    delta = []
    for _ in range(1000):
        indices = rng.integers(0, blocks.height, blocks.height)
        delta.append(
            float(
                blocks["delta"].to_numpy()[indices].sum()
                / blocks["n"].to_numpy()[indices].sum()
            )
        )
    result["nn_minus_nearest_95pct_aircraft_bootstrap"] = np.quantile(
        delta, [0.025, 0.975]
    ).tolist()
    return result


def summarize(data, sources):
    normal = data.filter(pl.col("loss_seconds") == 0)
    groups = {
        "airline_reported": pl.col("label_quality") == "airline_reported",
        "faa_corroborated_all": pl.col("label_quality").is_in(
            ["faa_airline_agreement", "faa_ground_corroborated"]
        ),
        "faa_ground_without_bts": (pl.col("label_quality") == "faa_ground_corroborated")
        & pl.col("bts_reference_id").is_null(),
        "faa_all_resolved_agreement_only": (pl.col("reference_source") == "FAA")
        & (pl.col("truth") != ""),
        "faa_uncorroborated_agreement_only": pl.col("label_quality")
        == "faa_uncorroborated",
    }
    result = {
        "sources": sources,
        "groups": {},
        "label_quality_counts": normal.select(
            *KEY, "reference_source", "truth", "label_quality"
        )
        .unique()
        .group_by("reference_source", "label_quality")
        .len()
        .sort("reference_source", "label_quality")
        .to_dicts(),
    }
    for name, condition in groups.items():
        cohort = normal.filter(condition)
        common = (
            cohort.group_by(KEY)
            .agg(
                (pl.col("match_status") == "matched").all().alias("all_matched"),
                pl.col("source").n_unique().alias("sources"),
            )
            .filter(pl.col("all_matched") & (pl.col("sources") == len(sources)))
            .select(KEY)
        )
        output = {
            "reference_endpoints": cohort.select(KEY).unique().height,
            "common_matched_endpoints": common.height,
            "full_reference": {},
            "source_matched": {},
            "common_matched": {},
            "strata": {},
            "stress": {},
        }
        common_frame = cohort.join(common, on=KEY, how="semi")
        for source in sources:
            sub = cohort.filter(pl.col("source") == source)
            output["full_reference"][source] = accuracy(sub)
            output["source_matched"][source] = accuracy(
                sub.filter(pl.col("match_status") == "matched")
            )
            output["common_matched"][source] = accuracy(
                common_frame.filter(pl.col("source") == source)
            )
        if name not in ("faa_uncorroborated_agreement_only",):
            strata = {
                "large_airport": pl.col("airport_type") == "large_airport",
                "medium_airport": pl.col("airport_type") == "medium_airport",
                "small_airport": pl.col("airport_type") == "small_airport",
                "unseen_aircraft": ~pl.col("seen_aircraft"),
                "unseen_airport": ~pl.col("seen_airport"),
                "non_us": pl.col("country") != "US",
                "alaska_hawaii_territories": pl.col("region").is_in(["US-AK", "US-HI"])
                | pl.col("country").is_in(["PR", "VI", "GU", "AS", "MP"]),
                "takeoff": pl.col("endpoint") == "takeoff",
                "landing": pl.col("endpoint") == "landing",
                "ground_only_adsbx": pl.col("ground_adsbx") & ~pl.col("ground_adsblol"),
                "ground_both_networks": pl.col("ground_adsbx")
                & pl.col("ground_adsblol"),
            }
            for group, expr in strata.items():
                output["strata"][group] = {
                    source: accuracy(
                        common_frame.filter(expr & (pl.col("source") == source))
                    )
                    for source in sources
                }
        stress = data.filter(condition & (pl.col("loss_seconds") > 0)).join(
            common, on=KEY, how="semi"
        )
        if stress.height:
            # Baseline on exactly the same selected endpoints as both stress levels.
            common_stress = (
                stress.group_by(KEY)
                .agg(pl.struct("source", "loss_seconds").n_unique().alias("n"))
                .filter(pl.col("n") == len(sources) * 2)
                .select(KEY)
            )
            for seconds in (0, 60, 180):
                sub = data.filter(condition & (pl.col("loss_seconds") == seconds)).join(
                    common_stress, on=KEY, how="semi"
                )
                output["stress"][str(seconds)] = {
                    source: accuracy(sub.filter(pl.col("source") == source))
                    for source in sources
                }
        result["groups"][name] = output
    return result


def render_report(output, metrics, protocol):
    sources = metrics["sources"]
    lines = [
        "# Airport evaluation with independent network segmentation",
        "",
        f"Dates: {', '.join(protocol['dates'])}. Frozen checkpoint SHA256: `{protocol['model_sha256']}`.",
        "",
        "Each network is segmented from its own full-day observations. Reference airport labels and event times do not enter segmentation, candidate generation, NN inference, or nearest-airport selection. Both methods receive identical observations within each native segment. Nearest queries the entire supported airport catalog at the first/last valid endpoint observation.",
        "",
        "The primary table measures airport assignment on exactly the same reference endpoints uniquely matched in every network. The full-reference table additionally counts unmatched and ambiguous flights as failures; those are detection/association failures, not proven airport-classification errors.",
        "",
        "## Common matched endpoints",
        "",
        "| Reference cohort | Endpoints | "
        + " | ".join(f"{s} NN | {s} nearest" for s in sources)
        + " |",
        "|---|---:|" + "---:|---:|" * len(sources),
    ]
    display = {
        "airline_reported": "BTS airline reported",
        "faa_corroborated_all": "FAA corroborated (includes airlines)",
        "faa_ground_without_bts": "FAA ground corroborated, no BTS match",
        "faa_all_resolved_agreement_only": "All resolved FAA — label agreement only",
        "faa_uncorroborated_agreement_only": "Uncorroborated FAA — label agreement only",
    }

    def row(label, stats):
        n = stats[sources[0]]["endpoints"]
        values = []
        for source in sources:
            m = stats[source]
            values.extend(
                [f"{m['nn_accuracy']:.2%}", f"{m['nearest_accuracy']:.2%}"]
                if m["endpoints"]
                else ["—", "—"]
            )
        return f"| {label} | {n:,} | " + " | ".join(values) + " |"

    for name, group in metrics["groups"].items():
        lines.append(row(display[name], group["common_matched"]))
    lines += [
        "",
        "BTS labels are airline-reported airport events. FAA corroborated labels require a conservative airport-code match plus either BTS agreement or sustained ground/low-speed evidence at the labeled airport in at least one network, using unmasked tracks within 15 minutes of the FAA event. FAA/BTS conflicts are excluded from corroborated cohorts. The evidence check never uses either method's predictions. FAA code resolution rejects ambiguous three-letter route points and worldwide IATA fallback.",
        "Reference records assigning overlapping flights to the same aircraft within one source (more than 60 seconds of overlap) are marked as identity/time conflicts on both records and excluded from the reliable cohorts. They remain in label_review.parquet; their count is reported in metrics.json. This rule uses reference metadata alone.",
        "Ground corroboration requires at least three points spanning ten seconds within 2 km of the airport center, with on_ground true OR altitude within 300 ft of airport elevation and speed <=100 knots. This geofence can miss distant runways at large airports. Absence of corroboration never establishes that a reference is wrong.",
        "",
        "Observational corroboration selects flights with better terminal visibility and is not an independent third truth source. Its scores must be read with the uncorroborated stratum and controlled reception-loss results. Neither resolved FAA agreement nor consensus between predictions is treated as verified accuracy. Cohorts overlap and must not be summed.",
        "",
        "## Full reference denominators",
        "",
        "| Cohort | Source | Reference endpoints | Matched | Ambiguous | Unmatched | NN correct / all | Nearest correct / all |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in metrics["groups"].items():
        for source, m in group["full_reference"].items():
            if m["endpoints"]:
                lines.append(
                    f"| {display[name]} | {source} | {m['endpoints']:,} | {m['matched']:,} | {m['ambiguous']:,} | {m['unmatched']:,} | {m['nn_accuracy']:.2%} | {m['nearest_accuracy']:.2%} |"
                )
    for name in ("airline_reported", "faa_ground_without_bts"):
        group = metrics["groups"][name]
        lines += [
            "",
            f"## Harder strata: {display[name]}",
            "",
            "| Stratum | Endpoints | "
            + " | ".join(f"{s} NN | {s} nearest" for s in sources)
            + " |",
            "|---|---:|" + "---:|---:|" * len(sources),
        ]
        for key in (
            "small_airport",
            "medium_airport",
            "large_airport",
            "unseen_aircraft",
            "unseen_airport",
            "non_us",
            "alaska_hawaii_territories",
            "ground_only_adsbx",
            "ground_both_networks",
        ):
            lines.append(row(key, group["strata"][key]))
        lines += [
            "",
            "Reception loss: deterministic 20% aircraft sample; remove observations in the first/last 60 or 180 seconds of each native segment, holding segment boundaries fixed. This tests airport inference under missing terminal data, not the segmentation algorithm under reception loss. The zero-loss row uses exactly the same sample.",
            "",
            "| Seconds removed | Endpoints | "
            + " | ".join(f"{s} NN | {s} nearest" for s in sources)
            + " |",
            "|---|---:|" + "---:|---:|" * len(sources),
        ]
        for key, stats in group["stress"].items():
            lines.append(row(key, stats))
    lines += [
        "",
        "## Scope and reproducibility",
        "",
        "- Aircraft/airport novelty includes both February pretraining and subsequent broad fine-tuning. All evaluated dates are excluded from training and model selection; March 1 and June 5–6 were examined previously and are not newly untouched test dates.",
        "- March 1 is the only locally available ADSBx date outside model training/selection. June 5–6 extend ADSB.lol testing; no ADSBx scores are invented for unavailable dates.",
        "- References are same-UTC-day flights with known aircraft identity. Cross-day and invalid-identity/time rows are saved in excluded-flights Parquet files. BTS cache construction excludes cancelled/diverted flights and flights <=30 minutes. FAA cache construction excludes flights <=20 minutes; its aircraft associations may depend on ADSB.lol, limiting conclusions about aircraft unseen by that network.",
        "- Missing segments, missing candidates, and abstentions remain visible. Unlabeled world flights are outside reference recall: this is not a measurement of worldwide flight-detection precision or all VFR/private-airport traffic.",
        "- Full counts, per-airport macro accuracy, paired NN/nearest wins, and 95% aircraft-cluster bootstrap intervals for their accuracy difference are in metrics.json. Small strata should be interpreted with their sample sizes.",
        "- endpoint_results.parquet preserves reference/native times, match_status, predictions, label quality, observation coordinates, candidate lists, and reception-loss scenario. label_review.parquet includes every unresolved/conflicting/uncorroborated label, not just algorithm errors. errors.parquet retains all disagreements on corroborated or airline references.",
        "- protocol.json records input files, implementation versions, model lineage and policy before scoring. No model retraining, per-airport rule tuning, or prediction-based label correction was performed.",
    ]
    (output / "results.md").write_text("\n".join(lines) + "\n")


def run(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    report_policy = {
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_conflicts": "Exclude both reference records when flights of the same aircraft and source overlap by more than 60 seconds; no predictions used.",
        "corroboration": "Conservative airport code plus BTS agreement or ground evidence in any available network; exclude reference conflicts. Observational selection is reported explicitly.",
    }
    policy_path = output / "report_policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != report_policy:
        raise ValueError(
            "Report policy changed; preserve the prior report before recomputing"
        )
    policy_path.write_text(json.dumps(report_policy, indent=2) + "\n")
    references = pl.concat(
        [
            pl.read_parquet(output / f"references-{day}.parquet")
            for day in protocol["dates"]
        ]
    )
    frame = pl.read_parquet(output / "endpoint_predictions.parquet")
    data, check = annotate(frame, references, protocol["lineage"], load_catalog())
    data.write_parquet(output / "endpoint_results.parquet")
    check.write_parquet(output / "bts_crosscheck.parquet")
    normal = data.filter(pl.col("loss_seconds") == 0)
    normal.filter(
        pl.col("label_quality").is_in(
            [
                "unresolved",
                "reference_conflict",
                "reference_identity_time_conflict",
                "faa_uncorroborated",
            ]
        )
    ).write_parquet(output / "label_review.parquet")
    normal.filter(
        pl.col("label_quality").is_in(
            ["airline_reported", "faa_airline_agreement", "faa_ground_corroborated"]
        )
        & (~pl.col("nn_correct") | ~pl.col("nearest_correct"))
    ).write_parquet(output / "errors.parquet")
    metrics = summarize(data, protocol["sources"])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    render_report(output, metrics, protocol)
    print(output / "results.md", flush=True)
    print(
        json.dumps(
            {k: v["common_matched"] for k, v in metrics["groups"].items()}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
