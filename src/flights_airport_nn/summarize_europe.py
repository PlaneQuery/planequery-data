"""Assemble completed European runs, validate denominators and retain diagnostics."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import polars as pl

from flights_airport_nn.evaluate_native_sources import KEY
from flights_airport_nn.report_native_evaluation import accuracy


def summarize(root):
    names = ["eurocontrol-june01", "eurocontrol-june15", "opensky-march01"]
    lines = [
        "# European airport evaluation",
        "",
        "Frozen model; independent native segmentation for each network; no retraining or prediction-driven label corrections.",
        "",
        "## Main results",
        "",
        "Scores are exact airport-identifier accuracy per **European endpoint**. Where two networks are tested, these rows use precisely the same reference endpoints uniquely matched in both. Single-network June 15 uses its own matched sample.",
        "",
        "| Reference | Date | Network | Endpoints | Airports | NN | Nearest | NN minus nearest, 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    coverage, small, stress, reviews, verification, extras = [], [], [], [], [], {}
    model_hashes = set()
    for name in names:
        output = root / name
        protocol = json.loads((output / "protocol.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())
        model_hashes.add(protocol["model_sha256"])
        assert (
            hashlib.sha256(Path(protocol["model"]).read_bytes()).hexdigest()
            == protocol["model_sha256"]
        )
        assert not set(protocol["dates"]) & set(
            protocol["lineage"]["train_dates"] + protocol["lineage"]["validation_dates"]
        )
        data = pl.read_parquet(output / "endpoint_results.parquet")
        normal = data.filter(pl.col("loss_seconds") == 0)
        refs = pl.concat(
            [
                pl.read_parquet(output / f"references-{d}.parquet")
                for d in protocol["dates"]
            ]
        )
        assert normal.height == refs.height * 2 * len(protocol["sources"])
        assert not data.select(*KEY, "source", "loss_seconds").is_duplicated().any()
        assert normal.filter(
            (pl.col("match_status") != "matched")
            & (pl.col("nn_correct") | pl.col("nearest_correct"))
        ).is_empty()
        observed = data.filter(pl.col("observed"))
        assert observed.filter(
            (
                pl.col("observation_time")
                < pl.col("segment_start") + pl.duration(seconds=pl.col("loss_seconds"))
            )
            | (
                pl.col("observation_time")
                > pl.col("segment_end") - pl.duration(seconds=pl.col("loss_seconds"))
            )
        ).is_empty()
        valid = normal.filter(
            (pl.col("continent") == "EU")
            & (pl.col("truth") != "")
            & ~pl.col("reference_time_conflict")
        )
        common = valid.filter(pl.col("common_matched"))
        sets = [
            set(common.filter(pl.col("source") == s).select(KEY).iter_rows())
            for s in protocol["sources"]
        ]
        assert all(k == sets[0] for k in sets)
        label = (
            "EUROCONTROL"
            if metrics["reference_source"] == "EUROCONTROL"
            else "OpenSky estimates (agreement only)"
        )
        day = ", ".join(protocol["dates"])
        for source, groups in metrics["groups"]["all_european_endpoints"].items():
            s, full = groups["common_matched"], groups["all_reference"]
            actual = common.filter(pl.col("source") == source)
            assert s["endpoints"] == actual.height
            assert s["nn_correct"] == actual["nn_correct"].sum()
            assert s["nearest_correct"] == actual["nearest_correct"].sum()
            ci = s["nn_minus_nearest_95pct_aircraft_bootstrap"]
            lines.append(
                f"| {label} | {day} | {source} | {s['endpoints']:,} | {s['airports']:,} | {s['nn_accuracy']:.2%} | {s['nearest_accuracy']:.2%} | {100 * ci[0]:+.2f} to {100 * ci[1]:+.2f} pp |"
            )
            coverage.append(
                f"| {label} | {day} | {source} | {full['matched']:,}/{full['endpoints']:,} ({full['matched'] / full['endpoints']:.2%}) | {full['ambiguous']:,} | {full['unmatched']:,} | {full['nn_accuracy']:.2%} | {full['nearest_accuracy']:.2%} |"
            )
            s = metrics["groups"]["small_airports"][source]["common_matched"]
            if s["endpoints"]:
                small.append(
                    f"| {label} | {day} | {source} | {s['endpoints']:,} | {s['airports']:,} | {s['nn_accuracy']:.2%} | {s['nearest_accuracy']:.2%} |"
                )
            s = metrics["stress"]["180"][source]["masked"]
            if s["endpoints"]:
                stress.append(
                    f"| {label} | {day} | {source} | {s['endpoints']:,} | {s['nn_accuracy']:.2%} | {s['nearest_accuracy']:.2%} |"
                )
        audit = json.loads((output / "disagreement_audit.json").read_text())
        for row in audit["counts"]:
            reviews.append({"run": name, **row})
        # Within-Europe routes separate airport inference from requiring
        # coverage at a distant non-European endpoint for time matching.
        eu_airports = set(valid["truth"])
        intra = refs.filter(
            pl.col("takeoff_truth").is_in(eu_airports)
            & pl.col("landing_truth").is_in(eu_airports)
        )["reference_id"]
        extras[name] = {
            source: {
                "intra_europe_full_reference": accuracy(
                    valid.filter(
                        (pl.col("source") == source)
                        & pl.col("reference_id").is_in(intra.implode())
                    )
                ),
                "intra_europe_common_matched": accuracy(
                    common.filter(
                        (pl.col("source") == source)
                        & pl.col("reference_id").is_in(intra.implode())
                    )
                ),
                "countries": valid.filter(pl.col("source") == source)
                .group_by("country")
                .agg(
                    pl.len().alias("reference_endpoints"),
                    (pl.col("match_status") == "matched")
                    .sum()
                    .alias("matched_endpoints"),
                )
                .sort("country")
                .to_dicts(),
            }
            for source in protocol["sources"]
        }
        verification.append(
            {
                "run": name,
                "reference_flights": refs.height,
                "normal_endpoint_rows": normal.height,
                "sources": protocol["sources"],
                "passed": [
                    "model hash unchanged",
                    "training/selection dates disjoint",
                    "every reference endpoint retained per network",
                    "no duplicate endpoint scenarios",
                    "unmatched rows never counted correct",
                    "observations within native segments and reception-loss bounds",
                    "identical common endpoint sets",
                    "reported scores recomputed from rows",
                ],
            }
        )
    assert len(model_hashes) == 1
    lines += [
        "",
        "## Coverage and full-population correctness",
        "",
        "An unmatched flight is not an airport-classification error, but it remains a failure to produce the reference flight in these full-population scores. Both native departure and arrival must time-match; poor coverage outside Europe can prevent a European endpoint from matching. Intra-European-route diagnostics are in additional_metrics.json.",
        "",
        "| Reference | Date | Network | Matched/reference endpoints | Ambiguous | Unmatched | NN full population | Nearest full population |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        *coverage,
        "",
        "## Small airports",
        "",
        "| Reference | Date | Network | Endpoints | Airports | NN | Nearest |",
        "|---|---|---|---:|---:|---:|---:|",
        *small,
        "",
        "## Remove the first and last 180 seconds",
        "",
        "Fixed native boundaries, deterministic 20% aircraft sample, paired across networks. Same-sample normal scores and the 60-second condition are in each run's metrics.json. This isolates endpoint reception loss after segmentation; it does not simulate how segmentation would change under a network outage.",
        "",
        "| Reference | Date | Network | Endpoints | NN | Nearest |",
        "|---|---|---|---:|---:|---:|",
        *stress,
        "",
        "## Reference quality and scope",
        "",
        "- EUROCONTROL is an independent Network Manager reference for scheduled/non-scheduled commercial flights. Its departure time is off-block, arrival time is landing; aircraft/time matching allows -5/+60 minutes around off-block and +/-15 minutes around landing. The stricter shared subset is in each run's report. [Official metadata](https://www.eurocontrol.int/sites/default/files/2025-04/eurocontrol-aviation-data-repository-research-metadata.pdf).",
        "- This research sample excludes general aviation, military and other categories. Both EUROCONTROL dates are Saturdays in June 2024: historical/geographic generalization, with limited season and weekday coverage. The model and current airport catalog were used unchanged; this is not a reconstruction of a 2024 deployment.",
        "- Daily cached EUROCONTROL references already require aircraft registration mapping and resolvable airports. The daily and monthly June 1 caches agree on both airports and arrival times for all 28,673 paired registration/off-block records; one ICAO mapping differs (LX-VAM: daily 4d021b, monthly 4d0215). Daily records were selected before scoring. Identity mappings and NM updates can still be wrong.",
        "- Primary scores exclude only unresolved labels, non-European endpoints and overlapping-reference identity/time conflicts using reference metadata. All excluded/unmatched rows remain available. No prediction-driven ground-truth corrections were made.",
        "- OpenSky supplies estimated airports and observed flight boundaries, so its numbers measure agreement, not independently verified accuracy. Nearby airfields and shared airport locations can produce misleading disagreements. [OpenSky documentation](https://openskynetwork.github.io/opensky-api/python.html).",
        "- Error audits check ground/low-and-slow observations near the reference and predicted airports, plus trace registration metadata. They distinguish supported-reference errors, nearby alternatives, candidate reference problems and insufficient evidence. These same-network checks cannot establish independent truth, and do not alter the primary scores. Ground-corroborated subsets favor good reception; they are secondary diagnostics.",
        "- Confidence intervals resample aircraft, accounting for repeated flights but not systematic reference-label uncertainty. Small-airport samples in EUROCONTROL are limited; the wider OpenSky small-airport sample has weaker labels. No universal performance claim follows from this test.",
        "",
        "## Disagreement review (scores unchanged)",
        "",
    ]
    lines += [
        "| Run | Network | Physical evidence category | Endpoints |",
        "|---|---|---|---:|",
    ]
    for row in reviews:
        lines.append(
            f"| {row['run']} | {row['source']} | {row['review_status']} | {row['len']:,} |"
        )
    lines += [
        "",
        "These are all matched European disagreements, not just the common-network subset. Alternative support without reference support is a review flag, not proof of a wrong label. Missing reception leaves many cases unresolved.",
        "",
        "A concrete identifier issue is LSZM: the current catalog resolves it to Mollis, but some OpenSky records have sustained ground observations at LFSB (Basel), about 129 km away. Mollis adopted LSZM on 2021-01-01 ([airport announcement](https://www.mollisairport.ch/post/aus-lsmf-wird-lszm)); the catalog also retains LSZM as a historical Basel keyword. This suggests stale identifier usage in these OpenSky records. A blanket LSZM remapping would corrupt actual Mollis flights, so no remapping was applied.",
        "",
        "## Saved data",
        "",
    ]
    for name in names:
        lines.append(
            f"- [{name} results]({name}/results.md), [paired reference/native flights]({name}/paired_endpoints.parquet), [all endpoint scenarios]({name}/endpoint_results.parquet), [disagreement audit]({name}/disagreement_audit.parquet)."
        )
    lines += [
        "",
        "Read paired_endpoints.parquet with Polars or pandas. Columns include reference_id, endpoint, truth and each network's flight_id, segment_start, segment_end, match_status, nn and nearest. Complete reference flight details are in references-DATE.parquet and join on reference_id. See input_inventory.json, reference_inventory.json and verification.json for populations and checks.",
    ]
    (root / "results.md").write_text("\n".join(lines) + "\n")
    (root / "additional_metrics.json").write_text(json.dumps(extras, indent=2) + "\n")
    (root / "disagreement_summary.json").write_text(
        json.dumps(reviews, indent=2) + "\n"
    )
    (root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    sources = [
        Path(__file__).with_name(name)
        for name in (
            "summarize_europe.py",
            "evaluate_europe.py",
            "audit_europe_errors.py",
            "prepare_europe_traces.py",
            "evaluate_native_sources.py",
            "report_native_evaluation.py",
            "reference_labels.py",
            "broader_evaluation.py",
        )
    ]
    (root / "analysis_provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                },
                "versions": {
                    p: version(p) for p in ("polars", "numpy", "torch", "pyarrow")
                },
                "model_sha256": next(iter(model_hashes)),
            },
            indent=2,
        )
        + "\n"
    )
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
