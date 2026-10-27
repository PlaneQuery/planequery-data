"""Paired cohort and boundary intervention audit; never modifies source caches.

Run from the repository root with .venv/bin/python. Inputs are existing current
evaluation caches. --rerun validates every differing segment pair plus 200
deterministically sampled agreement controls with fresh, identical observations.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, timedelta
import hashlib
import html
import json
from pathlib import Path

import polars as pl

from data_engineering.utils import OUTPUT_DIR
from flights.airport_nn import AlgorithmNNFlightSource, OpenSkyNNFlightSource
from flights.evaluation.cache import EVALUATION_CACHE_VERSION, SourceFingerprint

SIDES = ("takeoff", "landing")
KEY = ["icao", "gold_takeoff_time", "gold_landing_time"]
PAIR = ["icao", *[f"{s}_{e}_time" for s in ("algorithm", "opensky") for e in SIDES]]


def read_comparison(source, gold, day, provider, manifests):
    fingerprint = SourceFingerprint.from_source(source)
    root = OUTPUT_DIR / "data/cache/evaluation" / f"cache_version={EVALUATION_CACHE_VERSION}"
    paths = list(root.glob(
        f"test_source={source.name}/test_source_version={source.version}/"
        f"test_source_config_fingerprint={fingerprint.config_fingerprint}/"
        f"gold_source={gold}/**/adsb_source={provider}/**/test_date={day}/manifest.json"
    ))
    assert len(paths) == 1, paths
    p = paths[0]
    manifest = json.loads(p.read_text())
    manifests.append(dict(path=str(p), **manifest))
    raw = pl.read_parquet(p.parent / "result.parquet")
    d = raw.filter(pl.col("match_status") == "both")
    columns = {"icao_df1": "icao", "callsign_df1": "gold_callsign"}
    for e in SIDES:
        for suffix in ("time", "airport_id", "airport_ident"):
            columns[f"{e}_{suffix}_df1"] = f"gold_{e}_{suffix}"
            columns[f"{e}_{suffix}"] = f"{source.name.split('-')[0]}_{e}_{suffix}"
    d = d.select(*[pl.col(k).alias(v) for k, v in columns.items()])
    assert d.unique(KEY).height == d.height
    return d, raw.filter(pl.col("match_status") != "df0_only").height


def correct(source, side):
    if side == "both":
        return correct(source, "takeoff") & correct(source, "landing")
    return (pl.col(f"{source}_{side}_airport_id") == pl.col(f"gold_{side}_airport_id")).fill_null(False)


def metrics(d, source):
    return {"flights": d.height, **{
        e: {"correct": int(d.select(correct(source, e).sum()).item()),
            "accuracy_pct": d.select(correct(source, e).mean() * 100).item()}
        for e in (*SIDES, "both")
    }}


def paired_metrics(d):
    result = {s: metrics(d, s) for s in ("algorithm", "opensky")}
    result["paired"] = {}
    for e in (*SIDES, "both"):
        a, o = correct("algorithm", e), correct("opensky", e)
        result["paired"][e] = d.select(
            (a & ~o).sum().alias("algorithm_only_correct"),
            (o & ~a).sum().alias("opensky_only_correct"),
            (~a & ~o).sum().alias("neither_correct"),
            (a & o).sum().alias("both_correct"),
            ((a.cast(pl.Int64) - o.cast(pl.Int64)).mean() * 100).alias("algorithm_advantage_pp"),
        ).row(0, named=True)
    return result


def cohort(a, o):
    gold_cols = [c for c in o.columns if c.startswith("gold_") and c not in KEY]
    check = a.join(o, on=KEY, suffix="_o", validate="1:1")
    assert check.select(*[pl.col(c).eq_missing(pl.col(c + "_o")).all() for c in gold_cols]).row(0) == (True,) * len(gold_cols)
    d = check.drop(*[c + "_o" for c in gold_cols])
    for e in SIDES:
        d = d.with_columns(
            ((pl.col(f"opensky_{e}_time") - pl.col(f"algorithm_{e}_time")).dt.total_milliseconds() / 1000).alias(f"{e}_opensky_minus_algorithm_s"),
            *[((pl.col(f"{s}_{e}_time") - pl.col(f"gold_{e}_time")).dt.total_milliseconds() / 1000).alias(f"{s}_{e}_gold_error_s") for s in ("algorithm", "opensky")],
        )
    return d


def summarize(a, o, d, gold_n):
    out = {"gold_flights": gold_n,
           "all_matched": {"algorithm": metrics(a, "algorithm"), "opensky": metrics(o, "opensky")},
           "shared": paired_metrics(d),
           "exclusive": {"algorithm": metrics(a.join(o.select(KEY), on=KEY, how="anti"), "algorithm"),
                         "opensky": metrics(o.join(a.select(KEY), on=KEY, how="anti"), "opensky")},
           "boundary_sensitivity": {}, "gold_time_sensitivity": {}, "timing": {}, "gap_concentration": {}}
    for seconds in (30, 60, 120, 300, 600, 900):
        small = d.filter(pl.all_horizontal(*[pl.col(f"{e}_opensky_minus_algorithm_s").abs() <= seconds for e in SIDES]))
        out["boundary_sensitivity"][str(seconds)] = paired_metrics(small)
    for seconds in (300, 900, 1800):
        small = d.filter(pl.all_horizontal(*[pl.col(f"{s}_{e}_gold_error_s").abs() <= seconds for s in ("algorithm", "opensky") for e in SIDES]))
        out["gold_time_sensitivity"][str(seconds)] = paired_metrics(small)
    out["nonoverlapping_source_segments"] = d.filter(pl.max_horizontal("algorithm_takeoff_time", "opensky_takeoff_time") > pl.min_horizontal("algorithm_landing_time", "opensky_landing_time")).height
    for e in SIDES:
        out["timing"][e] = {}
        net = correct("algorithm", e).cast(pl.Int64) - correct("opensky", e).cast(pl.Int64)
        out["gap_concentration"][e] = {}
        for seconds in (60, 120, 300, 600):
            large = d.filter(pl.col(f"{e}_opensky_minus_algorithm_s").abs() > seconds)
            excess = large.select(net.sum()).item()
            total_excess = d.select(net.sum()).item()
            out["gap_concentration"][e][str(seconds)] = {"flights": large.height, "net_excess_opensky_errors": excess,
                                                        "share_of_net_accuracy_gap_pct": 100 * excess / total_excess if total_excess else None}
        for group, sub in (("all", d), ("algorithm_only_correct", d.filter(correct("algorithm", e) & ~correct("opensky", e))),
                           ("opensky_only_correct", d.filter(correct("opensky", e) & ~correct("algorithm", e)))):
            err = pl.col(f"{e}_opensky_minus_algorithm_s")
            out["timing"][e][group] = {"flights": sub.height, **sub.select(
                err.median().alias("median_opensky_minus_algorithm_s"),
                err.abs().median().alias("median_absolute_boundary_difference_s"),
                err.abs().quantile(.9, interpolation="linear").alias("p90_absolute_boundary_difference_s"),
                *[v for s in ("algorithm", "opensky") for v in (
                    pl.col(f"{s}_{e}_gold_error_s").abs().mean().alias(f"{s}_mae_s"),
                    pl.col(f"{s}_{e}_gold_error_s").abs().median().alias(f"{s}_median_abs_error_s"),
                    pl.col(f"{s}_{e}_gold_error_s").mean().alias(f"{s}_bias_s"),
                )],
            ).row(0, named=True)}
    return out


def rerun(pairs, outdir, day):
    import torch
    from airports.runtime import airport_catalog_for_date
    from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
    from data_engineering.adsb.canonical import adsb_day_is_complete
    from flights.airport_nn.features import COLUMNS, trajectories
    from flights.airport_nn.inference import predict
    from flights.airport_nn.model import load_model
    from flights.airport_nn.source import DEFAULT_MODEL_PATH

    catalog = airport_catalog_for_date(day)
    torch.set_num_threads(4)
    model = load_model(DEFAULT_MODEL_PATH, device="cpu")
    different = pl.any_horizontal(*[~pl.col(f"algorithm_{e}_airport_id").eq_missing(pl.col(f"opensky_{e}_airport_id")) for e in SIDES])
    unique = pairs.select(*PAIR, *[f"{s}_{e}_airport_id" for s in ("algorithm", "opensky") for e in SIDES]).unique().sort(PAIR)
    assert unique.unique(PAIR).height == unique.height, "Cross-reference cached predictions must agree for identical source segments"
    chosen = pl.concat([unique.filter(different).with_columns(pl.lit("disagreement").alias("selection")),
                        unique.filter(~different).sample(n=min(200, unique.filter(~different).height), seed=20260301).with_columns(pl.lit("agreement_control").alias("selection"))]).with_row_index("pair_id")
    chosen.write_parquet(outdir / "intervention_pairs.parquet")
    outputs = []
    for batch in chosen.iter_slices(256):
        start = min(batch[f"{s}_takeoff_time"].min() for s in ("algorithm", "opensky"))
        end = max(batch[f"{s}_landing_time"].max() for s in ("algorithm", "opensky")) + timedelta(milliseconds=1)
        for n in range((end.date() - start.date()).days + 1):
            assert adsb_day_is_complete(ADSBSource.OPENSKY, start.date() + timedelta(days=n)), "Audit requires pre-existing local observations"
        messages = read_adsb(start, end, columns=COLUMNS, icaos=batch["icao"].unique().to_list(), source=ADSBSource.OPENSKY)
        # All four settings use exactly the same checkpoint, catalog and messages.
        # AO changes only OpenSky's departure; OA changes only its arrival.
        for variant, takeoff_source, landing_source in (("OO", "opensky", "opensky"), ("AA", "algorithm", "algorithm"),
                                                       ("AO", "algorithm", "opensky"), ("OA", "opensky", "algorithm")):
            flights = batch.select("icao", pl.col(f"{takeoff_source}_takeoff_time").alias("takeoff_time"), pl.col(f"{landing_source}_landing_time").alias("landing_time"))
            valid = flights.select(pl.col("landing_time") >= pl.col("takeoff_time")).to_series().to_list()
            # The evaluator's one-hour tolerance can pair nonoverlapping short
            # segments. Keep those rows visible; invalid hybrids yield no points.
            if variant in ("OO", "AA"):
                assert all(valid)
            metadata = []
            def endpoints():
                for ep in trajectories(flights, messages, catalog, include_nearest=False):
                    metadata.append({"candidate_ids": ep.candidate_ids,
                                     "points": ep.values.shape[1],
                                     "first_distance_feature": float(ep.values[0, 0, 4]) if ep.values.size else None})
                    yield ep
            predictions = predict(endpoints(), model)
            for pred, meta in zip(predictions, metadata):
                outputs.append({"pair_id": batch["pair_id"][pred["row"]], "variant": variant,
                                "endpoint": pred["endpoint"], "valid_segment": valid[pred["row"]], "airport_id": pred["airport_id"],
                                "airport_ident": pred["airport_ident"], "confidence": pred["confidence"], **meta})
        pl.DataFrame(outputs).write_parquet(outdir / "intervention_predictions.parquet")
        print(f"Rerun {min(batch['pair_id'].max()+1, chosen.height)}/{chosen.height} pairs", flush=True)
    return chosen, pl.DataFrame(outputs)


def summarize_interventions(shared, outdir):
    pairs = pl.read_parquet(outdir / "intervention_pairs.parquet")
    predictions = pl.read_parquet(outdir / "intervention_predictions.parquet")
    assert predictions.height == pairs.height * 8
    assert predictions.unique(["pair_id", "variant", "endpoint"]).height == predictions.height
    assert all(r["airport_id"] is None or r["airport_id"] in r["candidate_ids"] for r in predictions.select("airport_id", "candidate_ids").to_dicts())
    joined = predictions.join(pairs, on="pair_id", validate="m:1")
    baseline = []
    for variant, source in (("OO", "opensky"), ("AA", "algorithm")):
        for endpoint in SIDES:
            for selection in ("disagreement", "agreement_control"):
                d = joined.filter((pl.col("variant") == variant) & (pl.col("endpoint") == endpoint) & (pl.col("selection") == selection))
                baseline.append({"variant": variant, "endpoint": endpoint, "selection": selection,
                                 "flights": d.height, "reproduced_cached_prediction": d.select(pl.col("airport_id").eq_missing(pl.col(f"{source}_{endpoint}_airport_id")).sum()).item()})
    rows = []
    for ref in ("bts", "sfdps"):
        d = shared.filter(pl.col("reference") == ref).join(pairs.select(*PAIR, "pair_id", "selection"), on=PAIR, validate="m:1")
        for endpoint in SIDES:
            ep = d.join(predictions.filter(pl.col("endpoint") == endpoint), on="pair_id")
            # This subgroup covers every endpoint where the algorithm repairs
            # an OpenSky mistake, regardless of whether the other end differs.
            ep = ep.filter(correct("algorithm", endpoint) & ~correct("opensky", endpoint))
            for variant in ("OO", "AA", "AO", "OA"):
                sub = ep.filter(pl.col("variant") == variant)
                truth = pl.col(f"gold_{endpoint}_airport_id")
                rows.append({"reference": ref, "endpoint": endpoint, "variant": variant,
                             "flights": sub.height,
                             "correct": sub.select((pl.col("airport_id") == truth).fill_null(False).sum()).item(),
                             # Python membership avoids list.contains().sum()
                             # miscounting sliced multi-chunk joined List arrays.
                             "gold_in_candidates": sum(gold_id is not None and gold_id in candidates for gold_id, candidates in sub.select(f"gold_{endpoint}_airport_id", "candidate_ids").iter_rows()),
                             "no_observations": sub.filter(pl.col("points") == 0).height,
                             "invalid_hybrid": sub.filter(~pl.col("valid_segment")).height})
                assert rows[-1]["correct"] <= rows[-1]["gold_in_candidates"]
    pl.DataFrame(baseline).write_csv(outdir / "rerun_validation.csv")
    pl.DataFrame(rows).write_csv(outdir / "boundary_interventions.csv")
    # Auditable endpoint examples include every variant, candidate list, and
    # both gold labels when BTS and SFDPS cover the same physical flight.
    examples = shared.join(pairs.select(*PAIR, "pair_id", "selection"), on=PAIR, validate="m:1").join(predictions, on="pair_id")
    examples.write_parquet(outdir / "intervention_details.parquet")
    return {"selection": pairs.group_by("selection").len().to_dicts(), "baseline_reproduction": baseline,
            "algorithm_correct_opensky_wrong": rows,
            "scope": "Every shared source-segment pair with any unequal cached airport prediction, plus 200 deterministic agreement controls. Four variants keep observations, model and catalog fixed: OO=OpenSky boundaries, AA=algorithm boundaries, AO=algorithm departure/OpenSky arrival, OA=OpenSky departure/algorithm arrival. Do not interpret selected-subset accuracy as full-population accuracy."}


def write_report(report, outdir):
    def table(headers, rows):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"
    accuracy_rows, overview = [], []
    for ref, data in report["references"].items():
        for scope in ("all_matched", "shared", "exclusive"):
            for source in ("algorithm", "opensky"):
                m = data[scope][source]
                accuracy_rows.append({"reference": ref, "cohort": scope, "source": source, "flights": m["flights"],
                                      **{f"{e}_accuracy_pct": m[e]["accuracy_pct"] for e in (*SIDES, "both")}})
        for e in SIDES:
            raw = data["all_matched"]["algorithm"][e]["accuracy_pct"] - data["all_matched"]["opensky"][e]["accuracy_pct"]
            paired = data["shared"]["paired"][e]["algorithm_advantage_pp"]
            overview.append([ref.upper(), e, f"{data['shared']['algorithm']['flights']:,}",
                             f"{data['shared']['algorithm'][e]['accuracy_pct']:.2f}%", f"{data['shared']['opensky'][e]['accuracy_pct']:.2f}%",
                             f"{raw:.2f} pp", f"{paired:.2f} pp", f"{100*paired/raw:.1f}%"])
    pl.DataFrame(accuracy_rows).write_csv(outdir / "airport_accuracy.csv")
    parts = [f"<h1>OpenSky NN airport accuracy: {report['date']}</h1>",
             "<p>The airport accuracy gap persists on shared flights. Boundary timing and the resulting endpoint observations explain the paired prediction differences; cohort composition contributes much less.</p>",
             "<h2>Same-flight comparison</h2>",
             table(["Reference", "Endpoint", "Shared flights", "Algorithm NN", "OpenSky NN", "Original gap", "Shared gap", "Gap retained"], overview),
             "<p>Both methods use OpenSky observations and the same NN checkpoint. Shared means both sources independently match the same reference flight using aircraft identity and times, without using airport labels. It does not mean their boundaries are identical. The original gap compares each source's own matched cohort. Gap retained is a descriptive standardization, not an additive causal decomposition.</p>",
             "<h2>Where the gap occurs</h2>"]
    rows = []
    timing = []
    for ref, data in report["references"].items():
        close = data["boundary_sensitivity"]["60"]
        rows.append([ref.upper(), close["algorithm"]["flights"], *[f"{close['paired'][e]['algorithm_advantage_pp']:.2f} pp" for e in SIDES]])
        for e in SIDES:
            t = data["timing"][e]["algorithm_only_correct"]
            timing.append([ref.upper(), e, t["flights"], f"{t['median_opensky_minus_algorithm_s']/60:+.2f} min",
                           f"{data['gap_concentration'][e]['60']['share_of_net_accuracy_gap_pct']:.1f}%"])
    parts += ["<p>When both methods' departure and arrival boundaries differ by at most one minute:</p>", table(["Reference", "Flights", "Takeoff gap", "Landing gap"], rows),
              "<p>On endpoints where algorithm NN is correct and OpenSky NN is wrong:</p>",
              table(["Reference", "Endpoint", "Endpoints", "Median OpenSky minus algorithm time", "Share of all net excess errors on >1 min shifts"], timing),
              "<p>Positive time differences mean OpenSky starts later; negative differences mean it ends earlier. The OpenSky reader maps firstseen and lastseen to takeoff and landing time. The NN only uses observations inside the segment, within ten minutes of the relevant endpoint. Candidates are the nearest 32 airports within 200 km of the first usable endpoint observation (working backward for arrival). Boundary shifts therefore change the observations, candidates and NN features.</p>"]
    if "intervention" in report:
        audit = report["intervention"]
        parts += ["<h2>Controlled boundary intervention</h2>", f"<p>{html.escape(audit['scope'])}</p>"]
        rows = []
        for ref in ("bts", "sfdps"):
            for e in SIDES:
                variants = {r["variant"]: r for r in audit["algorithm_correct_opensky_wrong"] if r["reference"] == ref and r["endpoint"] == e}
                native = variants["OO"]
                own = variants["AO" if e == "takeoff" else "OA"]
                rows.append([ref.upper(), e, native["flights"], variants["AA"]["correct"], own["correct"],
                             native["flights"] - native["gold_in_candidates"], native["gold_in_candidates"]])
        parts += [table(["Reference", "Endpoint", "Algorithm correct / OpenSky wrong", "Fixed by both boundaries", "Fixed by relevant boundary only", "Gold absent from OpenSky candidates", "Gold present but ranked wrong"], rows),
                  "<p>These intervention counts refer to the selected error cases, not whole-cohort accuracies. Full candidate lists and fresh predictions are saved in intervention_details.parquet.</p>",
                  table(["Variant", "Endpoint", "Selection", "Rerun", "Exactly reproduce cache"], [[r[k] for k in ("variant", "endpoint", "selection", "flights", "reproduced_cached_prediction")] for r in audit["baseline_reproduction"]])]
    parts += ["<h2>Interpretation and limits</h2>",
              "<p>This is primarily an endpoint-boundary problem in the shared cohort. The NN receives different trajectories even when the matcher calls the segments the same flight. OpenSky's exclusive flights have lower accuracy, but removing them leaves almost the entire gap.</p>",
              "<p>This does not mean every OpenSky timestamp is worse. Against BTS, OpenSky has a smaller median absolute landing-time error on the full shared cohort; large earlier arrival cutoffs are concentrated in the airport errors. Selecting only similar boundaries changes the cohort, so that comparison alone is correlational. The fixed-input intervention is the direct test.</p>",
              f"<p>On {report['cross_reference']['shared_segment_pairs']:,} identical source-segment pairs covered by both references, SFDPS landing times are a median {report['cross_reference']['median_sfdps_minus_bts_seconds']['landing']:.0f} seconds later than BTS. Reference timing errors should therefore be interpreted separately from the airport-label comparison.</p>",
              f"<p>The one-hour matching tolerance allows {report['references']['sfdps']['nonoverlapping_source_segments']} nonoverlapping SFDPS segment pairs ({report['references']['bts']['nonoverlapping_source_segments']} BTS pairs); invalid single-boundary hybrids are retained and flagged, not treated as valid flight segments. BTS and SFDPS overlap substantially and are not independent replications. Gold labels can themselves be wrong. Null predicted or gold airports count as errors, consistent with the evaluator.</p>",
              "<p>Main results allow observations through overnight arrivals. Target-day-only algorithm results are included separately in summary.json because truncating at midnight changes landing accuracy. Source caches and production code were not modified.</p>",
              "<h2>Provenance and reproduction</h2>",
              f"<p>{html.escape(report['method'])}</p>", f"<p>Model: <code>{html.escape(report['model_path'])}</code><br>SHA-256: <code>{report['model_sha256']}</code></p>",
              "<p>Evaluation cache v12; algorithm NN v9, OpenSky NN v4; BTS v16, SFDPS v14. Complete cache paths and fingerprints are in summary.json.</p>",
              "<pre>.venv/bin/python scripts/paper/investigate_opensky_nn_boundaries.py --rerun</pre>",
              "<p>Files: <a href='airport_accuracy.csv'>accuracy table</a> · <a href='summary.json'>complete statistics and provenance</a> · <a href='shared_flights.csv'>shared flight set</a> · <a href='boundary_interventions.csv'>boundary interventions</a></p>"]
    (outdir / "report.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><title>OpenSky NN boundary audit</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#182c3c;max-width:1150px;margin:40px auto;padding:0 24px}h1{font-size:32px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #dce4ea}th{background:#edf3f6}code,pre{font-size:12px;overflow-wrap:anywhere}pre{padding:15px;background:#edf3f6}</style>" + "".join(parts) + "</html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-03-01")
    parser.add_argument("--output", type=Path, default=Path("reports/opensky_nn_attribution_2026-03-01"))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--summarize-rerun", action="store_true", help="Summarize already completed intervention predictions")
    args = parser.parse_args()
    day = date.fromisoformat(args.date)
    args.output.mkdir(parents=True, exist_ok=True)
    manifests, shared, summary = [], [], {}
    for gold in ("bts", "sfdps"):
        a, gold_n = read_comparison(AlgorithmNNFlightSource(), gold, day, "opensky", manifests)
        o, other_n = read_comparison(OpenSkyNNFlightSource(), gold, day, "none", manifests)
        assert gold_n == other_n
        d = cohort(a, o).with_columns(pl.lit(gold).alias("reference"))
        shared.append(d)
        summary[gold] = summarize(a, o, d, gold_n)
        # Show the additional observation-window confound separately.
        a_day, day_n = read_comparison(replace(AlgorithmNNFlightSource(), target_day_adsb=True), gold, day, "opensky", manifests)
        assert day_n == gold_n
        summary[gold]["algorithm_target_day_only"] = summarize(a_day, o, cohort(a_day, o), gold_n)
        print(gold, "shared flights:", d.height, "airport accuracy gaps (pp):",
              {e: round(summary[gold]["shared"]["paired"][e]["algorithm_advantage_pp"], 3) for e in SIDES}, flush=True)
    shared = pl.concat(shared)
    shared.write_parquet(args.output / "shared_flights.parquet")
    shared.write_csv(args.output / "shared_flights.csv")
    cross = shared.filter(pl.col("reference") == "bts").select(*PAIR, *[f"gold_{e}_time" for e in SIDES]).join(
        shared.filter(pl.col("reference") == "sfdps").select(*PAIR, *[f"gold_{e}_time" for e in SIDES]), on=PAIR, suffix="_sfdps", validate="1:1")
    cross_summary = {"shared_segment_pairs": cross.height, "median_sfdps_minus_bts_seconds": cross.select(
        *[((pl.col(f"gold_{e}_time_sfdps") - pl.col(f"gold_{e}_time")).dt.total_milliseconds() / 1000).median().alias(e) for e in SIDES]).row(0, named=True)}
    from flights.airport_nn.source import DEFAULT_MODEL_PATH
    report = {"date": str(day), "method": "Join time-only matches by reference ICAO and both reference endpoint times; same OpenSky observation provider, current default NN, full overnight coverage. Null airports count as errors. Standard evaluator uses both errors <=60 minutes, gold departure 01:00-23:00 UTC.",
              "model_path": str(DEFAULT_MODEL_PATH.resolve()), "model_sha256": hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest(),
              "inputs": manifests, "references": summary, "cross_reference": cross_summary}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.rerun:
        rerun(shared, args.output, day)
    if args.rerun or args.summarize_rerun:
        report["intervention"] = summarize_interventions(shared, args.output)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report(report, args.output)


if __name__ == "__main__":
    main()
