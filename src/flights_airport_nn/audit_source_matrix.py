"""Audit source coverage and FAA-label disagreements using wider time windows."""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from airports.airport_lookup import AirportLookup
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import flights_algorithm_output_path
from data_engineering.flights.sfdps_to_flights import sfdps_flights_parquet_path
from flights_airport_nn.broader_evaluation import load_catalog, match_by_time

TYPES = ("large_airport", "medium_airport", "small_airport")
COLS = [
    "icao",
    "time",
    "lat",
    "lon",
    "baro_altitude_ft",
    "ground_speed_kt",
    "on_ground",
]
_CATALOG = {}
_CODES = {}


def initialize():
    _CATALOG.update(load_catalog())
    AirportLookup()
    for ident, airport in _CATALOG.items():
        if airport["type"] not in TYPES:
            continue
        for column in ("ident", "local_code", "gps_code", "icao_code", "iata_code"):
            code = airport.get(column)
            if code:
                _CODES.setdefault(code, set()).add(ident)


def distances(lat, lon, airport):
    lat = np.radians(lat)
    lon = np.radians(lon)
    alat = np.radians(float(airport["latitude_deg"]))
    alon = np.radians(float(airport["longitude_deg"]))
    h = (
        np.sin((lat - alat) / 2) ** 2
        + np.cos(lat) * np.cos(alat) * np.sin((lon - alon) / 2) ** 2
    )
    return 12742 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def evidence(arrays, ident, lo, hi):
    if ident not in _CATALOG or hi <= lo:
        return {
            "ident": ident,
            "strong": False,
            "points": 0,
            "span_s": 0.0,
            "closest_km": None,
        }
    airport = _CATALOG[ident]
    dist = distances(arrays["lat"][lo:hi], arrays["lon"][lo:hi], airport)
    elevation = (
        float(airport["elevation_ft"]) if airport.get("elevation_ft") else np.nan
    )
    low = (np.abs(arrays["alt"][lo:hi] - elevation) <= 300) & (
        arrays["speed"][lo:hi] <= 100
    )
    support = (dist <= 2) & (arrays["ground"][lo:hi] | low)
    times = arrays["time"][lo:hi][support]
    span = float((times[-1] - times[0]) / 1000) if len(times) else 0.0
    return {
        "ident": ident,
        "strong": bool(len(times) >= 3 and span >= 10),
        "points": len(times),
        "span_s": span,
        "closest_km": float(dist.min()),
    }


def audit_part(source, endpoints, path):
    messages = (
        read_adsb(
            date(2026, 3, 1),
            icaos=endpoints["icao"].unique().to_list(),
            columns=COLS,
            source=source,
            lazy=True,
        )
        .filter(
            pl.col("lat").is_finite()
            & pl.col("lon").is_finite()
            & (pl.col("lat").abs() <= 90)
            & (pl.col("lon").abs() <= 180)
        )
        .unique(subset=["icao", "time"])
        .sort(["icao", "time"])
        .collect(engine="streaming")
    )
    tracks = {}
    for (icao,), frame in messages.partition_by("icao", as_dict=True).items():
        tracks[icao] = {
            "time": frame["time"].dt.epoch("ms").to_numpy(),
            "lat": frame["lat"].to_numpy(),
            "lon": frame["lon"].to_numpy(),
            "alt": frame["baro_altitude_ft"].cast(pl.Float64).to_numpy(),
            "speed": frame["ground_speed_kt"].cast(pl.Float64).to_numpy(),
            "ground": frame["on_ground"].fill_null(False).to_numpy(),
        }
    records = []
    for row in endpoints.iter_rows(named=True):
        track = tracks.get(row["icao"])
        record = {
            k: row[k]
            for k in (
                "flight_id",
                "endpoint",
                "icao",
                "truth",
                "raw_label",
                "airport_type",
                "audit_needed",
            )
        }
        record.update(
            source=source,
            faa_time=row["faa_time"],
            segment_time=row["segment_time"],
            observation_gap_s=None,
            faa_nearest="",
            faa_correct=False,
            outside_segment=False,
            window_points=0,
            truth_strong=False,
            truth_closest_km=None,
            supported_other=[],
            candidate_evidence=[],
        )
        if track is None:
            records.append(record)
            continue
        time = row["faa_time_ms"]
        lo, hi = np.searchsorted(track["time"], [time - 900_000, time + 900_000])
        if hi <= lo:
            records.append(record)
            continue
        ix = lo + int(np.argmin(np.abs(track["time"][lo:hi] - time)))
        nearest = (
            AirportLookup().getClosestAirport(track["lat"][ix], track["lon"][ix]).ident
        )
        record.update(
            observation_gap_s=float(abs(track["time"][ix] - time) / 1000),
            faa_nearest=nearest,
            faa_correct=nearest == row["truth"],
            outside_segment=bool(
                track["time"][ix] < row["segment_start_ms"]
                or track["time"][ix] > row["segment_end_ms"]
            ),
            window_points=int(hi - lo),
        )
        truth = evidence(track, row["truth"], lo, hi)
        record.update(
            truth_strong=truth["strong"], truth_closest_km=truth["closest_km"]
        )
        if row["audit_needed"]:
            candidates = (
                set(row["candidate_airports"])
                | {nearest, row["truth"]}
                | _CODES.get(row["raw_label"], set())
            )
            ground = np.flatnonzero(
                track["ground"][lo:hi] | (track["speed"][lo:hi] < 100)
            )
            if len(ground):
                for index in (
                    ground[
                        np.unique(
                            np.linspace(
                                0, len(ground) - 1, min(12, len(ground))
                            ).astype(int)
                        )
                    ]
                    + lo
                ):
                    candidates.add(
                        AirportLookup()
                        .getClosestAirport(track["lat"][index], track["lon"][index])
                        .ident
                    )
            ev = [
                evidence(track, ident, lo, hi) for ident in sorted(candidates) if ident
            ]
            record["candidate_evidence"] = ev
            others = []
            for e in ev:
                if not e["strong"] or e["ident"] == row["truth"]:
                    continue
                airport = _CATALOG[e["ident"]]
                sep = float(
                    distances(
                        float(airport["latitude_deg"]),
                        float(airport["longitude_deg"]),
                        _CATALOG[row["truth"]],
                    )
                )
                if sep >= 10:
                    others.append(e["ident"])
            record["supported_other"] = others
        records.append(record)
    frame = pl.DataFrame(records, infer_schema_length=None).with_columns(
        pl.col("observation_gap_s").cast(pl.Float64),
        pl.col("truth_closest_km").cast(pl.Float64),
        pl.col("supported_other").cast(pl.List(pl.String)),
    )
    temporary = path.with_suffix(".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)
    return source, path.name, frame.height


def write_report(output, summary, suspects):
    rows = [
        "# Coverage and FAA-label audit, 2026-03-01",
        "",
        "The frozen source comparison clips both networks to ADSB.lol flight segments and requires an ADSB.lol detection. It cannot measure the full coverage advantage of ADSBx. This audit retains the same 48,625 scored endpoints but reads each aircraft's full-day observations.",
        "",
        "The diagnostic selects the valid observation closest in time to the FAA departure/arrival timestamp, within +/-15 minutes, then chooses its nearest airport. It uses supplied event times: these are not replacement deployment scores, and the NN has not been rerun with these windows.",
        "",
        "| FAA label group | Endpoints | ADSB.lol nearest at FAA time | ADSBx nearest at FAA time |",
        "|---|---:|---:|---:|",
    ]
    for group in ("all", *TYPES):
        lol = summary["sources"]["adsblol"][group]
        x = summary["sources"]["adsbx"][group]
        rows.append(
            f"| {group} | {lol['endpoints']:,} | {lol['nearest_at_faa_time_accuracy']:.2%} | {x['nearest_at_faa_time_accuracy']:.2%} |"
        )
    rows += [
        "",
        "| Coverage measure | ADSB.lol | ADSBx |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Observation within 60 seconds of FAA event", "observation_within_60s"),
        ("Chosen observation outside original segment", "outside_original_segment"),
        ("Sustained ground evidence at labeled airport", "truth_ground_corroborated"),
    ):
        rows.append(
            f"| {label} | {summary['sources']['adsblol']['all'][key]:,} | {summary['sources']['adsbx']['all'][key]:,} |"
        )
    rows += [
        "",
        "Ground corroboration requires at least three observations spanning ten seconds within 2 km of the airport center, with on_ground set OR altitude within 300 ft of airport elevation and speed <=100 knots. Absence of this evidence does not establish that a label is wrong, particularly at airports with distant runways.",
        "",
        f"Among {summary['audited_disagreement_endpoints']:,} endpoints where any frozen method disagrees, {summary['disagreement_truth_supported_by_either_source']:,} have ground corroboration for the label in at least one network. {suspects.height:,} have corroboration for the same alternative airport in both networks, at least 10 km from the label, and none for the label. These flag label, timing, segmentation or aircraft-association problems; they are not automatic corrections.",
        "",
        "The local FIXM NAS schemas define departurePoint/arrivalPoint as the initial/final entity of the ATC route, without requiring an airport (NasDeparture.xsd:88 and NasArrival.xsd:89). read_sfdps_logs.py extracts those fields; broader_evaluation.resolve_labels passes them to the worldwide airport lookup. This permits unrelated IATA/local-code matches. Examples below show both-network ground evidence, not manually certified replacement labels.",
        "",
        "| Flight | Endpoint | Raw code | Resolved label | Airport supported in both networks |",
        "|---|---|---|---|---|",
    ]
    examples = suspects.filter(
        pl.col("flight_id").is_in(
            [
                "a5cbeb_2026-03-01_00-28",
                "a6adef_2026-03-01_18-11",
                "a60217_2026-03-01_14-23",
            ]
        )
    )
    for row in examples.iter_rows(named=True):
        rows.append(
            f"| {row['flight_id']} | {row['endpoint']} | {row['raw_label']} | {row['truth']} | {', '.join(row['shared_other'])} |"
        )
    rows += [
        "",
        "VTU resolves to MUVT (Cuba), RZS to OPSW (Pakistan), and BEN to AYBH (Papua New Guinea), while the corroborated airports above are in the United States.",
    ]
    bts_path = output / "bts_crosscheck.json"
    if bts_path.exists():
        bts = json.loads(bts_path.read_text())
        rows += [
            "",
            f"The separately saved BTS cross-check agrees with FAA on {bts['agree']:,} of {bts['overlap_endpoints']:,} overlapping endpoints. It supports the FAA label on all {bts['adsblol']['nn_disagreements_in_overlap']} ADSB.lol NN disagreements and {bts['adsbx']['nn_disagreements_in_overlap']} ADSBx NN disagreements in that overlap. This supports the commercial cohort, without validating the small-airport cohort.",
        ]
    rows += [
        "",
        summary["limitations"],
        "",
        "The audit cannot establish a numerical accuracy ceiling or an adjusted NN accuracy. A revised benchmark needs airport-specific reference labels with provenance and flight boundaries derived independently for each network. Do not drop disagreements simply because predictions agree with each other.",
        "",
        "Files: endpoint_metadata.parquet retains original predictions and both sets of timestamps; endpoint_audit.parquet contains 97,250 source-endpoint evidence rows; label_or_timing_suspects.parquet preserves candidate evidence for review; summary.json contains exact diagnostic counts. Original paired_predictions.parquet and metrics.json in the parent directory remain frozen.",
    ]
    (output / "results.md").write_text("\n".join(rows) + "\n")


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = output.parent
    day = date(2026, 3, 1)
    labels = (
        pl.read_parquet(sfdps_flights_parquet_path(day))
        .filter(
            pl.col("icao").is_not_null()
            & pl.col("takeoff_time").is_not_null()
            & pl.col("landing_time").is_not_null()
            & (pl.col("landing_time") > pl.col("takeoff_time"))
        )
        .unique(
            subset=[
                "icao",
                "takeoff_time",
                "landing_time",
                "takeoff_airport_ident",
                "landing_airport_ident",
            ],
            maintain_order=True,
        )
    )
    matched, _ = match_by_time(
        pl.read_parquet(flights_algorithm_output_path(day, no_airports_model=True)),
        labels,
    )
    matched = matched.join(
        labels.with_row_index("_label_id").select(
            "_label_id",
            pl.col("takeoff_time").alias("faa_start"),
            pl.col("landing_time").alias("faa_end"),
        ),
        on="_label_id",
    )
    paired = pl.read_parquet(base / "paired_predictions.parquet").filter(
        pl.col("airport_type").is_in(TYPES) & (pl.col("truth") != "")
    )
    paired = paired.with_columns(
        (
            ~pl.all_horizontal(
                [
                    pl.col(f"{s}_{m}_correct")
                    for s in ("adsblol", "adsbx")
                    for m in ("nn", "nearest")
                ]
            )
        ).alias("audit_needed"),
        pl.concat_list(
            [
                pl.col(f"{s}_{m}")
                for s in ("adsblol", "adsbx")
                for m in ("nn", "nearest")
            ]
        ).alias("candidate_airports"),
    )
    endpoints = paired.join(
        matched.select(
            "flight_id", "takeoff_time", "landing_time", "faa_start", "faa_end"
        ),
        on="flight_id",
        validate="m:1",
    ).with_columns(
        pl.when(pl.col("endpoint") == "takeoff")
        .then(pl.col("faa_start"))
        .otherwise(pl.col("faa_end"))
        .alias("faa_time"),
        pl.when(pl.col("endpoint") == "takeoff")
        .then(pl.col("takeoff_time"))
        .otherwise(pl.col("landing_time"))
        .alias("segment_time"),
        pl.col("takeoff_time").dt.epoch("ms").alias("segment_start_ms"),
        pl.col("landing_time").dt.epoch("ms").alias("segment_end_ms"),
    )
    endpoints = endpoints.with_columns(
        pl.col("faa_time").dt.epoch("ms").alias("faa_time_ms")
    )
    endpoints.write_parquet(output / "endpoint_metadata.parquet")
    aircraft = endpoints["icao"].unique().sort().to_list()
    tasks = []
    for source in ("adsblol", "adsbx"):
        for start in range(0, len(aircraft), 256):
            path = output / f"{source}-part-{start // 256:04d}.parquet"
            if not path.exists():
                tasks.append(
                    (
                        source,
                        endpoints.filter(
                            pl.col("icao").is_in(aircraft[start : start + 256])
                        ),
                        path,
                    )
                )
    print(
        f"Auditing {paired.height} endpoints; union of disagreements {paired.filter(pl.col('audit_needed')).height}; {len(tasks)} batches",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=3,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
    ) as pool:
        fs = [pool.submit(audit_part, *task) for task in tasks]
        for i, future in enumerate(as_completed(fs), 1):
            print(i, len(tasks), future.result(), flush=True)
    data = pl.concat(
        [pl.read_parquet(p) for p in sorted(output.glob("*-part-*.parquet"))],
        how="diagonal_relaxed",
    )
    data.write_parquet(output / "endpoint_audit.parquet")
    summary = {
        "scored_endpoints": paired.height,
        "audited_disagreement_endpoints": paired.filter(pl.col("audit_needed")).height,
        "sources": {},
    }
    for source in ("adsblol", "adsbx"):
        frame = data.filter(pl.col("source") == source)
        if frame.height != paired.height:
            raise ValueError("Audit endpoint count differs")
        summary["sources"][source] = {}
        for group in ("all", *TYPES):
            sub = (
                frame
                if group == "all"
                else frame.filter(pl.col("airport_type") == group)
            )
            summary["sources"][source][group] = {
                "endpoints": sub.height,
                "nearest_at_faa_time_accuracy": sub["faa_correct"].mean(),
                "observation_within_60s": sub.filter(
                    pl.col("observation_gap_s") <= 60
                ).height,
                "outside_original_segment": sub["outside_segment"].sum(),
                "truth_ground_corroborated": sub["truth_strong"].sum(),
                "truth_min_distance_over_10km": sub.filter(
                    pl.col("truth_closest_km") > 10
                ).height,
                "median_window_points": sub["window_points"].median(),
            }
    joined = data.filter(pl.col("source") == "adsblol").join(
        data.filter(pl.col("source") == "adsbx"),
        on=["flight_id", "endpoint"],
        suffix="_x",
        validate="1:1",
    )
    both_other = joined.with_columns(
        pl.col("supported_other")
        .list.set_intersection(pl.col("supported_other_x"))
        .alias("shared_other")
    )
    suspects = both_other.filter(
        ~pl.col("truth_strong")
        & ~pl.col("truth_strong_x")
        & (pl.col("shared_other").list.len() > 0)
    )
    suspects.write_parquet(output / "label_or_timing_suspects.parquet")
    summary["both_sources_support_same_other_without_truth"] = suspects.height
    summary["disagreement_truth_supported_by_either_source"] = joined.filter(
        pl.col("audit_needed") & (pl.col("truth_strong") | pl.col("truth_strong_x"))
    ).height
    summary["limitations"] = (
        "FAA-time nearest is a timing diagnostic using supplied FAA timestamps. Sustained ground evidence within +/-15 minutes may reflect label, timing, segment or aircraft-association errors; it is not corrected ground truth. The two ADS-B networks can share receivers. No labels or frozen scores are changed."
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(output, summary, suspects)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
