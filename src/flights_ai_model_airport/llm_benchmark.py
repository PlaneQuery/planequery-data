from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, UTC
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Sequence

import numpy as np
import polars as pl

from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import add_flight_id_col
from data_engineering.utils import OUTPUT_DIR
from flights_ai_model_airport.features import (
    ADSB_SEQUENCE_COLUMNS,
    _candidate_airports,
    _resample_endpoint_sequence,
    get_labeled_segmented_flights,
)
from flights_ai_model_airport.flights_ai_model_utils import get_model_runs_dir
from utils import haversine


DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
OBSERVATIONS_PER_CASE = 16
STRICT_TCN_RUN = "2026-07-30_tcn_strict_sfdps_bts_adsblol_2days"
FULL_TEST_STRATUM_COUNTS = {
    ("takeoff", "baseline_correct"): 1946,
    ("takeoff", "baseline_wrong"): 263,
    ("landing", "baseline_correct"): 1815,
    ("landing", "baseline_wrong"): 349,
}


def default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return (
        OUTPUT_DIR
        / "data"
        / "experiments"
        / "airport-llm"
        / f"{timestamp}_{DEFAULT_MODEL}"
    )


def default_airport_split_path() -> Path:
    return get_model_runs_dir() / STRICT_TCN_RUN / "airport_split.json"


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sample_observation_indices(position_mask: np.ndarray) -> np.ndarray:
    valid = np.flatnonzero(position_mask)
    if len(valid) <= OBSERVATIONS_PER_CASE:
        return valid
    positions = np.linspace(
        0,
        len(valid) - 1,
        OBSERVATIONS_PER_CASE,
    ).round().astype(int)
    return valid[positions]


def _case_from_row(
    row: dict,
    *,
    endpoint: str,
    target_date: date,
    df_adsb: pl.DataFrame,
) -> dict | None:
    first_time = row["first_message_time"]
    last_time = row["last_message_time"]
    if first_time is None or last_time is None:
        return None
    df_flight_adsb = df_adsb.filter(
        (pl.col("icao") == row["icao"])
        & (pl.col("time") >= first_time)
        & (pl.col("time") <= last_time)
    )
    endpoint_time = first_time if endpoint == "takeoff" else last_time
    sequence = _resample_endpoint_sequence(
        df_flight_adsb,
        endpoint_time=endpoint_time,
        endpoint=endpoint,
    )
    if sequence is None:
        return None

    baseline_ident = row[f"{endpoint}_airport_ident"]
    target_ident = row[f"target_{endpoint}_airport_ident"]
    candidates = _candidate_airports(
        float(sequence["lat"][0]),
        float(sequence["lon"][0]),
        baseline_ident,
    )
    observation_indices = _sample_observation_indices(
        sequence["position_mask"]
    )
    observations = []
    for index in observation_indices:
        on_ground = sequence["on_ground"][index]
        observations.append(
            {
                "offset_minutes": round(
                    float(sequence["offset_seconds"][index]) / 60.0,
                    2,
                ),
                "lat": round(float(sequence["lat"][index]), 5),
                "lon": round(float(sequence["lon"][index]), 5),
                "baro_altitude_ft": _json_value(
                    float(sequence["altitude"][index])
                ),
                "ground_speed_kt": _json_value(
                    float(sequence["speed"][index])
                ),
                "track_deg": _json_value(float(sequence["track"][index])),
                "on_ground": (
                    None if on_ground < 0 else bool(on_ground)
                ),
            }
        )

    endpoint_lat = float(sequence["lat"][0])
    endpoint_lon = float(sequence["lon"][0])
    candidate_rows = [
        {
            "ident": airport.ident,
            "type": airport.type.value,
            "lat": round(airport.lat, 5),
            "lon": round(airport.lon, 5),
            "elevation_ft": airport.elevation_ft,
            "endpoint_distance_km": round(
                haversine(
                    endpoint_lat,
                    endpoint_lon,
                    airport.lat,
                    airport.lon,
                ),
                2,
            ),
            "baseline": airport.ident == baseline_ident,
        }
        for airport in candidates
    ]
    case_id = (
        f"{target_date.isoformat()}_{row['flight_id']}_{endpoint}"
        .replace(":", "-")
    )
    return {
        "case_id": case_id,
        "date": target_date.isoformat(),
        "flight_id": row["flight_id"],
        "endpoint": endpoint,
        "offset_meaning": (
            "minutes after the first flight message"
            if endpoint == "takeoff"
            else "minutes before the last flight message"
        ),
        "observations": observations,
        "candidates": candidate_rows,
        "baseline_airport_ident": baseline_ident,
        "target_airport_ident": target_ident,
        "baseline_correct": baseline_ident == target_ident,
        "target_in_candidates": target_ident
        in {candidate["ident"] for candidate in candidate_rows},
        "label_source": row["label_source"],
    }


def prepare_cases(
    dates: Sequence[date],
    *,
    unseen_test_airports: set[str],
    sample_per_stratum: int,
    seed: int,
) -> list[dict]:
    selected_frames = []
    for target_date in dates:
        df = add_flight_id_col(get_labeled_segmented_flights(target_date))
        for endpoint in ("takeoff", "landing"):
            target_column = f"target_{endpoint}_airport_ident"
            part = (
                df
                .filter(pl.col(target_column).is_in(unseen_test_airports))
                .with_columns(
                    pl.lit(target_date).alias("_date"),
                    pl.lit(endpoint).alias("_endpoint"),
                    (
                        pl.col(f"{endpoint}_airport_ident")
                        == pl.col(target_column)
                    ).alias("_baseline_correct"),
                )
            )
            selected_frames.append(part)
    eligible = pl.concat(selected_frames)

    selected = []
    for endpoint in ("takeoff", "landing"):
        for baseline_correct in (True, False):
            stratum = eligible.filter(
                (pl.col("_endpoint") == endpoint)
                & (pl.col("_baseline_correct") == baseline_correct)
            )
            count = min(sample_per_stratum, stratum.height)
            if count == 0:
                continue
            selected.append(
                stratum.sample(
                    n=count,
                    shuffle=True,
                    seed=seed
                    + (0 if endpoint == "takeoff" else 10)
                    + (0 if baseline_correct else 1),
                )
            )
    df_selected = pl.concat(selected).sort(
        "_date",
        "icao",
        "first_message_time",
        "_endpoint",
    )

    cases = []
    for target_date_value, df_date in df_selected.group_by(
        "_date",
        maintain_order=True,
    ):
        target_date = target_date_value[0]
        icaos = df_date.get_column("icao").unique().to_list()
        df_adsb = read_adsb(
            target_date,
            columns=ADSB_SEQUENCE_COLUMNS,
            icaos=icaos,
            source="adsblol",
        )
        for row in df_date.iter_rows(named=True):
            case = _case_from_row(
                row,
                endpoint=row["_endpoint"],
                target_date=target_date,
                df_adsb=df_adsb,
            )
            if case is not None:
                cases.append(case)
    return cases


def _public_case(case: dict) -> dict:
    return {
        key: value
        for key, value in case.items()
        if key
        not in {
            "target_airport_ident",
            "baseline_correct",
            "target_in_candidates",
            "label_source",
            "date",
            "flight_id",
        }
    }


def build_prompt(cases: Sequence[dict]) -> str:
    public_cases = [_public_case(case) for case in cases]
    return (
        "You are identifying the takeoff or landing airport for already "
        "segmented flights. Do not use tools, browse, or inspect files. Use "
        "only the cases below. Each case is independent.\n\n"
        "For takeoff, observations start at the first flight message and "
        "offset increases forward in time. For landing, observations start at "
        "the last flight message and offset increases backward in time. Track "
        "is the aircraft's actual broadcast heading. Infer the airport from "
        "position, motion, altitude relative to airport elevation, and whether "
        "the trajectory points away from or toward the candidate. The candidate "
        "marked baseline is the current nearest-airport heuristic; keep it when "
        "the evidence supports it, but correct it when the trajectory supports "
        "another candidate.\n\n"
        "Return exactly one prediction for every case_id. airport_ident must be "
        "one of that case's candidate idents. Keep each rationale under 240 "
        "characters.\n\n"
        f"CASES_JSON:\n{json.dumps(public_cases, separators=(',', ':'))}"
    )


def run_codex_batch(
    cases: Sequence[dict],
    *,
    model: str,
    reasoning_effort: str,
    schema_path: Path,
    audit_dir: Path,
    batch_index: int,
) -> tuple[list[dict], dict]:
    prompt = build_prompt(cases)
    prompt_path = audit_dir / f"batch-{batch_index:03d}-prompt.txt"
    prompt_path.write_text(prompt)
    output_path = audit_dir / f"batch-{batch_index:03d}-response.json"
    events_path = audit_dir / f"batch-{batch_index:03d}-events.jsonl"
    stderr_path = audit_dir / f"batch-{batch_index:03d}-stderr.txt"

    command = [
        "codex",
        "exec",
        "-",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--json",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
    ]
    start = time.monotonic()
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    duration_seconds = time.monotonic() - start
    events_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Codex batch {batch_index} failed with exit code "
            f"{completed.returncode}; see {stderr_path}"
        )
    response = json.loads(output_path.read_text())
    predictions = response["predictions"]

    expected_ids = {case["case_id"] for case in cases}
    received_ids = [prediction["case_id"] for prediction in predictions]
    if len(received_ids) != len(set(received_ids)):
        raise ValueError(f"Codex batch {batch_index} returned duplicate case IDs")
    if set(received_ids) != expected_ids:
        raise ValueError(
            f"Codex batch {batch_index} case mismatch: expected "
            f"{sorted(expected_ids)}, received {sorted(received_ids)}"
        )
    candidates_by_case = {
        case["case_id"]: {
            candidate["ident"] for candidate in case["candidates"]
        }
        for case in cases
    }
    for prediction in predictions:
        if (
            prediction["airport_ident"]
            not in candidates_by_case[prediction["case_id"]]
        ):
            raise ValueError(
                f"Codex selected non-candidate airport "
                f"{prediction['airport_ident']!r} for "
                f"{prediction['case_id']}"
            )

    usage = {}
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage", {})
    batch_metadata = {
        "batch_index": batch_index,
        "case_count": len(cases),
        "duration_seconds": duration_seconds,
        "usage": usage,
    }
    return predictions, batch_metadata


def _wilson_interval(correct: int, count: int) -> tuple[float, float]:
    if count == 0:
        return 0.0, 0.0
    z = 1.96
    rate = correct / count
    denominator = 1 + z**2 / count
    center = (rate + z**2 / (2 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            rate * (1 - rate) / count + z**2 / (4 * count**2)
        )
        / denominator
    )
    return center - margin, center + margin


def evaluate_predictions(
    cases: Sequence[dict],
    predictions: Sequence[dict],
) -> tuple[list[dict], dict]:
    prediction_by_case = {
        prediction["case_id"]: prediction for prediction in predictions
    }
    result_rows = []
    for case in cases:
        prediction = prediction_by_case[case["case_id"]]
        result_rows.append(
            {
                "case_id": case["case_id"],
                "date": case["date"],
                "endpoint": case["endpoint"],
                "stratum": (
                    "baseline_correct"
                    if case["baseline_correct"]
                    else "baseline_wrong"
                ),
                "target_airport_ident": case["target_airport_ident"],
                "baseline_airport_ident": case[
                    "baseline_airport_ident"
                ],
                "llm_airport_ident": prediction["airport_ident"],
                "target_in_candidates": case["target_in_candidates"],
                "baseline_correct": case["baseline_correct"],
                "llm_correct": (
                    prediction["airport_ident"]
                    == case["target_airport_ident"]
                ),
                "llm_changed_baseline": (
                    prediction["airport_ident"]
                    != case["baseline_airport_ident"]
                ),
                "confidence": prediction["confidence"],
                "rationale": prediction["rationale"],
                "label_source": case["label_source"],
            }
        )

    grouped = defaultdict(list)
    for row in result_rows:
        grouped[(row["endpoint"], row["stratum"])].append(row)
    stratum_metrics = []
    estimated_correct = 0.0
    weighted_wilson_low = 0.0
    weighted_wilson_high = 0.0
    total_population = sum(FULL_TEST_STRATUM_COUNTS.values())
    for key in (
        ("takeoff", "baseline_correct"),
        ("takeoff", "baseline_wrong"),
        ("landing", "baseline_correct"),
        ("landing", "baseline_wrong"),
    ):
        rows = grouped[key]
        count = len(rows)
        correct = sum(row["llm_correct"] for row in rows)
        rate = correct / count if count else 0.0
        lower, upper = _wilson_interval(correct, count)
        population_count = FULL_TEST_STRATUM_COUNTS[key]
        estimated_correct += rate * population_count
        population_weight = population_count / total_population
        weighted_wilson_low += lower * population_weight
        weighted_wilson_high += upper * population_weight
        stratum_metrics.append(
            {
                "endpoint": key[0],
                "stratum": key[1],
                "sample_count": count,
                "llm_correct": correct,
                "llm_accuracy": rate,
                "wilson_95_low": lower,
                "wilson_95_high": upper,
                "changed_baseline_count": sum(
                    row["llm_changed_baseline"] for row in rows
                ),
                "target_candidate_recall": (
                    sum(row["target_in_candidates"] for row in rows) / count
                    if count
                    else 0.0
                ),
                "full_test_population_count": population_count,
            }
        )

    correct = sum(row["llm_correct"] for row in result_rows)
    correct_rows = [row for row in result_rows if row["llm_correct"]]
    incorrect_rows = [row for row in result_rows if not row["llm_correct"]]
    high_confidence_rows = [
        row for row in result_rows if row["confidence"] >= 0.9
    ]
    baseline_correct_rows = [
        row for row in result_rows if row["baseline_correct"]
    ]
    baseline_wrong_rows = [
        row for row in result_rows if not row["baseline_correct"]
    ]
    correctable_baseline_wrong_rows = [
        row for row in baseline_wrong_rows if row["target_in_candidates"]
    ]
    metrics = {
        "sample_count": len(result_rows),
        "balanced_sample_accuracy": correct / len(result_rows),
        "baseline_balanced_sample_accuracy": (
            sum(row["baseline_correct"] for row in result_rows)
            / len(result_rows)
        ),
        "prevalence_weighted_llm_accuracy_estimate": (
            estimated_correct / total_population
        ),
        "prevalence_weighted_wilson_low": weighted_wilson_low,
        "prevalence_weighted_wilson_high": weighted_wilson_high,
        "full_test_baseline_accuracy": (
            sum(
                count
                for (endpoint, stratum), count
                in FULL_TEST_STRATUM_COUNTS.items()
                if stratum == "baseline_correct"
            )
            / total_population
        ),
        "baseline_preservation_rate": (
            sum(row["llm_correct"] for row in baseline_correct_rows)
            / len(baseline_correct_rows)
        ),
        "baseline_error_correction_rate": (
            sum(row["llm_correct"] for row in baseline_wrong_rows)
            / len(baseline_wrong_rows)
        ),
        "baseline_error_change_rate": (
            sum(row["llm_changed_baseline"] for row in baseline_wrong_rows)
            / len(baseline_wrong_rows)
        ),
        "baseline_error_candidate_recall": (
            len(correctable_baseline_wrong_rows) / len(baseline_wrong_rows)
        ),
        "correction_rate_when_target_is_candidate": (
            sum(
                row["llm_correct"]
                for row in correctable_baseline_wrong_rows
            )
            / len(correctable_baseline_wrong_rows)
        ),
        "mean_confidence_when_correct": (
            sum(row["confidence"] for row in correct_rows)
            / len(correct_rows)
            if correct_rows
            else None
        ),
        "mean_confidence_when_incorrect": (
            sum(row["confidence"] for row in incorrect_rows)
            / len(incorrect_rows)
            if incorrect_rows
            else None
        ),
        "high_confidence_prediction_count": len(high_confidence_rows),
        "high_confidence_error_count": sum(
            not row["llm_correct"] for row in high_confidence_rows
        ),
        "high_confidence_accuracy": (
            sum(row["llm_correct"] for row in high_confidence_rows)
            / len(high_confidence_rows)
            if high_confidence_rows
            else None
        ),
        "strata": stratum_metrics,
    }
    return result_rows, metrics


def run_benchmark(
    *,
    dates: Sequence[date],
    airport_split_path: Path,
    output_dir: Path,
    sample_per_stratum: int,
    batch_size: int,
    model: str,
    reasoning_effort: str,
    seed: int,
    prepare_only: bool = False,
    evaluate_only: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.json"
    if cases_path.exists():
        cases = json.loads(cases_path.read_text())
        print(f"Loaded {len(cases)} prepared cases from {cases_path}")
    else:
        airport_split = json.loads(airport_split_path.read_text())
        cases = prepare_cases(
            dates,
            unseen_test_airports=set(
                airport_split["unseen_test_airports"]
            ),
            sample_per_stratum=sample_per_stratum,
            seed=seed,
        )
        cases_path.write_text(json.dumps(cases, indent=2) + "\n")
        print(f"Prepared {len(cases)} cases at {cases_path}")
    if prepare_only:
        return output_dir

    schema_path = Path(__file__).with_name("llm_response_schema.json")
    audit_dir = output_dir / "codex-audit"
    audit_dir.mkdir(exist_ok=True)
    raw_predictions_path = output_dir / "codex_predictions.json"
    existing_metrics_path = output_dir / "metrics.json"
    if evaluate_only:
        predictions = json.loads(raw_predictions_path.read_text())
        batch_metadata = (
            json.loads(existing_metrics_path.read_text()).get(
                "batch_metadata",
                [],
            )
            if existing_metrics_path.exists()
            else []
        )
    else:
        predictions = []
        batch_metadata = []
        for batch_index, start in enumerate(
            range(0, len(cases), batch_size),
            start=1,
        ):
            batch = cases[start : start + batch_size]
            print(
                f"Running Codex batch {batch_index} with {len(batch)} cases"
            )
            batch_predictions, metadata = run_codex_batch(
                batch,
                model=model,
                reasoning_effort=reasoning_effort,
                schema_path=schema_path,
                audit_dir=audit_dir,
                batch_index=batch_index,
            )
            predictions.extend(batch_predictions)
            batch_metadata.append(metadata)
            print(
                f"  completed in {metadata['duration_seconds']:.1f}s; "
                f"usage={metadata['usage']}"
            )
        raw_predictions_path.write_text(
            json.dumps(predictions, indent=2) + "\n"
        )

    result_rows, metrics = evaluate_predictions(cases, predictions)
    total_usage = defaultdict(int)
    for batch in batch_metadata:
        for key, value in batch.get("usage", {}).items():
            total_usage[key] += value
    metrics.update(
        {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "sample_per_stratum_requested": sample_per_stratum,
            "batch_size": batch_size,
            "seed": seed,
            "dates": [value.isoformat() for value in dates],
            "airport_split_path": str(airport_split_path),
            "batch_metadata": batch_metadata,
            "total_duration_seconds": sum(
                batch["duration_seconds"] for batch in batch_metadata
            ),
            "total_usage": dict(total_usage),
        }
    )
    pl.DataFrame(result_rows).write_csv(output_dir / "predictions.csv")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Codex CLI airport identification from ADS-B endpoint "
            "messages and nearby airport candidates."
        )
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["2026-02-01", "2026-03-01"],
    )
    parser.add_argument(
        "--airport-split-path",
        type=Path,
        default=default_airport_split_path(),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--sample-per-stratum", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        dates=[date.fromisoformat(value) for value in args.dates],
        airport_split_path=args.airport_split_path,
        output_dir=args.output_dir,
        sample_per_stratum=args.sample_per_stratum,
        batch_size=args.batch_size,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
        prepare_only=args.prepare_only,
        evaluate_only=args.evaluate_only,
    )
