from datetime import date, datetime
import json
import pickle
import re

import polars as pl
import pytest

from data_engineering.filters import pia_or_american_ladd_icao
from data_engineering.utils import OUTPUT_DIR
from pia.ground_truth import get_ground_truth_mappings

from data_engineering.flights.flight_type import FlightIndex, create_pia_evaluation_df, flights_df_to_flights, flights_to_flights_df, get_flights, make_pia_consistent
from pia.resolver import build_candidates, check_candidate_mapping_paths, resolve_pia_mappings
from pia.evaluation.main import evaluation
from pia.evaluation.pia_graph_0 import flights_0, flights_2, flights_3, test_mapping_dict_3, test_mapping_dict_0
from pia.main import main
from pia.evaluation.real_flights_0 import a039a2_flights

END_DATE = date(2026,8,24)
def test_simulator_flights_0():
    # TODO: What is the random assignment percentage
    from pia.evaluation.simulator_flights import get_simulated_flights_0

    df_flights_sim, gold_mapping = get_simulated_flights_0()
    mapping_dict, _ = resolve_pia_mappings(flights_df_to_flights(df_flights_sim))
    correct = 0
    total = 0
    for flight, real_icao in gold_mapping.items():
        if flight in mapping_dict:
            total += 1
            if mapping_dict[flight] == real_icao:
                correct += 1
    print(f"Resolved accuracy: {correct}/{total} = {correct/total:.2%}")

def test_real_flights_2(request):
    start_date = date(2024,1,1)
    end_date = END_DATE
    stats = pia_tester(start_date, end_date, correct_number=0, test_name=request.node.name,label=request.config.getoption("label"))
    print_real_flight_stats(stats)


def test_real_flights_3(request):
    start_date = date(2025,1,1)
    end_date = END_DATE
    stats = pia_tester(start_date, end_date, correct_number=0, test_name=request.node.name,label=request.config.getoption("label"),)
    print_real_flight_stats(stats)


def test_real_flights_4(request):
    start_date = date(2026,1,1)
    end_date = END_DATE
    stats = pia_tester(
        start_date,
        end_date,
        correct_number=0,
        test_name=request.node.name,
        label=request.config.getoption("label"),
    )
    print_real_flight_stats(stats)
    assert stats['pia_icao_resolved_accuracy'] >= 0.4

@pytest.mark.parametrize(
    ("start_date", "expected_stats"),
    [
        pytest.param(
            date(2026, 5, 1),
            {
                "correct_pia_icaos": 6,
                "resolved_pia_icaos": 23,
                "expected_pia_icaos": 30,
                "pia_icao_resolved_accuracy": 0.4,
                "pia_icao_coverage": 23 / 30,
                "pia_icao_overall_accuracy": 6 / 30,
                "unknown_label_pia_icaos": 93,
                "resolved_expected_flight_mappings": 391,
                "expected_flight_mappings": 454,
                "correct_flight_mappings": 122,
                "flight_coverage": 391 / 454,
                "flight_resolved_accuracy": 122 / 391,
                "flight_overall_accuracy": 122 / 454,
                "traceable_aircraft": 5,
                "expected_aircraft": 28,
                "traceability_index": 5 / 28,
            },
            id="may",
        ),
        pytest.param(
            date(2026, 6, 1),
            {
                "correct_pia_icaos": 4,
                "resolved_pia_icaos": 17,
                "expected_pia_icaos": 21,
                "pia_icao_resolved_accuracy": 4 / 17,
                "pia_icao_coverage": 17 / 21,
                "pia_icao_overall_accuracy": 4 / 21,
                "unknown_label_pia_icaos": 77,
                "resolved_expected_flight_mappings": 153,
                "expected_flight_mappings": 182,
                "correct_flight_mappings": 63,
                "flight_coverage": 153 / 182,
                "flight_resolved_accuracy": 63 / 153,
                "flight_overall_accuracy": 63 / 182,
                "traceable_aircraft": 4,
                "expected_aircraft": 20,
                "traceability_index": 4 / 20,
            },
            id="june",
        ),
    ],
)
def test_real_flights_5(request, start_date, expected_stats):
    end_date = END_DATE
    stats = pia_tester(
        start_date,
        end_date,
        correct_number=0,
        filter=False,
        test_name=request.node.name,
        label=request.config.getoption("label"),
    )
    print_real_flight_stats(stats)
    minimum_metrics = (
        "pia_icao_resolved_accuracy",

    )
    assert all(stats[key] >= expected_stats[key] for key in minimum_metrics)

def print_real_flight_stats(stats):
    print(
        "PIA ICAO resolved accuracy: "
        f"{stats['correct_pia_icaos']}/{stats['resolved_pia_icaos']} "
        f"= {stats['pia_icao_resolved_accuracy']:.2%}"
    )
    print(
        "Ground-truth PIA ICAO coverage: "
        f"{stats['resolved_pia_icaos']}/{stats['expected_pia_icaos']} "
        f"= {stats['pia_icao_coverage']:.2%}"
    )
    print(
        "All-observed PIA ICAO coverage: "
        f"{stats['resolved_observed_pia_icaos']}/"
        f"{stats['observed_pia_icaos']} "
        f"= {stats['observed_pia_icao_coverage']:.2%}"
    )

    print(
        "Observed PIA ICAOs assigned UNKNOWN_LABEL: "
        f"{stats['unknown_label_pia_icaos']}"
    )
    print(
        "Flight coverage: "
        f"{stats['resolved_expected_flight_mappings']}/"
        f"{stats['expected_flight_mappings']} "
        f"= {stats['flight_coverage']:.2%}"
    )
    print(
        "Flight resolved accuracy: "
        f"{stats['correct_flight_mappings']}/"
        f"{stats['resolved_expected_flight_mappings']} "
        f"= {stats['flight_resolved_accuracy']:.2%}"
    )
    print(
        "Flight overall accuracy: "
        f"{stats['correct_flight_mappings']}/{stats['expected_flight_mappings']} "
        f"= {stats['flight_overall_accuracy']:.2%}"
    )
    print(
        "Traceability: "
        f"{stats['traceable_aircraft']}/{stats['expected_aircraft']} aircraft "
        f"= {stats['traceability_index']:.2%}"
    )

def save_pia_mapping_audit(
    flights,
    gold_mapping,
    mappings,
    pia_to_real_icao,
    run_dir,
):
    """Write the chronological per-flight PIA mapping audit for one test run."""
    real_icaos = set(pia_to_real_icao.values())
    audit_rows = []
    for flight in flights:
        expected_icao = gold_mapping.get(flight)
        predicted_icao = mappings.get(flight)
        if expected_icao is None:
            mapping_status = "not_evaluated"
        elif predicted_icao is None:
            mapping_status = "missing_prediction"
        elif predicted_icao == expected_icao:
            mapping_status = "correct"
        else:
            mapping_status = "incorrect"

        audit_rows.append({
            "flight_id": flight.flight_id,
            "takeoff_time": flight.takeoff_time,
            "landing_time": flight.landing_time,
            "observed_icao": flight.icao,
            "pia_icao": flight.icao if flight.pia else None,
            # Visible real-ICAO flights are included as history anchors.
            "real_icao": expected_icao or (
                flight.icao if flight.icao in real_icaos else None
            ),
            "predicted_icao": predicted_icao,
            "correct": (
                predicted_icao == expected_icao
                if expected_icao is not None
                else None
            ),
            "mapping_status": mapping_status,
            "pia": flight.pia,
            "callsign": flight.callsign,
            "takeoff_airport_ident": flight.takeoff_airport_ident,
            "landing_airport_ident": flight.landing_airport_ident,
        })

    output_path = run_dir / "pia_mapping_audit.parquet"
    audit_schema = {
        "flight_id": pl.String,
        "takeoff_time": pl.Datetime,
        "landing_time": pl.Datetime,
        "observed_icao": pl.String,
        "pia_icao": pl.String,
        "real_icao": pl.String,
        "predicted_icao": pl.String,
        "correct": pl.Boolean,
        "mapping_status": pl.String,
        "pia": pl.Boolean,
        "callsign": pl.String,
        "takeoff_airport_ident": pl.String,
        "landing_airport_ident": pl.String,
    }
    (
        pl.DataFrame(audit_rows, schema=audit_schema)
        .sort("takeoff_time")
        .write_parquet(output_path)
    )
    print(f"Wrote PIA mapping audit: {output_path}")
    return output_path


def pia_tester(
    start_date: date,
    end_date: date,
    correct_number: int,
    filter=False,
    filter_known_mappings: bool = False,
    test_name=None,
    label=None,
    pia_location_radius_km: float = 10,
):
    """Run the PIA evaluation and save a per-flight mapping audit as Parquet.

    ``test_name`` controls the artifact directory; pytest callers can pass
    ``request.node.name``. When provided, ``label`` is appended to the
    timestamped run directory. Set ``filter_known_mappings`` to exclude known
    PIA and real ICAOs while resolving mappings.
    """
    safe_test_name = test_name or "pia_tester"
    safe_label = None
    if label:
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-._")
        if not safe_label:
            raise ValueError("label must contain at least one letter or number")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"test_pia_{safe_test_name}_{current_time}"
    if safe_label:
        run_name = f"{run_name}_{safe_label}"
    run_dir = (
        OUTPUT_DIR
        / "data"
        / "tests"
        / "test_pia"
        / safe_test_name
        / run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    df_mappings = get_ground_truth_mappings()
    pia_to_real_icao = dict(zip(df_mappings["pia_icao"], df_mappings["real_icao"]))
    df_flights = get_flights(start_date, end_date)
    df_flights = make_pia_consistent(df_flights)
    for pia_icao, real_icao in list(pia_to_real_icao.items()):
        n_pia_flights = len(df_flights.filter(pl.col("icao") == pia_icao))
        n_real_flights = len(df_flights.filter(pl.col("icao") == real_icao))
        if n_pia_flights == 0 or n_real_flights == 0:
            # print(f"Warning: No flights found for pia_icao {pia_icao} or real_icao {real_icao}")
            pia_to_real_icao.pop(pia_icao) 
    print(f"len(pia_to_real_icao): {len(pia_to_real_icao)}")
    print(pia_to_real_icao)
    if filter:
        df_flights = df_flights.filter(pl.col("icao").is_in(pia_to_real_icao.keys() | pia_to_real_icao.values()))
    else:
        df_flights = df_flights.filter(pl.col("icao").str.len_chars() == 6)
        df_flights = df_flights.filter(pia_or_american_ladd_icao())
    # icaos_to_filter = set(pia_to_real_icao.keys()) | set(pia_to_real_icao.values())
    # df_flights = df_flights.filter(~pl.col("icao").is_in(icaos_to_filter))
    pia_icao_count = df_flights.filter(pl.col("pia")).select("icao").n_unique()
    real_icao_count = df_flights.filter(~pl.col("pia")).select("icao").n_unique()
    total_icao_count = pia_icao_count + real_icao_count
    print(f"(Real), (PIA), Total ICAO counts in flights: ({real_icao_count}), ({pia_icao_count}), ({total_icao_count})")
    flights = flights_df_to_flights(df_flights)
    flight_index = FlightIndex(flights, pia_location_radius_km)
    # candidates = build_candidates(flight_index)
    # check_candidate_path_is_possible(pia_to_real_icao, flight_index, candidates)

    known_pia_mappings = pia_to_real_icao if filter_known_mappings else {}
    mappings, df_margins, df_mapping_stats = resolve_pia_mappings(
        flights,
        known_pia_mappings=known_pia_mappings,
    )
    gold_mapping = {
        flight: pia_to_real_icao[flight.icao]
        for flight in flights
        if flight.icao in pia_to_real_icao
    }
    known_mappings = {
        flight: real_icao
        for flight, real_icao in mappings.items()
        if flight in gold_mapping
    }
    stats = evaluation(known_mappings, gold_mapping)
    pia_icaos = {flight.icao for flight in flight_index.get_pia_flights()}
    resolved_pia_icaos = {flight.icao for flight in mappings}
    resolved_observed_pia_icaos = pia_icaos & resolved_pia_icaos
    stats["observed_pia_icaos"] = len(pia_icaos)
    stats["resolved_observed_pia_icaos"] = len(resolved_observed_pia_icaos)
    stats["observed_pia_icao_coverage"] = (
        len(resolved_observed_pia_icaos) / len(pia_icaos) if pia_icaos else 0
    )
    stats["unknown_label_pia_icaos"] = len(pia_icaos - resolved_observed_pia_icaos)

    print(f"Evaluation stats: {stats}")
    stats_path = run_dir / "evaluation_stats.json"
    with stats_path.open("w") as stats_file:
        json.dump(stats, stats_file, indent=2, sort_keys=True)
        stats_file.write("\n")
    print(f"Wrote evaluation stats: {stats_path}")

    save_pia_mapping_audit(
        flights,
        gold_mapping,
        mappings,
        pia_to_real_icao,
        run_dir,
    )
    mappings_path = run_dir / "mappings.pkl"
    with mappings_path.open("wb") as mappings_file:
        pickle.dump(mappings, mappings_file)
    print(f"Wrote PIA mappings: {mappings_path}")

    evaluation_df = create_pia_evaluation_df(flights, known_mappings, gold_mapping)
    evaluation_path = run_dir / "pia_evaluation_flights.parquet"
    evaluation_df.write_parquet(evaluation_path)
    print(f"Wrote flights DataFrame with real_icao column: {evaluation_path}")
    margins_path = run_dir / "margins.csv"
    df_margins.write_csv(margins_path)
    mapping_stats_path = run_dir / "mapping_stats.csv"
    df_mapping_stats.write_csv(mapping_stats_path)

    print(f"Wrote margins: {margins_path}")
    print(f"Wrote mapping stats: {mapping_stats_path}")
    print(f"Wrote PIA test artifacts: {run_dir}")
    return stats

def check_candidate_path_is_possible(pia_to_real_icao, flight_index, candidates):
    candidate_path_results = check_candidate_mapping_paths(
        candidates,
        flight_index,
        pia_to_real_icao,
    )
    possible_mappings = sum(candidate_path_results.values())
    print(
        "Ground-control mappings with a path in the candidate graph: "
        f"{possible_mappings}/{len(candidate_path_results)}"
    )
    for pia_icao, path_exists in candidate_path_results.items():
        if not path_exists:
            print(
                "No candidate path for ground-control mapping: "
                f"{pia_icao} -> {pia_to_real_icao[pia_icao]}"
            )
